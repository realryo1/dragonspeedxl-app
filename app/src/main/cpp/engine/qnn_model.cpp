#include "engine/qnn_model.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <android/log.h>
#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "GPU/QnnGpuGraph.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

namespace dsxl {
namespace {

size_t DtypeBytes(Qnn_DataType_t t) {
    switch (t) {
        case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32: return 4;
        case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_SFIXED_POINT_16: return 2;
        case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8: case QNN_DATATYPE_BOOL_8: case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8: return 1;
        case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64: return 8;
        default: return 0;
    }
}

// Qnn_Tensor_t の v1 / v2 の共通部分を触るためのアクセサ
const char* TName(const Qnn_Tensor_t& t) { return t.version == QNN_TENSOR_VERSION_2 ? t.v2.name : t.v1.name; }
Qnn_DataType_t TType(const Qnn_Tensor_t& t) { return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dataType : t.v1.dataType; }
uint32_t TRank(const Qnn_Tensor_t& t) { return t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank; }
uint32_t* TDims(const Qnn_Tensor_t& t) { return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions; }
void TSetDims(Qnn_Tensor_t& t, uint32_t* d) { if (t.version == QNN_TENSOR_VERSION_2) t.v2.dimensions = d; else t.v1.dimensions = d; }
void TSetName(Qnn_Tensor_t& t, const char* n) { if (t.version == QNN_TENSOR_VERSION_2) t.v2.name = n; else t.v1.name = n; }
void TSetBuf(Qnn_Tensor_t& t, void* data, size_t bytes) {
    Qnn_ClientBuffer_t cb{data, static_cast<uint32_t>(bytes)};
    if (t.version == QNN_TENSOR_VERSION_2) { t.v2.memType = QNN_TENSORMEMTYPE_RAW; t.v2.clientBuf = cb; }
    else { t.v1.memType = QNN_TENSORMEMTYPE_RAW; t.v1.clientBuf = cb; }
}

uint64_t NowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

TensorDesc MakeDesc(const Qnn_Tensor_t& src) {
    TensorDesc d;
    d.name = TName(src) ? TName(src) : "";
    d.tensor = src;
    uint32_t rank = TRank(src);
    d.dims.assign(TDims(src), TDims(src) + rank);
    TSetDims(d.tensor, d.dims.data());   // 以降は複製側の実体を指す
    TSetName(d.tensor, d.name.c_str());   // vector 再配置で無効にならないよう、使用直前に再設定する（Run 内）
    d.dtype = TType(src);
    size_t n = DtypeBytes(d.dtype);
    for (uint32_t x : d.dims) n *= x;
    d.bytes = n;
    return d;
}

void LogCb(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
    int prio = level <= QNN_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : level == QNN_LOG_LEVEL_WARN ? ANDROID_LOG_WARN : ANDROID_LOG_DEBUG;
    __android_log_vprint(prio, "dsxl-qnn", fmt, args);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
bool QnnBackendLib::Init(const std::string& lib_dir, std::string* err, bool gpu) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    is_gpu = gpu;
    const char* backend_so = gpu ? "libQnnGpu.so" : "libQnnHtp.so";
    std::string htp = lib_dir + "/" + backend_so, sy = lib_dir + "/libQnnSystem.so";
    htp_lib_ = dlopen(htp.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!htp_lib_) return fail(std::string("dlopen ") + backend_so + " 失敗: " + dlerror());
    sys_lib_ = dlopen(sy.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!sys_lib_) return fail(std::string("dlopen libQnnSystem.so 失敗: ") + dlerror());

    using GetProv = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
    auto getProv = reinterpret_cast<GetProv>(dlsym(htp_lib_, "QnnInterface_getProviders"));
    if (!getProv) return fail("QnnInterface_getProviders が見つかりません");
    const QnnInterface_t** provs = nullptr; uint32_t n = 0;
    if (getProv(&provs, &n) != QNN_SUCCESS || !provs || n == 0) return fail("QNN インターフェースを取得できません");
    bool ok = false;
    for (uint32_t i = 0; i < n; i++) {
        if (provs[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
            provs[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) { qnn = provs[i]->QNN_INTERFACE_VER_NAME; iface = *provs[i]; ok = true; break; }
    }
    if (!ok) return fail("対応する QNN API の版がありません");

    using GetSysProv = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
    auto getSys = reinterpret_cast<GetSysProv>(dlsym(sys_lib_, "QnnSystemInterface_getProviders"));
    if (!getSys) return fail("QnnSystemInterface_getProviders が見つかりません");
    const QnnSystemInterface_t** sprovs = nullptr; uint32_t sn = 0;
    if (getSys(&sprovs, &sn) != QNN_SUCCESS || !sprovs || sn == 0) return fail("QNN System インターフェースを取得できません");
    ok = false;
    for (uint32_t i = 0; i < sn; i++) {
        if (sprovs[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
            sprovs[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) { sys = sprovs[i]->QNN_SYSTEM_INTERFACE_VER_NAME; ok = true; break; }
    }
    if (!ok) return fail("対応する QNN System API の版がありません");

    if (qnn.logCreate) qnn.logCreate(LogCb, QNN_LOG_LEVEL_INFO, &log_);
    if (qnn.backendCreate(log_, nullptr, &backend) != QNN_SUCCESS) return fail("backendCreate 失敗");
    if (gpu) return true;   // GPU は device 作成も burst 設定も不要
    if (qnn.deviceCreate) {
        Qnn_ErrorHandle_t e = qnn.deviceCreate(log_, nullptr, &device);
        if (e != QNN_SUCCESS && e != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) return fail("deviceCreate 失敗 (" + std::to_string(e) + ")");
    }
    // （起動時には投票しない。投票は生成のたびに頭で行い、終わったら ReleaseVotes で解除する。待機中は通常のクロックに戻す）
    return true;
}

void QnnBackendLib::Vote() {
    if (is_gpu || !qnn.deviceGetInfrastructure) return;
    std::lock_guard<std::mutex> lock(vote_mu_);
    QnnDevice_Infrastructure_t infra = nullptr;
    if (qnn.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) return;
    auto* h = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
    if (h->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    auto& pi = h->perfInfra;
    if (!pi.createPowerConfigId || !pi.setPowerConfig) return;
    // 電力設定の ID は、作ったスレッドからしか使えない（別のスレッドから setPowerConfig すると 13001 "Invalid contextId … use Id created by createPowerConfigId"）。
    // なので、投票するスレッドごとに ID を作る。スレッド id は使い回されうるので、失敗したら作り直してもう一度だけ試す。
    const std::thread::id tid = std::this_thread::get_id();
    uint32_t pid = 0;
    auto get_id = [&](bool fresh) -> bool {
        auto it = power_ids_.find(tid);
        if (it != power_ids_.end() && !fresh) { pid = it->second; return true; }
        if (it != power_ids_.end()) power_ids_.erase(it);
        if (pi.createPowerConfigId(0, 0, &pid) != QNN_SUCCESS) return false;
        power_ids_[tid] = pid;
        return true;
    };
    if (!get_id(false)) return;

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    dcvs.dcvsV3Config.contextId = pid;   // DCVS_V3 は contextId に、createPowerConfigId で作った ID を入れる必要がある（入れないと 13001 "Invalid contextId … use Id created by createPowerConfigId"）
    dcvs.dcvsV3Config.setDcvsEnable = 1;
    dcvs.dcvsV3Config.dcvsEnable = 0;  // DCVS を切って最大クロック固定
    dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvs.dcvsV3Config.setSleepLatency = 1;
    dcvs.dcvsV3Config.sleepLatency = 40;
    dcvs.dcvsV3Config.setSleepDisable = 1;
    dcvs.dcvsV3Config.sleepDisable = 1;
    dcvs.dcvsV3Config.setBusParams = 1;
    dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvs.dcvsV3Config.setCoreParams = 1;
    dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;

    QnnHtpPerfInfrastructure_PowerConfig_t poll = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    poll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    poll.rpcPollingTimeConfig = 9999;
    QnnHtpPerfInfrastructure_PowerConfig_t lat = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    lat.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    lat.rpcControlLatencyConfig = 100;

    // HMX（行列演算ユニット）のクロックを最大に固定する。DCVS_V3 の投票とは別で、同じ ID で投票する（無指定だと最大かどうか保証がない）。
    QnnHtpPerfInfrastructure_PowerConfig_t hmx = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    hmx.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
    hmx.hmxV2Config.hmxPickDefault = 0;
    hmx.hmxV2Config.hmxVoltageCornerMin = DCVS_EXP_VCORNER_MAX;
    hmx.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_MAX;
    hmx.hmxV2Config.hmxVoltageCornerMax = DCVS_EXP_VCORNER_MAX;
    hmx.hmxV2Config.hmxPerfMode = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

    const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, &poll, &lat, &hmx, nullptr};
    Qnn_ErrorHandle_t e = pi.setPowerConfig(pid, cfgs);
    if (e != QNN_SUCCESS && get_id(true)) { dcvs.dcvsV3Config.contextId = pid; e = pi.setPowerConfig(pid, cfgs); }   // 古い（別スレッドの）ID だったときのやり直し
    if (e != QNN_SUCCESS) {   // HMX の投票が通らない SoC / 版でも、従来の投票は入れる
        __android_log_print(ANDROID_LOG_WARN, "dsxl-qnn", "HTP power vote with HMX_V2 failed (code %d); retrying without it", static_cast<int>(e));
        const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs2[] = {&dcvs, &poll, &lat, nullptr};
        e = pi.setPowerConfig(pid, cfgs2);
        hmx_ = false;
    } else {
        hmx_ = true;
    }
    power_ = e == QNN_SUCCESS;
    __android_log_print(power_ ? ANDROID_LOG_INFO : ANDROID_LOG_WARN, "dsxl-qnn", "HTP power vote: %s (code %d, hmx %d, thread %zu, %zu ids)", power_ ? "ok" : "failed", static_cast<int>(e),
                        hmx_ ? 1 : 0, std::hash<std::thread::id>()(tid) % 100000, power_ids_.size());
    PinToPrimeCores();
}

// 呼び出しスレッド（HTP の完了待ちをするスレッド）を、最高周波数のコア（prime）に固定する。
// スリープからの復帰や別コアへの移動を避け、RPC polling の待ちを速いコアで行う。失敗したら何もしない。
void QnnBackendLib::PinToPrimeCores() {
    static const cpu_set_t mask = [] {
        cpu_set_t m;
        CPU_ZERO(&m);
        long best = 0;
        std::vector<long> freq;
        const int n = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
        for (int i = 0; i < n && i < CPU_SETSIZE; i++) {
            long f = 0;
            std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq");
            if (!(in >> f)) f = 0;
            freq.push_back(f);
            if (f > best) best = f;
        }
        if (best > 0) for (size_t i = 0; i < freq.size(); i++) if (freq[i] == best) CPU_SET(static_cast<int>(i), &m);
        return m;
    }();
    if (CPU_COUNT(&mask) == 0) return;
    const int r = sched_setaffinity(0, sizeof(mask), &mask);
    __android_log_print(r == 0 ? ANDROID_LOG_INFO : ANDROID_LOG_WARN, "dsxl-qnn", "pin to prime cores (%d cores): %s", CPU_COUNT(&mask), r == 0 ? "ok" : "failed");
}

void QnnBackendLib::ReleaseVotes(const std::atomic<int>* busy) {
    if (is_gpu || !device || !qnn.deviceGetInfrastructure) return;
    std::lock_guard<std::mutex> lock(vote_mu_);
    if (busy && busy->load() > 0) return;   // 次のジョブがもう始まっている（Begin は busy を上げてから Vote する）
    if (power_ids_.empty()) return;
    QnnDevice_Infrastructure_t infra = nullptr;
    if (qnn.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) return;
    auto* h = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
    if (h->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF || !h->perfInfra.destroyPowerConfigId) return;
    for (auto& kv : power_ids_) h->perfInfra.destroyPowerConfigId(kv.second);
    __android_log_print(ANDROID_LOG_INFO, "dsxl-qnn", "HTP power votes released (%zu ids)", power_ids_.size());
    power_ids_.clear();
    power_ = false;
}

bool QnnBackendLib::BuildGpuContextBinary(const std::string& dlc_path, const std::string& out_bin, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (!sys.systemDlcCreateFromFile || !sys.systemDlcComposeGraphs) return fail("この QNN System には DLC 読み込み機能がありません");
    size_t slash = dlc_path.find_last_of('/'), dot = dlc_path.find_last_of('.');
    const std::string gname = dlc_path.substr(slash == std::string::npos ? 0 : slash + 1,
                                              dot == std::string::npos || (slash != std::string::npos && dot < slash) ? std::string::npos
                                                  : dot - (slash == std::string::npos ? 0 : slash + 1));
    Qnn_LogHandle_t slog = nullptr;
    QnnSystemDlc_Handle_t dlc = nullptr;
    if (sys.systemLogCreate) sys.systemLogCreate(LogCb, QNN_LOG_LEVEL_WARN, &slog);
    if (sys.systemDlcCreateFromFile(slog, dlc_path.c_str(), &dlc) != QNN_SUCCESS) return fail("DLC を開けません: " + dlc_path);
    Qnn_ContextHandle_t ctx = nullptr;
    auto cleanup = [&]() {
        if (ctx) qnn.contextFree(ctx, nullptr);
        if (dlc) sys.systemDlcFree(dlc);
        if (slog && sys.systemLogFree) sys.systemLogFree(slog);
    };
    if (qnn.contextCreate(backend, nullptr, nullptr, &ctx) != QNN_SUCCESS) { cleanup(); return fail("GPU contextCreate 失敗"); }

    // 精度は FP16（重み・中間とも半精度。速度はこれで NPU と同程度になる）
    QnnGpuGraph_CustomConfig_t gcfg{};
    gcfg.precision = QNN_GPU_PRECISION_FP16;
    QnnGraph_Config_t cfg = QNN_GRAPH_CONFIG_INIT;
    cfg.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    cfg.customConfig = &gcfg;
    const QnnGraph_Config_t* cfgs[] = {&cfg, nullptr};
    QnnSystemDlc_GraphConfigInfo_t gci{};
    gci.version = QNN_SYSTEM_DLC_GRAPH_CONFIG_INFO_VERSION_1;
    gci.v1.graphName = gname.c_str();
    gci.v1.graphConfigs = cfgs;
    gci.v1.numConfigs = 1;
    const QnnSystemDlc_GraphConfigInfo_t* gcis[] = {&gci};

    QnnSystemContext_GraphInfo_t* gi = nullptr;
    uint32_t ng = 0;
    if (sys.systemDlcComposeGraphs(dlc, gcis, 1, backend, ctx, iface, QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1, &gi, &ng) != QNN_SUCCESS || !gi || ng < 1) {
        cleanup(); return fail("DLC から GPU グラフを作れません");
    }
    Qnn_GraphHandle_t graph = nullptr;
    if (qnn.graphRetrieve(ctx, gi[0].graphInfoV1.graphName, &graph) != QNN_SUCCESS) { cleanup(); return fail("GPU graphRetrieve 失敗"); }
    if (qnn.graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) { cleanup(); return fail("GPU graphFinalize 失敗"); }

    Qnn_ContextBinarySize_t sz = 0, written = 0;
    if (qnn.contextGetBinarySize(ctx, &sz) != QNN_SUCCESS || sz == 0) { cleanup(); return fail("GPU contextGetBinarySize 失敗"); }
    std::vector<uint8_t> buf(sz);
    if (qnn.contextGetBinary(ctx, buf.data(), sz, &written) != QNN_SUCCESS) { cleanup(); return fail("GPU contextGetBinary 失敗"); }
    cleanup();
    const std::string tmp = out_bin + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(written));
        if (!f) return fail("GPU キャッシュを書き込めません: " + tmp);
    }
    if (rename(tmp.c_str(), out_bin.c_str()) != 0) return fail("GPU キャッシュの名前変更に失敗");
    return true;
}

uint64_t QnnBackendLib::BusyNs() const {
    const uint64_t start = run_start_ns.load();
    return busy_ns.load() + (start ? NowNs() - start : 0);
}

QnnBackendLib::~QnnBackendLib() {
    if (!power_ids_.empty() && device && qnn.deviceGetInfrastructure) {
        QnnDevice_Infrastructure_t infra = nullptr;
        if (qnn.deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
            auto* h = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
            if (h->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF && h->perfInfra.destroyPowerConfigId)
                for (auto& kv : power_ids_) h->perfInfra.destroyPowerConfigId(kv.second);   // 作ったスレッドでないと失敗するものもある（無視する）
        }
    }
    power_ids_.clear();
    if (device && qnn.deviceFree) qnn.deviceFree(device);
    if (backend && qnn.backendFree) qnn.backendFree(backend);
    // dlclose はあえて行わない（HTP のスレッドが残る可能性があるため）
}

// ---------------------------------------------------------------------------------------------
QnnCtx::~QnnCtx() {
    if (ctx && lib) lib->qnn.contextFree(ctx, nullptr);
    if (map) munmap(map, map_size);
}

bool QnnModel::Load(QnnBackendLib* lib, const std::string& bin_path, std::string* err) {
    std::vector<std::shared_ptr<QnnModel>> all;
    if (!LoadAll(lib, bin_path, &all, err)) return false;
    *this = *all[0];   // 最初のグラフ（コンテキストは shared_ptr で共有される。残りのグラフの QnnModel は捨てる）
    return true;
}

// 複数グラフ（重み共有）の bin を 1 回だけ読み込み、グラフごとに QnnModel を作る。
bool QnnModel::LoadAll(QnnBackendLib* lib, const std::string& bin_path, std::vector<std::shared_ptr<QnnModel>>* out, std::string* err,
                       const std::vector<std::string>& enable_graphs) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    auto c = std::make_shared<QnnCtx>();
    c->lib = lib;
    int fd = open(bin_path.c_str(), O_RDONLY);
    if (fd < 0) return fail("モデルを開けません: " + bin_path);
    struct stat st{};
    fstat(fd, &st);
    c->map_size = static_cast<size_t>(st.st_size);
    c->map = mmap(nullptr, c->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (c->map == MAP_FAILED) { c->map = nullptr; return fail("mmap 失敗: " + bin_path); }

    QnnSystemContext_Handle_t sctx = nullptr;
    if (lib->sys.systemContextCreate(&sctx) != QNN_SUCCESS) return fail("systemContextCreate 失敗");
    const QnnSystemContext_BinaryInfo_t* info = nullptr; Qnn_ContextBinarySize_t isz = 0;
    if (lib->sys.systemContextGetBinaryInfo(sctx, c->map, c->map_size, &info, &isz) != QNN_SUCCESS || !info) {
        lib->sys.systemContextFree(sctx); return fail("バイナリ情報を取得できません");
    }
    const QnnSystemContext_GraphInfo_t* graphs = nullptr;
    uint32_t ngraphs = 0;
    if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) { graphs = info->contextBinaryInfoV1.graphs; ngraphs = info->contextBinaryInfoV1.numGraphs; }
    else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) { graphs = info->contextBinaryInfoV2.graphs; ngraphs = info->contextBinaryInfoV2.numGraphs; }
    else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) { graphs = info->contextBinaryInfoV3.graphs; ngraphs = info->contextBinaryInfoV3.numGraphs; }
    if (!graphs || ngraphs == 0) { lib->sys.systemContextFree(sctx); return fail("グラフ情報がありません"); }
    std::vector<std::shared_ptr<QnnModel>> models;
    for (uint32_t gi = 0; gi < ngraphs; gi++) {
        const QnnSystemContext_GraphInfo_t* g = &graphs[gi];
        std::string gname;
        const Qnn_Tensor_t *gin = nullptr, *gout = nullptr; uint32_t nin = 0, nout = 0;
        if (g->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            gname = g->graphInfoV1.graphName; gin = g->graphInfoV1.graphInputs; nin = g->graphInfoV1.numGraphInputs;
            gout = g->graphInfoV1.graphOutputs; nout = g->graphInfoV1.numGraphOutputs;
        } else if (g->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
            gname = g->graphInfoV2.graphName; gin = g->graphInfoV2.graphInputs; nin = g->graphInfoV2.numGraphInputs;
            gout = g->graphInfoV2.graphOutputs; nout = g->graphInfoV2.numGraphOutputs;
        } else {
            gname = g->graphInfoV3.graphName; gin = g->graphInfoV3.graphInputs; nin = g->graphInfoV3.numGraphInputs;
            gout = g->graphInfoV3.graphOutputs; nout = g->graphInfoV3.numGraphOutputs;
        }
        if (!enable_graphs.empty() && std::find(enable_graphs.begin(), enable_graphs.end(), gname) == enable_graphs.end()) continue;   // 除くグラフ
        auto m = std::make_shared<QnnModel>();
        m->lib_ = lib;
        m->graph_name_ = gname;
        for (uint32_t i = 0; i < nin; i++) m->in_.push_back(MakeDesc(gin[i]));
        for (uint32_t i = 0; i < nout; i++) m->out_.push_back(MakeDesc(gout[i]));
        models.push_back(std::move(m));
    }
    lib->sys.systemContextFree(sctx);

    // NPU の context は 1 つずつ作る（同時に作ると、DSP の空間を奪い合って失敗し、失敗した分のメモリが戻らないことがある。2026-10-18 に実機で踏んだ）
    static std::mutex create_mu;
    std::unique_lock<std::mutex> create_lock(create_mu, std::defer_lock);
    if (!lib->is_gpu) create_lock.lock();

    // GPU は低優先度のコンテキストで作る（OS の描画など、優先度の高い GPU 処理が割り込めるように）。受け付けなければ通常で作り直す。
    Qnn_ErrorHandle_t ce = QNN_COMMON_ERROR_GENERAL;
    if (lib->is_gpu) {
        QnnContext_Config_t pri = QNN_CONTEXT_CONFIG_INIT;
        pri.option = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
        pri.priority = QNN_PRIORITY_LOW;
        const QnnContext_Config_t* cfgs[] = {&pri, nullptr};
        ce = lib->qnn.contextCreateFromBinary(lib->backend, lib->device, cfgs, c->map, c->map_size, &c->ctx, nullptr);
        __android_log_print(ANDROID_LOG_INFO, "dsxl-qnn", "GPU context priority=LOW: %s (code %d)", ce == QNN_SUCCESS ? "accepted" : "rejected, retry default", static_cast<int>(ce));
        if (ce != QNN_SUCCESS) c->ctx = nullptr;
    }
    // LoRA の adapter を適用する context（UNet の unet_lora_4g.bin と、CLIP の clip_*_lora.bin）は、adapter のチェックサムの検査を省く:
    // 手元で作った adapter（loraconverter と LoraConverter.kt / ClipLoraConverter.kt）は、見出しのチェックサムが元のひな型のまま
    // （窓や重みの中身は検査されない。docs/Research.md の 6 節）。重み共有の bin（複数グラフ）は、読み込み側でも重み共有を有効にする
    // （qnn-net-run が設定ファイルの weight_sharing_enabled を読み込み時にも使うのと同じ）。
    if (models.empty()) return fail("指定のグラフが bin にありません: " + bin_path);
    // 読み込むグラフを絞るとき（メモリ節約モード）は、bin 内のグラフ数（絞る前）で重み共有の要否を決める
    const bool multi = ngraphs > 1;
    const bool lora_ctx = bin_path.find("lora") != std::string::npos;
    if (ce != QNN_SUCCESS && !lib->is_gpu && (multi || lora_ctx)) {
        QnnHtpContext_CustomConfig_t ws = QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT, sv = QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
        ws.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
        ws.weightSharingEnabled = true;
        sv.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION;
        sv.skipValidationOnBinarySection = true;
        QnnContext_Config_t c1 = QNN_CONTEXT_CONFIG_INIT, c2 = QNN_CONTEXT_CONFIG_INIT;
        c1.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c1.customConfig = &ws;
        c2.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c2.customConfig = &sv;
        // 読み込むグラフの絞り込み（無ければ全部）。配列は NULL で終わる
        std::vector<const char*> en;
        for (const auto& s : enable_graphs) en.push_back(s.c_str());
        en.push_back(nullptr);
        QnnContext_Config_t c3 = QNN_CONTEXT_CONFIG_INIT;
        c3.option = QNN_CONTEXT_CONFIG_ENABLE_GRAPHS; c3.enableGraphs = en.data();
        const bool filtered = !enable_graphs.empty();
        const QnnContext_Config_t* cfgs_multi[] = {&c1, &c2, filtered ? &c3 : nullptr, nullptr};
        const QnnContext_Config_t* cfgs_single[] = {&c2, filtered ? &c3 : nullptr, nullptr};
        // 失敗したら、同じ設定でもう一度試す（最大 3 回）。アプリを起動した直後の生成は、CLIP-L・CLIP-G・UNet の context を同時に作るので、DSP の空間が一時的に足りず
        // 失敗することがある（code 1007。2026-10-18 に実機で踏んだ）。ほかの読み込みが終われば通る。検査を省かない読み込みに落とすと、手元で作った adapter の
        // チェックサムが合わず、adapter の適用が code 1009 になる。
        for (int attempt = 1; attempt <= 3; attempt++) {
            ce = lib->qnn.contextCreateFromBinary(lib->backend, lib->device, multi ? cfgs_multi : cfgs_single, c->map, c->map_size, &c->ctx, nullptr);
            __android_log_print(ce == QNN_SUCCESS ? ANDROID_LOG_INFO : ANDROID_LOG_WARN, "dsxl-qnn", "HTP context (%s skip-validation, %u graph(s)): %s (code %d, attempt %d)",
                                multi ? "weight sharing +" : "LoRA", static_cast<unsigned>(models.size()), ce == QNN_SUCCESS ? "ok" : "failed", static_cast<int>(ce), attempt);
            if (ce == QNN_SUCCESS) break;
            c->ctx = nullptr;
            if (attempt < 3) std::this_thread::sleep_for(std::chrono::milliseconds(2500));
        }
    }
    if (ce != QNN_SUCCESS && !lib->is_gpu && multi) {
        // 検査を省く設定を受け付けないとき: 重み共有だけで（以前の読み込み方）
        QnnHtpContext_CustomConfig_t ws = QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
        ws.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
        ws.weightSharingEnabled = true;
        QnnContext_Config_t cc = QNN_CONTEXT_CONFIG_INIT;
        cc.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
        cc.customConfig = &ws;
        const QnnContext_Config_t* cfgs[] = {&cc, nullptr};
        ce = lib->qnn.contextCreateFromBinary(lib->backend, lib->device, cfgs, c->map, c->map_size, &c->ctx, nullptr);
        __android_log_print(ce == QNN_SUCCESS ? ANDROID_LOG_INFO : ANDROID_LOG_WARN, "dsxl-qnn", "HTP context (weight sharing only, %u graphs): %s (code %d)",
                            static_cast<unsigned>(models.size()), ce == QNN_SUCCESS ? "ok" : "failed, retry default", static_cast<int>(ce));
        if (ce != QNN_SUCCESS) c->ctx = nullptr;
    }
    for (int attempt = 1; ce != QNN_SUCCESS && attempt <= 3; attempt++) {   // 普通の context（CLIP・VAE）も、DSP の空間が一時的に足りないときは待って試し直す
        if (attempt > 1) std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        ce = lib->qnn.contextCreateFromBinary(lib->backend, lib->device, nullptr, c->map, c->map_size, &c->ctx, nullptr);
        if (ce != QNN_SUCCESS) { c->ctx = nullptr; __android_log_print(ANDROID_LOG_WARN, "dsxl-qnn", "context create failed (code %d, attempt %d)", static_cast<int>(ce), attempt); }
    }
    if (ce != QNN_SUCCESS) return fail("contextCreateFromBinary 失敗 (code " + std::to_string(static_cast<int>(ce)) + "): " + bin_path);
    for (auto& m : models) {
        m->ctx_ = c;
        if (lib->qnn.graphRetrieve(c->ctx, m->graph_name_.c_str(), &m->graph_) != QNN_SUCCESS) return fail("graphRetrieve 失敗: " + m->graph_name_);
    }
    // NPU の context は、作り終えたら bin の mmap は要らない（重みは DSP 側の DMA-BUF にコピー済み）。mmap を残すと、bin のページ（UNet 3.7GB、CLIP 1.4GB）が
    // 常駐するプロセスの RSS とページキャッシュに居座り、メモリが逼迫したとき lowmemorykiller が最大の RSS のプロセス（CLIP のサービス）を落とす。
    // 手放して、ページキャッシュも捨てる（2026-10-18）。GPU の context は bin を参照し続けるかもしれないので残す。
    if (!lib->is_gpu && c->map) {
        munmap(c->map, c->map_size);
        c->map = nullptr;
        int fd2 = open(bin_path.c_str(), O_RDONLY);
        if (fd2 >= 0) { posix_fadvise(fd2, 0, 0, POSIX_FADV_DONTNEED); close(fd2); }
    }
    __android_log_print(ANDROID_LOG_INFO, "dsxl-qnn", "loaded %s: %u graph(s)", bin_path.c_str(), static_cast<unsigned>(models.size()));
    // （投票は、グラフを実行するスレッドで行う: Init と、生成の開始（Begin）・VAE のスレッドの開始。読み込みのスレッドで投票しても、実行するスレッドとは別のスレッドなので意味がない）
    *out = std::move(models);
    return true;
}

const TensorDesc* QnnModel::Input(const std::string& n) const { for (auto& d : in_) if (d.name == n) return &d; return nullptr; }
const TensorDesc* QnnModel::Output(const std::string& n) const { for (auto& d : out_) if (d.name == n) return &d; return nullptr; }

std::string QnnModel::Describe() const {
    std::ostringstream os;
    for (auto& d : in_) { os << "in " << d.name << " ["; for (auto x : d.dims) os << x << ","; os << "] " << d.bytes << "B\n"; }
    for (auto& d : out_) { os << "out " << d.name << " ["; for (auto x : d.dims) os << x << ","; os << "] " << d.bytes << "B\n"; }
    return os.str();
}

QnnModel::~QnnModel() {
    if (adapter_map_) munmap(adapter_map_, adapter_map_size_);   // グラフ（コンテキスト）より先に、adapter の mmap を手放す
}

bool QnnModel::ApplyAdapter(const std::string& path, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    // このグラフに同じ adapter を適用済みなら何もしない（extended_udma の共有 bin は、差し替える重みをグラフごとに持つ。
    // 全グラフで 1 つを共有する lora_weight_sharing の bin では、グラフを切り替えるたびに適用し直す必要があったが、出力が壊れたので使っていない）
    if (path == adapter_) return true;
    if (!ctx_ || !lib_->qnn.contextApplyBinarySection) return fail("このバックエンドは adapter の適用に対応していません");
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return fail("adapter を開けません: " + path);
    struct stat st{};
    fstat(fd, &st);
    const size_t size = static_cast<size_t>(st.st_size);
    void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return fail("adapter の mmap に失敗: " + path);
    QnnContext_Buffer_t buf{};
    buf.version = QNN_CONTEXT_BUFFER_VERSION_1;
    buf.v1.memType = QNN_CONTEXTMEMTYPE_RAW;
    buf.v1.binaryBuf.data = map;
    buf.v1.binaryBuf.dataSize = static_cast<Qnn_ContextBinarySize_t>(size);
    const uint64_t t0 = NowNs();
    std::lock_guard<std::mutex> exec(lib_->exec_mu);   // 実行中のグラフには適用しない
    Qnn_ErrorHandle_t e = lib_->qnn.contextApplyBinarySection(ctx_->ctx, graph_, QNN_CONTEXT_SECTION_UPDATABLE, &buf, nullptr, nullptr);
    __android_log_print(e == QNN_SUCCESS ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, "dsxl-qnn", "applyBinarySection %s (%.1f MB): code %d, %.0f ms",
                        path.c_str(), size / 1048576.0, static_cast<int>(e), (NowNs() - t0) / 1e6);
    if (e != QNN_SUCCESS) { munmap(map, size); return fail("adapter の適用に失敗 (code " + std::to_string(static_cast<int>(e)) + "): " + path); }
    if (adapter_map_) munmap(adapter_map_, adapter_map_size_);
    adapter_map_ = map; adapter_map_size_ = size; adapter_ = path;
    return true;
}

bool QnnModel::Run(const std::map<std::string, const void*>& in, const std::map<std::string, void*>& out, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    std::vector<Qnn_Tensor_t> ti, to;
    // FP16 の入出力は、呼び出し側の float32 の配列と、ここで相互に変換する（LoRA つきベースの bin は FP16 の入出力。従来の bin は float32）
    std::vector<std::vector<uint16_t>> in16(in_.size()), out16(out_.size());
    for (size_t k = 0; k < in_.size(); k++) {
        const TensorDesc& d = in_[k];
        auto it = in.find(d.name);
        if (it == in.end()) return fail("入力が足りません: " + d.name);
        Qnn_Tensor_t t = d.tensor; TSetName(t, d.name.c_str()); TSetDims(t, const_cast<uint32_t*>(d.dims.data()));
        if (d.dtype == QNN_DATATYPE_FLOAT_16) {
            const size_t n = d.bytes / 2;
            const float* src = static_cast<const float*>(it->second);
            in16[k].resize(n);
            for (size_t i = 0; i < n; i++) { __fp16 h = static_cast<__fp16>(src[i]); std::memcpy(&in16[k][i], &h, 2); }
            TSetBuf(t, in16[k].data(), d.bytes);
        } else {
            TSetBuf(t, const_cast<void*>(it->second), d.bytes);
        }
        ti.push_back(t);
    }
    for (size_t k = 0; k < out_.size(); k++) {
        const TensorDesc& d = out_[k];
        auto it = out.find(d.name);
        if (it == out.end()) return fail("出力先が足りません: " + d.name);
        Qnn_Tensor_t t = d.tensor; TSetName(t, d.name.c_str()); TSetDims(t, const_cast<uint32_t*>(d.dims.data()));
        if (d.dtype == QNN_DATATYPE_FLOAT_16) { out16[k].resize(d.bytes / 2); TSetBuf(t, out16[k].data(), d.bytes); }
        else TSetBuf(t, it->second, d.bytes);
        to.push_back(t);
    }
    std::lock_guard<std::mutex> exec(lib_->exec_mu);   // 同じ backend では 1 グラフずつ。待った時間は busy に含めない
    const uint64_t t0 = NowNs();
    lib_->run_start_ns = t0;
    Qnn_ErrorHandle_t e = lib_->qnn.graphExecute(graph_, ti.data(), static_cast<uint32_t>(ti.size()), to.data(), static_cast<uint32_t>(to.size()), nullptr, nullptr);
    lib_->busy_ns += NowNs() - t0;
    lib_->run_start_ns = 0;
    if (e != QNN_GRAPH_NO_ERROR) return fail("graphExecute 失敗 (" + std::to_string(e) + ")");
    for (size_t k = 0; k < out_.size(); k++) {
        if (out_[k].dtype != QNN_DATATYPE_FLOAT_16) continue;
        float* dst = static_cast<float*>(out.find(out_[k].name)->second);
        for (size_t i = 0; i < out16[k].size(); i++) { __fp16 h; std::memcpy(&h, &out16[k][i], 2); dst[i] = static_cast<float>(h); }
    }
    return true;
}

}  // namespace dsxl
