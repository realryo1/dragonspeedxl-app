#include "engine/pipeline.h"

#include "engine/refine.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <future>
#include <mutex>
#include <thread>
#include <map>
#include <random>

#include <android/log.h>

namespace dsxl {
namespace {

// ---- ComfyUI / k-diffusion 互換の離散 sigma（EPS, linear beta 0.00085..0.012, 1000 step）----
struct Schedule {
    std::vector<double> log_sigmas;  // 1000
    Schedule() {
        const double a = std::sqrt(0.00085), b = std::sqrt(0.012);
        double ac = 1.0;
        for (int i = 0; i < 1000; i++) {
            double r = a + (b - a) * i / 999.0;
            ac *= 1.0 - r * r;
            log_sigmas.push_back(0.5 * std::log((1.0 - ac) / ac));
        }
    }
    double SigmaOfT(double t) const {  // 対数補間
        t = std::min(std::max(t, 0.0), 999.0);
        int lo = static_cast<int>(std::floor(t)), hi = static_cast<int>(std::ceil(t));
        double w = t - lo;
        return std::exp((1 - w) * log_sigmas[lo] + w * log_sigmas[hi]);
    }
    int TimestepOfSigma(double sigma) const {  // ComfyUI の timestep(): 対数距離が最小の整数 index
        double l = std::log(sigma);
        int best = 0; double bd = 1e30;
        for (int i = 0; i < 1000; i++) { double d = std::fabs(l - log_sigmas[i]); if (d < bd) { bd = d; best = i; } }
        return best;
    }
    // sgm_uniform: linspace(999, 0, N+1)[:-1] の sigma に 0 を足したもの
    std::vector<double> SgmUniform(int steps) const {
        std::vector<double> s;
        for (int k = 0; k < steps; k++) s.push_back(SigmaOfT(999.0 - 999.0 * k / steps));
        s.push_back(0.0);
        return s;
    }
    // ComfyUI の denoise<1: steps/denoise 段の sgm_uniform の末尾 steps+1 個
    std::vector<double> SgmUniformTail(int steps, double denoise) const {
        std::vector<double> all = SgmUniform(static_cast<int>(steps / denoise));
        return std::vector<double>(all.end() - (steps + 1), all.end());
    }
};

bool ReadFile(const std::string& path, std::vector<float>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    size_t n = static_cast<size_t>(f.tellg());
    out->resize(n / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out->data()), n);
    return true;
}

bool Exists(const std::string& p) { std::ifstream f(p); return f.good(); }

// 共通の必須ファイル。NPU の VAE は vae_64_w8a16.bin（あれば優先）か vae_64.bin（FP16）のどちらか 1 つでよい（MissingFiles で別に調べる）
const char* kFiles[] = {"clip_l.bin", "clip_g.bin", "text_projection.f32", "clip_vocab.tsv", "clip_merges.txt"};

std::atomic<int> g_gpu_cap{100};   // GPU の使用率の上限（%）。既定は制限なし
// メモリ節約モード: UNet の共有 bin を、今の向きの 2 グラフ（Base とタイル）だけで読む（約 1.4GB 減る）。向きが変わるたびに読み直す
std::atomic<bool> g_mem_saver{false};
// VAE の 1 タイルの所要（ms。直近の実測の移動平均。GPU は使用率の上限で休む分を含む）。GPU がタイルを取るかの判断に使う
std::atomic<int> g_vae_npu_ms{130}, g_vae_gpu_ms{1500};
void UpdateMs(std::atomic<int>& a, double ms) { a = static_cast<int>((3.0 * a.load() + ms) / 4.0); }

// GPU で動かすモデル（GPU 用 VAE と TAESD のプレビュー）は models/gpu/ にまとめる。*_fp32.dlc が元で、*_gpu.bin は初回に作るキャッシュ
const char* kGpuDlc = "gpu/vae_64_fp32.dlc";
const char* kGpuBin = "gpu/vae_64_gpu.bin";
const char* kGpuKey = "gpu_vae";

// /proc/meminfo の MemAvailable（MB）。読めなければ大きい値を返す（制限しない）。
long AvailMemMB() {
    std::ifstream f("/proc/meminfo");
    std::string k; long v = 0, kb = 0;
    while (f >> k >> v) {
        if (k == "MemAvailable:") { kb = v; break; }
        f.ignore(256, '\n');
    }
    return kb > 0 ? kb / 1024 : 1000000;
}

std::vector<int> TileStarts(int n, int tile = 64, int stride = 48) {
    std::vector<int> s;
    for (int x = 0; x + tile <= n || s.empty(); x += stride) s.push_back(x);
    if (s.back() + tile < n) s.push_back(n - tile);
    return s;
}

// 長さ n を長さ t のタイルで覆う開始位置（等間隔。重なりは最低 8）。refpipe.py の tile_starts と同じ。
std::vector<int> TileStartsEven(int n, int t, int min_overlap = 8) {
    if (n <= t) return {0};
    const int k = static_cast<int>(std::ceil(static_cast<double>(n - min_overlap) / (t - min_overlap)));
    std::vector<int> s;
    for (int i = 0; i < k; i++) s.push_back(static_cast<int>(std::lround(static_cast<double>(i) * (n - t) / (k - 1))));
    return s;
}

// タイルごとの 1 次元の重み。隣のタイルと重なる辺だけ、重なり幅 ov で (j+1)/(ov+1) と線形に立ち上げる（refpipe.py の _tile_weights と同じ）
std::vector<std::vector<float>> AxisWeights(const std::vector<int>& starts, int t) {
    std::vector<std::vector<float>> ws;
    for (size_t j = 0; j < starts.size(); j++) {
        std::vector<float> w(t, 1.f);
        if (j > 0) {
            const int ov = starts[j - 1] + t - starts[j];
            for (int k = 0; k < ov && k < t; k++) w[k] = static_cast<float>(k + 1) / (ov + 1);
        }
        if (j + 1 < starts.size()) {
            const int ov = starts[j] + t - starts[j + 1];
            for (int k = 0; k < ov && k < t; k++) w[t - ov + k] = static_cast<float>(ov - k) / (ov + 1);
        }
        ws.push_back(std::move(w));
    }
    return ws;
}

// ログ用の経過時間（ms。プロセス内の最初の呼び出しからの時刻）。UNet と VAE がどれだけ重なっているかを logcat で読むため
long LogMs() {
    static const auto t0 = std::chrono::steady_clock::now();
    return static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
}

// fg_busy_ を、CLIP / UNet を実行している間だけ持ち上げる
struct BusyGuard {
    std::atomic<int>& c;
    explicit BusyGuard(std::atomic<int>& x) : c(x) { ++c; }
    ~BusyGuard() { --c; }
};

}  // namespace

void Pipeline::SetGpuCap(int percent) { g_gpu_cap = std::min(100, std::max(10, percent)); }
void Pipeline::SetMemorySaver(bool on) { g_mem_saver = on; }

std::string Pipeline::MissingFiles(const std::string& dir) {
    std::string m;
    auto add = [&](const std::string& f) { m += std::string(m.empty() ? "" : ", ") + f; };
    for (const char* f : kFiles) if (!Exists(dir + "/" + f)) add(f);
    if (!Exists(dir + "/vae_64_w8a16.bin") && !Exists(dir + "/vae_64.bin")) add("vae_64_w8a16.bin（または vae_64.bin）");
    if (!Exists(dir + "/" + UnetFile())) add(UnetFile());
    if (!Exists(dir + "/" + DefaultAdapter())) add(DefaultAdapter());
    return m;
}

bool Pipeline::HasUnet(const std::string& dir) { return Exists(dir + "/" + UnetFile()) && Exists(dir + "/" + DefaultAdapter()); }

// UNet に適用する super adapter（4 グラフ共通の 1 ファイル）。LoRA なら loras/<名前>.4g.bin、無ければ default（重みが 0）
std::string Pipeline::AdapterPath(const std::string& lora) const {
    return lora.empty() ? dir_ + "/" + DefaultAdapter() : dir_ + "/loras/" + lora + ".4g.bin";
}

// UNet のグラフのキー。共有 bin の中では、latent の寸法（sample 入力の高さ・幅）で見分ける: Base は 720p、タイルは 1024x576
std::string Pipeline::GraphKey(bool portrait, bool tile) { return std::string("unet:") + (tile ? "tile:" : "base:") + (portrait ? "p" : "l"); }
bool Pipeline::GraphDims(const std::string& key, int* lh, int* lw) {
    for (bool p : {false, true}) {
        if (key == GraphKey(p, false)) { *lh = p ? 160 : 90; *lw = p ? 90 : 160; return true; }
        if (key == GraphKey(p, true)) { *lh = p ? 128 : 72; *lw = p ? 72 : 128; return true; }
    }
    return false;
}
std::string Pipeline::ClipFile(const char* which, const GenParams& p, std::string* adapter) const {
    return ClipFileFor(dir_, which, p.lora_adapter, adapter);
}
bool Pipeline::Init(const std::string& model_dir, const std::string& lib_dir, std::string* err) {
    dir_ = model_dir;
    mkdir((dir_ + "/gpu").c_str(), 0775);   // GPU 用のモデルとそのキャッシュの置き場（初回のコンパイルの出力先）
    std::string miss = MissingFiles(dir_);
    if (!miss.empty()) { if (err) *err = "モデルファイルがありません: " + miss; return false; }
    if (!lib_.Init(lib_dir, err)) return false;
    // GPU は任意（無くても NPU だけで動く）
    // DLC か、コンパイル済みの bin のどちらかがあれば使える（読むのは bin が先。DLC は bin を作るときだけ要る）
    if ((Exists(dir_ + "/" + kGpuDlc) || Exists(dir_ + "/" + kGpuBin)) && Exists(lib_dir + "/libQnnGpu.so")) {
        std::string gerr;
        gpu_lib_ok_ = gpu_lib_.Init(lib_dir, &gerr, true);
    }
    // プレビュー（TAESD）は GPU が使えるときだけ。モデル（taesd_*_fp32.dlc）が無ければ、使うときに黙って無効になる
    if (gpu_lib_ok_) prev_ = std::make_unique<Previewer>(&gpu_lib_, dir_, [] { return g_gpu_cap.load(); });
    return true;
}

void Pipeline::SetPreviewFn(Previewer::Fn fn) {
    if (prev_) { prev_->Reset(); prev_->SetCallback(std::move(fn)); }
}

// ---------------------------------------------------------------------------------------------
// モデルの常駐・先読み
void Pipeline::PrefetchGraph(const std::string& file, int lh, int lw) {
    const std::string sfile = UnetFile(), path = dir_ + "/" + sfile;
    // メモリ節約モードでは、今の向きの 2 グラフ（Base とタイル）だけを読む。グループのキーに向きを入れ、条件の違うグループが載っていたら、
    // 先に手放す（新しい context を作る前に空けないと、DSP の空間とメモリが足りない）。オフのときは今まで通り 4 グラフ 1 つ
    const bool saver = g_mem_saver.load();
    const bool portrait = lh > lw;
    const std::string gkey = saver ? sfile + (portrait ? "#p" : "#l") : sfile;
    std::vector<std::string> enable;
    if (saver) {
        for (const std::string& k : {GraphKey(portrait, false), GraphKey(portrait, true)}) {
            int h = 0, w = 0;
            if (GraphDims(k, &h, &w)) enable.push_back("unet_" + std::to_string(w * 8) + "x" + std::to_string(h * 8));   // 例: unet_720x1280（幅 x 高さ）
        }
    }
    {
        std::vector<std::shared_future<LoadResult>> old;
        std::vector<std::shared_future<GroupResult>> old_groups;
        {
            std::lock_guard<std::mutex> lock(cache_mu_);
            for (auto it = groups_.begin(); it != groups_.end();) {
                if (it->first == gkey) { ++it; continue; }
                old_groups.push_back(it->second);
                it = groups_.erase(it);
            }
            if (!old_groups.empty()) {   // UNet のグループは同時に 1 つだけなので、載っていた UNet のグラフはすべて手放したグループのもの
                for (auto it = cache_.begin(); it != cache_.end();) {
                    if (it->first.rfind("unet:", 0) == 0) { old.push_back(it->second); it = cache_.erase(it); }
                    else ++it;
                }
            }
        }
        if (!old_groups.empty()) {
            __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "UNet を手放して読み直す（条件が変わった: %s）", gkey.c_str());
            for (auto& f : old) f.wait();
            for (auto& f : old_groups) f.wait();
            old.clear();          // 先にグラフ側を手放し、
            old_groups.clear();   // そのあとグループ（コンテキストの最後の持ち主）を手放す
        }
    }
    std::lock_guard<std::mutex> lock(cache_mu_);
    if (cache_.count(file)) return;
    auto git = groups_.find(gkey);
    if (git == groups_.end()) {   // 共有 bin を 1 回だけ読み込む（同じ向きの Base・タイルが同じ読み込みを待つ）
        git = groups_.emplace(gkey, std::async(std::launch::async, [this, path, enable]() {
            GroupResult r;
            __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "load start (shared%s) %s  avail=%ldMB", enable.empty() ? "" : ", memory saver", path.c_str(), AvailMemMB());
            if (!QnnModel::LoadAll(&lib_, path, &r.models, &r.err, enable)) r.models.clear();
            __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "load %s (shared) %s  graphs=%d  avail=%ldMB", r.models.empty() ? ("FAILED " + r.err).c_str() : "done", path.c_str(),
                                static_cast<int>(r.models.size()), AvailMemMB());
            return r;
        }).share()).first;
    }
    auto grp = git->second;
    // 個別のファイルと同じ形（LoadResult）で、latent の寸法が合うグラフを取り出す。無ければ model は null
    cache_[file] = std::async(std::launch::async, [grp, lh, lw, sfile]() {
        LoadResult r;
        const GroupResult& g = grp.get();
        for (const auto& m : g.models) {
            const TensorDesc* sd = m->Input("sample");
            if (sd && sd->dims.size() == 4 && sd->dims[2] == static_cast<uint32_t>(lh) && sd->dims[3] == static_cast<uint32_t>(lw)) { r.model = m; break; }
        }
        if (!r.model) r.err = g.models.empty() ? g.err : (sfile + " に latent " + std::to_string(lh) + "x" + std::to_string(lw) + " のグラフがありません");
        return r;
    }).share();
}

void Pipeline::Prefetch(const std::string& file) {
    {
        std::lock_guard<std::mutex> lock(cache_mu_);
        if (cache_.count(file)) return;
    }
    int lh = 0, lw = 0;
    if (GraphDims(file, &lh, &lw)) { PrefetchGraph(file, lh, lw); return; }   // UNet のグラフは共有 bin から取り出す
    std::lock_guard<std::mutex> lock(cache_mu_);
    if (cache_.count(file)) return;
    // NPU の VAE は、W8A16 版（vae_64_w8a16.bin）があればそちらを使う（FP16 版より速い。画質を比べるときは、このファイルを置く／消すだけで切り替わる）
    const bool vae_q = file == "vae_64.bin" && Exists(dir_ + "/vae_64_w8a16.bin");
    const std::string path = dir_ + "/" + (vae_q ? "vae_64_w8a16.bin" : file);
    cache_[file] = std::async(std::launch::async, [this, path]() {
        LoadResult r;
        auto m = std::make_shared<QnnModel>();
        __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "load start %s  avail=%ldMB", path.c_str(), AvailMemMB());
        if (m->Load(&lib_, path, &r.err)) r.model = std::move(m);
        __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "load %s %s  avail=%ldMB", r.model ? "done" : ("FAILED " + r.err).c_str(), path.c_str(), AvailMemMB());
        return r;
    }).share();
}

std::shared_future<Pipeline::LoadResult> Pipeline::FutureOf(const std::string& file) {
    std::lock_guard<std::mutex> lock(cache_mu_);
    auto it = cache_.find(file);
    return it == cache_.end() ? std::shared_future<LoadResult>() : it->second;
}

bool Pipeline::WaitLoaded(const std::string& file, std::string* err) {
    auto f = FutureOf(file);
    if (!f.valid()) { if (err) *err = "読み込みが始まっていません: " + file; return false; }
    const LoadResult& r = f.get();
    if (r.model) return true;
    if (err) *err = file + ": " + r.err;
    { std::lock_guard<std::mutex> lock(cache_mu_); cache_.erase(file); }
    return false;
}

std::shared_ptr<QnnModel> Pipeline::Acquire(const std::string& file, std::string* err) {
    // 読み込みを（まだなら）始めて、終わるまで待つ。別スレッドの Evict と行き違って消えていたら、もう一度始める
    auto load = [&]() {
        LoadResult r;
        for (int i = 0; i < 2; i++) {
            Prefetch(file);
            auto f = FutureOf(file);
            if (f.valid()) return f.get();
        }
        r.err = "読み込みを開始できません: " + file;
        return r;
    };
    LoadResult r = load();
    if (!r.model) {
        // メモリ不足などで失敗した可能性がある: 他をすべて手放してもう一度だけ同期で試す（裏の VAE が持つモデルは、その VAE が終わるまで生きる）
        { std::lock_guard<std::mutex> lock(cache_mu_); cache_.erase(file); }
        DropCache();
        r = load();
        if (!r.model) { std::lock_guard<std::mutex> lock(cache_mu_); cache_.erase(file); if (err) *err = r.err; return nullptr; }
    }
    return r.model;
}

void Pipeline::PrefetchGpuVae() {
    if (!gpu_lib_ok_ || gpu_failed_) return;
    const std::string dlc = dir_ + "/" + kGpuDlc, bin = dir_ + "/" + kGpuBin;
    std::lock_guard<std::mutex> lock(cache_mu_);
    if (cache_.count(kGpuKey)) return;
    cache_[kGpuKey] = std::async(std::launch::async, [this, dlc, bin]() {
        LoadResult r;
        std::string e;
        auto m = std::make_shared<QnnModel>();
        if (Exists(bin) && m->Load(&gpu_lib_, bin, &e)) { r.model = std::move(m); return r; }
        if (Exists(bin)) std::remove(bin.c_str());     // 古い・壊れたキャッシュは作り直す
        m = std::make_shared<QnnModel>();
        if (gpu_lib_.BuildGpuContextBinary(dlc, bin, &r.err) && m->Load(&gpu_lib_, bin, &r.err)) r.model = std::move(m);
        return r;
    }).share();
}

std::shared_ptr<QnnModel> Pipeline::AcquireGpuVae() {
    PrefetchGpuVae();
    auto f = FutureOf(kGpuKey);
    if (!f.valid()) return nullptr;
    LoadResult r = f.get();
    if (!r.model) { std::lock_guard<std::mutex> lock(cache_mu_); cache_.erase(kGpuKey); gpu_failed_ = true; }
    return r.model;
}

void Pipeline::Evict(const std::string& file) {
    std::shared_future<LoadResult> f;
    {
        std::lock_guard<std::mutex> lock(cache_mu_);
        auto it = cache_.find(file);
        if (it == cache_.end()) return;
        f = it->second;
        cache_.erase(it);
    }
    f.wait();   // ロード中なら終わるまで待つ。f を手放すとモデルが解放される
}

void Pipeline::BusyNs(uint64_t* npu, uint64_t* gpu, uint64_t* now) const {
    *npu = lib_.BusyNs();
    *gpu = gpu_lib_.BusyNs();
    *now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void Pipeline::DropCache() {
    std::map<std::string, std::shared_future<LoadResult>> old;
    std::map<std::string, std::shared_future<GroupResult>> old_groups;
    { std::lock_guard<std::mutex> lock(cache_mu_); old.swap(cache_); old_groups.swap(groups_); }
    for (auto& kv : old) kv.second.wait();
    for (auto& kv : old_groups) kv.second.wait();
    old.clear();         // 先にグラフ側（QnnModel）を手放し、
    old_groups.clear();  // そのあとグループ（コンテキストの最後の持ち主）を手放す
}

void Pipeline::StopDecodes() {
    std::vector<DecodeEntry> entries;
    std::shared_future<DecodeResult> last;
    std::shared_ptr<DecodeState> last_state;
    {
        std::lock_guard<std::mutex> lock(decode_mu_);
        for (auto& kv : decodes_) entries.push_back(kv.second);
        last = last_decode_; last_state = last_state_;
    }
    for (auto& e : entries) e.state->cancel = true;
    if (last_state) last_state->cancel = true;
    for (auto& e : entries) e.fut.wait();
    if (last.valid()) last.wait();
    std::lock_guard<std::mutex> lock(decode_mu_);
    decodes_.clear();
    last_decode_ = std::shared_future<DecodeResult>();
    last_state_.reset();
}

void Pipeline::DropModels() {
    std::lock_guard<std::mutex> fg(fg_mu_);   // 実行中のジョブがあれば終わるまで待つ
    StopDecodes();
    DropCache();
    clip_.Unload();   // 別プロセスの CLIP も手放す（つながっていなければ何もしない）
    if (prev_) prev_->DropModels();   // プレビュー（TAESD）の GPU モデルも手放す
}

int Pipeline::ModelState(bool portrait) {
    int m = 0;
    bool loading = false;
    auto check = [&](const std::string& file, int bit) {
        auto f = FutureOf(file);
        if (!f.valid()) return;
        if (f.wait_for(std::chrono::seconds(0)) != std::future_status::ready) { loading = true; return; }
        if (f.get().model) m |= bit;
    };
    if (clip_.Connected()) m |= kBitClipL | kBitClipG;   // CLIP は別プロセス。つながっていれば（エンコードか Warm が通っていれば）載っているとみなす
    check("vae_64.bin", kBitVae);
    check(GraphKey(portrait, false), kBitBaseUnet); check(GraphKey(portrait, true), kBitTile); check(kGpuKey, kBitGpuVae);
    return loading ? (m | kBitLoading) : m;
}

std::string Pipeline::LoadModels(bool portrait, const ProgressFn& progress) {
    if (!HasUnet(dir_)) return "!モデルファイルがありません: " + MissingFiles(dir_);
    // UNet は共有 bin 1 つ（縦横の全グラフが同じ context）なので、向きのグラフを 2 つ取り出しても追加のメモリは要らない。
    // CLIP（CLIP-L・bigG）は別プロセス（:clip サービス）が読む。UNet のプロセスとは DSP の空間が別なので、並行して読んでよい（clip_service.h）
    const std::vector<std::string> files = {"vae_64.bin", GraphKey(portrait, false), GraphKey(portrait, true)};
    const bool with_gpu = gpu_lib_ok_ && !gpu_failed_;
    const int total = 1 + static_cast<int>(files.size()) + (with_gpu ? 1 : 0);

    progress(PStage::kLoad, 0, total);
    int done = 0;
    std::string note;
    auto clips = std::async(std::launch::async, [this]() { std::string e; return clip_.Warm("", &e) ? std::string() : e; });
    for (const std::string& f : files) Prefetch(f);
    if (with_gpu) PrefetchGpuVae();
    for (const std::string& f : files) {
        std::string e;
        if (!WaitLoaded(f, &e)) { clips.wait(); return "!" + e; }
        progress(PStage::kLoad, ++done, total);
    }
    {
        const std::string e = clips.get();
        if (!e.empty()) return "!" + e;
        progress(PStage::kLoad, ++done, total);
    }
    if (with_gpu) {
        if (!AcquireGpuVae()) note = "GPU 用 VAE は使えません（VAE は NPU だけで実行します）";
        progress(PStage::kLoad, ++done, total);
    }
    return note;
}

void Pipeline::PrefetchNext(bool portrait) {
    if (dir_.empty()) return;
    Prefetch("vae_64.bin");
    PrefetchGpuVae();
    if (HasUnet(dir_)) { Prefetch(GraphKey(portrait, false)); Prefetch(GraphKey(portrait, true)); }
}

// ---------------------------------------------------------------------------------------------
std::string Pipeline::EncodeKey(const GenParams& p) const {
    std::string ad_l, ad_g;
    const std::string fl = ClipFile("clip_l", p, &ad_l), fg = ClipFile("clip_g", p, &ad_g);
    char a[32];
    std::snprintf(a, sizeof(a), "%.6g", ad_l.empty() && ad_g.empty() ? 0.0 : static_cast<double>(p.lora_te_alpha));
    return fl + "|" + ad_l + "|" + fg + "|" + ad_g + "|" + a + "|" + p.prompt;
}

bool Pipeline::Encode(const std::string& prompt, std::vector<float>* hidden, std::vector<float>* pooled, std::string* err) {
    // CLIP は別プロセス（:clip サービス。clip_service.h）で実行する。LoRA の CLIP と adapter があるかの判断も向こうが行う（ClipFileFor）
    return clip_.Encode(prompt, last_.lora_adapter, last_.lora_te_alpha, hidden, pooled, err);
}

bool Pipeline::Sample(QnnModel& unet, PStage stage, const std::vector<double>& sig, std::vector<float>& x, std::mt19937_64& rng,
                      const ProgressFn& progress, std::string* err) {
    const TensorDesc* sd = unet.Input("sample");
    if (!sd || sd->dims.size() != 4) { if (err) *err = "UNet の入力 sample が不正です"; return false; }
    const int lh = sd->dims[2], lw = sd->dims[3], n = 4 * lh * lw;
    if (static_cast<int>(x.size()) != n) { if (err) *err = "latent の寸法が UNet と一致しません"; return false; }

    static const Schedule sched;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x_in(n), eps(n);
    const float time_ids[6] = {static_cast<float>(lh * 8), static_cast<float>(lw * 8), 0, 0, static_cast<float>(lh * 8), static_cast<float>(lw * 8)};
    const int steps = static_cast<int>(sig.size()) - 1;
    const float lora_alpha = last_.lora_adapter.empty() ? 0.f : last_.lora_alpha;   // UNet の入力 lora_alpha（LoRA なしは 0。default の adapter は重みも 0）
    for (int i = 0; i < steps; i++) {
        if (cancel_) { if (err) *err = "キャンセルしました"; return false; }
        progress(stage, i + 1, steps);
        const float s = static_cast<float>(sig[i]);
        const float c_in = 1.f / std::sqrt(s * s + 1.f);
        for (int k = 0; k < n; k++) x_in[k] = x[k] * c_in;
        float t = static_cast<float>(sched.TimestepOfSigma(sig[i]));
        if (!unet.Run({{"sample", x_in.data()}, {"timestep", &t}, {"encoder_hidden_states", hidden_.data()}, {"text_embeds", pooled_.data()}, {"time_ids", time_ids}, {"lora_alpha", &lora_alpha}},
                      {{"eps", eps.data()}}, err)) return false;
        const float sn = static_cast<float>(sig[i + 1]);
        for (int k = 0; k < n; k++) x[k] = x[k] - eps[k] * s;           // denoised
        // プレビュー: この x0 を絵にする。速度優先（TAESD で最終画像）のときだけ、最終段の最後のステップは除く（すぐ最終画像が出るので重複する。Hires を掛けない Base は Base が最終段）
        const bool final_stage = stage == PStage::kHiresUnet || !last_.hires;
        if (!(last_.draft && final_stage && i == steps - 1)) EmitPreview(x, lh, lw);
        if (sn > 0) for (int k = 0; k < n; k++) x[k] += nd(rng) * sn;   // LCM: 次の sigma のノイズを足す
    }
    return true;
}

bool Pipeline::SampleTiled(QnnModel& unet, PStage stage, const std::vector<double>& sig, std::vector<float>& x, int lh, int lw, std::mt19937_64& rng,
                           const ProgressFn& progress, Stream* stream, std::string* err) {
    const TensorDesc* sd = unet.Input("sample");
    if (!sd || sd->dims.size() != 4) { if (err) *err = "タイル UNet の入力 sample が不正です"; return false; }
    const int th = sd->dims[2], tw = sd->dims[3];
    if (lh < th || lw < tw || static_cast<int>(x.size()) != 4 * lh * lw) { if (err) *err = "タイル UNet の寸法が latent と合いません"; return false; }

    // タイルの配置（等間隔・重なりは最低 8）と、1 次元の重み（隣のタイルと重なる辺だけ、重なり幅で 0→1 に線形に立ち上げる）
    const std::vector<int> ys = TileStartsEven(lh, th), xs = TileStartsEven(lw, tw);
    const std::vector<std::vector<float>> wy = AxisWeights(ys, th), wx = AxisWeights(xs, tw);
    std::vector<float> wsum(static_cast<size_t>(lh) * lw, 0.f);
    for (size_t i = 0; i < ys.size(); i++) for (size_t j = 0; j < xs.size(); j++)
        for (int r = 0; r < th; r++) for (int q = 0; q < tw; q++) wsum[static_cast<size_t>(ys[i] + r) * lw + xs[j] + q] += wy[i][r] * wx[j][q];

    static const Schedule sched;
    std::normal_distribution<float> nd(0.f, 1.f);
    const int n = 4 * lh * lw, tn = 4 * th * tw;
    std::vector<float> x_in(n), acc(n), tin(tn), tout(tn);
    const float time_ids[6] = {static_cast<float>(lh * 8), static_cast<float>(lw * 8), 0, 0, static_cast<float>(lh * 8), static_cast<float>(lw * 8)};   // 全体の寸法
    const int steps = static_cast<int>(sig.size()) - 1;
    const float lora_alpha = last_.lora_adapter.empty() ? 0.f : last_.lora_alpha;   // UNet の入力 lora_alpha（LoRA なしは 0）
    // 最終ステップの確定の進め方。タイルは「外側の軸」に沿って 1 列ずつ処理する: その列が終わると、次の列と重ならない手前の範囲（最初の重なりの手前まで）が確定する。
    // 確定の隙間（最初の列の幅）が VAE タイル（64）以上ある軸を選ぶ（縦長は行方向、横長は列方向）。どちらも 64 未満なら、ストリームはしない。
    const int gap_rows = ys.size() > 1 ? ys[1] : lh, gap_cols = xs.size() > 1 ? xs[1] : lw;
    const bool by_cols = gap_cols > gap_rows;
    const bool can_stream = stream != nullptr && std::max(gap_rows, gap_cols) >= 64;
    const int n_outer = static_cast<int>(by_cols ? xs.size() : ys.size()), n_inner = static_cast<int>(by_cols ? ys.size() : xs.size());
    for (int i = 0; i < steps; i++) {
        if (cancel_) { if (err) *err = "キャンセルしました"; return false; }
        progress(stage, i + 1, steps);
        const float s = static_cast<float>(sig[i]);
        const float c_in = 1.f / std::sqrt(s * s + 1.f);
        for (int k = 0; k < n; k++) x_in[k] = x[k] * c_in;
        float t = static_cast<float>(sched.TimestepOfSigma(sig[i]));
        std::fill(acc.begin(), acc.end(), 0.f);
        const bool last_step = i == steps - 1;
        const bool streaming = can_stream && last_step && stream->started == false && StartStream(*stream, lh, lw, by_cols);   // 最終ステップの頭で VAE（GPU）を待機させる
        int done_end = 0;   // 確定済みの範囲（外側の軸で、ここまで）
        for (int o = 0; o < n_outer; o++) {
          for (int in = 0; in < n_inner; in++) {
            const size_t a = static_cast<size_t>(by_cols ? in : o), b = static_cast<size_t>(by_cols ? o : in);
            if (cancel_) { if (err) *err = "キャンセルしました"; return false; }
            const int y0 = ys[a], x0 = xs[b];
            for (int c = 0; c < 4; c++) for (int r = 0; r < th; r++)
                std::copy(x_in.begin() + (static_cast<size_t>(c) * lh + y0 + r) * lw + x0, x_in.begin() + (static_cast<size_t>(c) * lh + y0 + r) * lw + x0 + tw,
                          tin.begin() + (static_cast<size_t>(c) * th + r) * tw);
            if (!unet.Run({{"sample", tin.data()}, {"timestep", &t}, {"encoder_hidden_states", hidden_.data()}, {"text_embeds", pooled_.data()}, {"time_ids", time_ids}, {"lora_alpha", &lora_alpha}},
                          {{"eps", tout.data()}}, err)) return false;
            for (int c = 0; c < 4; c++) for (int r = 0; r < th; r++) for (int q = 0; q < tw; q++)
                acc[(static_cast<size_t>(c) * lh + y0 + r) * lw + x0 + q] += tout[(static_cast<size_t>(c) * th + r) * tw + q] * wy[a][r] * wx[b][q];
          }
          if (streaming) {
            // この列まで終わった。次の列と重ならない手前の範囲（次の列の開始位置まで。最後の列なら全部）が確定する → x に反映し、VAE が読む latent に写して公開する
            const int end = o + 1 < n_outer ? (by_cols ? xs[o + 1] : ys[o + 1]) : (by_cols ? lw : lh);
            for (int c = 0; c < 4; c++) for (int y = (by_cols ? 0 : done_end); y < (by_cols ? lh : end); y++) for (int xx = (by_cols ? done_end : 0); xx < (by_cols ? end : lw); xx++) {
                const size_t k = (static_cast<size_t>(c) * lh + y) * lw + xx, p = static_cast<size_t>(y) * lw + xx;
                x[k] = x[k] - (acc[k] / wsum[p]) * s;
                stream->src->latent[k] = x[k];
            }
            done_end = end;
            if (by_cols) stream->src->cols.store(end); else stream->src->rows.store(end);   // 書き込みのあとに公開する（VAE のスレッドが読み始める）
            __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "[%ld ms] hires last step: %s 0..%d final (VAE tiles done so far: %d)", LogMs(), by_cols ? "cols" : "rows", end, stream->state->done.load());
          }
        }
        if (last_step) __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "[%ld ms] hires last step: UNet done (streaming=%d)", LogMs(), streaming ? 1 : 0);
        if (streaming) { EmitPreview(x, lh, lw); continue; }   // 最終ステップは確定済み（sn = 0 なのでノイズは足さない）。確定した x0 をプレビューにも出す（VAE が終わるより早く見える）
        const float sn = static_cast<float>(sig[i + 1]);
        for (int c = 0; c < 4; c++) for (size_t p = 0; p < static_cast<size_t>(lh) * lw; p++) {
            const size_t k = static_cast<size_t>(c) * lh * lw + p;
            x[k] = x[k] - (acc[k] / wsum[p]) * s;                 // denoised（重みつき平均した eps）
        }
        if (!(last_step && last_.draft)) EmitPreview(x, lh, lw);   // プレビュー（タイル Hires は最終段。速度優先のときだけ、最後のステップは、すぐ最終画像が出るので除く）
        if (sn > 0) for (int k = 0; k < n; k++) x[k] += nd(rng) * sn;   // LCM: 次の sigma のノイズを足す
    }
    return true;
}

bool Pipeline::Denoise(const GenParams& p, const ProgressFn& progress, std::string* err) {
    auto unet = Acquire(base_file_, err);
    if (!unet) return false;
    if (!unet->ApplyAdapter(AdapterPath(p.lora_adapter), err)) return false;   // 同じグラフに同じ adapter なら何もしない
    Prefetch("vae_64.bin");                                   // UNet の計算中に VAE を読み込んでおく
    if (p.gpu_vae) PrefetchGpuVae();
    if (p.hires) Prefetch(tile_file_);   // タイル用グラフ（同じ共有 bin の中なので、追加のメモリは要らない）
    const TensorDesc* sd = unet->Input("sample");
    if (!sd || sd->dims.size() != 4) { if (err) *err = "UNet の入力 sample が不正です"; return false; }
    base_lh_ = sd->dims[2]; base_lw_ = sd->dims[3];

    static const Schedule sched;
    std::vector<double> sig = sched.SgmUniform(p.steps);
    std::mt19937_64 rng(static_cast<uint64_t>(p.seed));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(4 * static_cast<size_t>(base_lh_) * base_lw_);
    const float s0 = static_cast<float>(sig[0]);
    for (auto& v : x) v = nd(rng) * std::sqrt(1.f + s0 * s0);   // max_denoise: noise * sqrt(1 + sigma^2)
    if (!Sample(*unet, PStage::kBaseUnet, sig, x, rng, progress, err)) return false;
    base_latent_ = std::move(x);
    return true;
}

// Base latent を nearest-exact ×1.5 に拡大し、denoise 0.39 の LCM（hires_steps 段）を、タイル用グラフ（MultiDiffusion）で掛けて 1920x1080 の latent を作る。
bool Pipeline::Hires(const ProgressFn& progress, std::vector<float>* latent, int* lh_out, int* lw_out, Stream* stream, std::string* err) {
    if (base_latent_.empty()) { if (err) *err = "Base の latent がありません"; return false; }
    progress(PStage::kUpscale, 1, 1);
    const int lh = (base_lh_ * 3 + 1) / 2, lw = (base_lw_ * 3 + 1) / 2;   // round(h * 1.5)
    const double inv_h = static_cast<double>(base_lh_) / lh, inv_w = static_cast<double>(base_lw_) / lw;
    std::vector<float> x(4 * static_cast<size_t>(lh) * lw);
    for (int c = 0; c < 4; c++) for (int y = 0; y < lh; y++) {
        const int sy = std::min(static_cast<int>(std::floor((y + 0.5) * inv_h)), base_lh_ - 1);   // nearest-exact
        for (int xx = 0; xx < lw; xx++) {
            const int sx = std::min(static_cast<int>(std::floor((xx + 0.5) * inv_w)), base_lw_ - 1);
            x[(static_cast<size_t>(c) * lh + y) * lw + xx] = base_latent_[(static_cast<size_t>(c) * base_lh_ + sy) * base_lw_ + sx];
        }
    }
    static const Schedule sched;
    std::vector<double> sig = sched.SgmUniformTail(last_.hires_steps, last_.hires_denoise);
    std::mt19937_64 rng(static_cast<uint64_t>(last_.seed) ^ 0x9E3779B97F4A7C15ULL);
    std::normal_distribution<float> nd(0.f, 1.f);
    const float s0 = static_cast<float>(sig[0]);
    for (auto& v : x) v += nd(rng) * s0;      // 部分 denoise: latent + noise * sigma0

    auto tile = Acquire(tile_file_, err);
    if (!tile) return false;
    if (!tile->ApplyAdapter(AdapterPath(last_.lora_adapter), err)) return false;   // タイルのグラフにも、同じ super adapter を適用する
    if (!SampleTiled(*tile, PStage::kHiresUnet, sig, x, lh, lw, rng, progress, stream, err)) return false;
    final_latent_ = x; final_lh_ = lh; final_lw_ = lw; final_ready_ = true;   // 領域リファインが窓を切り出す（約 0.5MB）
    *latent = std::move(x); *lh_out = lh; *lw_out = lw;
    return true;
}

// ---------------------------------------------------------------------------------------------
// VAE（タイル）。裏のスレッドで実行する。latent は呼び出し側のコピーを受け取る。
Pipeline::DecodeResult Pipeline::DecodeImpl(std::shared_ptr<QnnModel> vae, std::shared_ptr<QnnModel> gvae, std::shared_ptr<DecodeSource> src,
                                            int lh, int lw, DecodeState* st) {
    DecodeResult res;
    const std::vector<float>& latent = src->latent;
    const int H = lh * 8, W = lw * 8;
    std::vector<float> acc(static_cast<size_t>(3) * H * W, 0.f);
    std::vector<float> ramp(512);
    for (int i = 0; i < 512; i++) ramp[i] = std::min(std::min(i + 1, 512 - i), 128) / 128.f;
    std::vector<int> ys = TileStarts(lh), xs = TileStarts(lw);
    const int total = static_cast<int>(ys.size() * xs.size());
    st->total = total;

    // タイルの取り方: 確定済みの範囲（src->rows / cols）に収まる、まだ誰も取っていないタイルを先頭から 1 つ取る。
    // 全部確定済みなら（普通の VAE）上から順。タイル Hires の最終ステップの途中から始めたときは、確定が進むのを待ちながら取る。
    std::atomic<int> nclaimed{0};
    std::vector<char> claimed(total, 0);
    std::mutex claim_mu;
    auto tile_ready = [&](int idx) { return ys[idx / xs.size()] + 64 <= src->rows.load() && xs[idx % xs.size()] + 64 <= src->cols.load(); };
    std::atomic<bool> stop{false}, gpu_alive{gvae != nullptr};
    // GPU がタイルを取ってよいか。NPU が UNet（次のジョブを含む）に塞がれている間は取る。NPU が空いているときは、
    // 残りを NPU だけで片付けるより GPU の 1 タイルが早く終わるときだけ取る（取ると、NPU が全部終えたあと GPU の 1 タイルを待つことになる。
    // 実測で NPU 124ms / GPU 約 1.5 秒。15 タイル中の最後の 1 タイルを GPU が持ち、約 0.9 秒待っていた）
    auto gpu_may_claim = [&]() {
        if (fg_busy_.load() > 0) return true;
        const int left = total - nclaimed.load();
        return static_cast<long>(left) * g_vae_npu_ms.load() > static_cast<long>(g_vae_gpu_ms.load()) * 11 / 10;
    };
    // 戻り値: 1=取れた（*out）、0=全部取られた、-1=中止
    auto claim = [&](int* out, bool is_gpu) -> int {
        while (true) {
            if (stop || st->cancel) return -1;
            {
                std::lock_guard<std::mutex> lock(claim_mu);
                bool any = false;
                const bool may = !is_gpu || gpu_may_claim();
                for (int i = 0; i < total; i++) {
                    if (claimed[i]) continue;
                    any = true;
                    if (may && tile_ready(i)) { claimed[i] = 1; ++nclaimed; *out = i; return 1; }
                }
                if (!any) return 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(4));   // まだ確定していない。UNet が進むのを待つ
        }
    };
    std::mutex mu;                 // retry / werr の保護
    std::vector<int> retry;        // GPU で失敗したタイル（NPU でやり直す）
    std::string werr;
    // 合成（タイルの出力に重みを掛けて足す）は専用のスレッドで行い、NPU は合成を待たずに次のタイルを実行する。
    // 重みは「ランプ（端から 128 画素で 0→1）÷ その画素の重みの合計」。合計はタイルの配置だけで決まるので、最初に 1 回だけ求める（最後の割り算が要らない）。
    // 各ワーカーは出力の置き場（Slot）を 2 つ持ち、合成が済んだ方を次のタイルに使う。
    struct Slot { std::vector<float> out = std::vector<float>(3 * 512 * 512); int idx = -1; bool queued = false; };
    std::mutex q_mu;
    std::condition_variable q_cv;
    std::deque<Slot*> queue;
    bool q_closed = false;
    // 8bit の RGB も合成のスレッドが作る: 画像の行ごとに、その行に掛かるタイルが全部合成し終わったら、すぐ変換する（最後にまとめて変換しない）。
    // 値は 0 以上なので、四捨五入は +0.5 の切り捨てで std::lround と同じ（関数呼び出しを避ける）
    res.img.w = W; res.img.h = H; res.img.rgb.resize(static_cast<size_t>(H) * W * 3);
    res.img.argb.resize(static_cast<size_t>(H) * W);
    auto convert_row = [&](int py) {
        uint8_t* dst = res.img.rgb.data() + static_cast<size_t>(py) * W * 3;
        for (int c = 0; c < 3; c++) {
            const float* a = acc.data() + static_cast<size_t>(c) * H * W + static_cast<size_t>(py) * W;
            for (int i = 0; i < W; i++) {
                const float v = std::min(std::max(a[i], -1.f), 1.f);
                dst[static_cast<size_t>(i) * 3 + c] = static_cast<uint8_t>(static_cast<int>((v + 1.f) * 0.5f * 255.f + 0.5f));
            }
        }
        int32_t* argb = res.img.argb.data() + static_cast<size_t>(py) * W;
        for (int i = 0; i < W; i++, dst += 3)
            argb[i] = static_cast<int32_t>(0xFF000000u | (static_cast<uint32_t>(dst[0]) << 16) | (static_cast<uint32_t>(dst[1]) << 8) | dst[2]);
    };
    std::thread acc_thread([&] {
        std::vector<float> inv(static_cast<size_t>(H) * W, 0.f);
        std::vector<int> remain(H, 0);   // 行ごとの、まだ合成していないタイルの数
        for (int idx = 0; idx < total; idx++) {
            const int y = ys[idx / xs.size()], x = xs[idx % xs.size()];
            for (int r = 0; r < 512; r++) {
                float* row = inv.data() + static_cast<size_t>(y * 8 + r) * W + x * 8;
                for (int q = 0; q < 512; q++) row[q] += ramp[r] * ramp[q];
                ++remain[y * 8 + r];
            }
        }
        for (float& v : inv) v = 1.f / std::max(v, 1e-6f);
        for (int py = 0; py < H; py++) if (remain[py] == 0) convert_row(py);   // どのタイルにも掛からない行（無いはず）
        while (true) {
            Slot* s = nullptr;
            {
                std::unique_lock<std::mutex> lock(q_mu);
                q_cv.wait(lock, [&] { return !queue.empty() || q_closed; });
                if (queue.empty()) break;
                s = queue.front(); queue.pop_front();
            }
            const int y = ys[s->idx / xs.size()], x = xs[s->idx % xs.size()];
            for (int r = 0; r < 512; r++) {
                const size_t p0 = static_cast<size_t>(y * 8 + r) * W + x * 8;
                const float* nw = inv.data() + p0;
                for (int c = 0; c < 3; c++) {
                    float* a = acc.data() + static_cast<size_t>(c) * H * W + p0;
                    const float* o = s->out.data() + (static_cast<size_t>(c) * 512 + r) * 512;
                    const float wr = ramp[r];
                    for (int q = 0; q < 512; q++) a[q] += o[q] * (wr * ramp[q] * nw[q]);
                }
                if (--remain[y * 8 + r] == 0) convert_row(y * 8 + r);
            }
            { std::lock_guard<std::mutex> lock(q_mu); s->queued = false; }
            q_cv.notify_all();
        }
    });
    auto take_slot = [&](Slot* slots) {   // 合成が済んでいる置き場を返す（両方とも合成待ちなら待つ）
        std::unique_lock<std::mutex> lock(q_mu);
        q_cv.wait(lock, [&] { return !slots[0].queued || !slots[1].queued; });
        return !slots[0].queued ? &slots[0] : &slots[1];
    };
    auto submit = [&](Slot* s, int idx) {
        { std::lock_guard<std::mutex> lock(q_mu); s->idx = idx; s->queued = true; queue.push_back(s); }
        q_cv.notify_all();
    };
    auto drain = [&](Slot* slots) {   // 置き場を手放す前に、合成が済むのを待つ
        std::unique_lock<std::mutex> lock(q_mu);
        q_cv.wait(lock, [&] { return !slots[0].queued && !slots[1].queued; });
    };
    // 1 タイルを model で実行して、出力を s に入れる。成功で true。
    auto run_tile = [&](QnnModel* m, int idx, std::vector<float>& tin, Slot* s, std::string* e) {
        const int y = ys[idx / xs.size()], x = xs[idx % xs.size()];
        for (int c = 0; c < 4; c++) for (int r = 0; r < 64; r++)
            std::copy(latent.begin() + (c * lh + y + r) * lw + x, latent.begin() + (c * lh + y + r) * lw + x + 64, tin.begin() + (c * 64 + r) * 64);
        return m->Run({{"latent", tin.data()}}, {{"image", s->out.data()}}, e);
    };
    auto worker = [&](QnnModel* m, bool is_main) {
        std::vector<float> tin(4 * 64 * 64);
        Slot slots[2];
        struct Drain { std::function<void()> f; ~Drain() { f(); } } drain_at_exit{[&] { drain(slots); }};
        while (!stop) {
            if (st->cancel) { stop = true; break; }
            if (is_main) {
                // 次のジョブの CLIP / UNet が NPU を使っている間は、GPU が動いていればタイルを任せて NPU は譲る（UNet の 1 ステップを遅らせない）
                while (!stop && !st->cancel && nclaimed.load() < total && fg_busy_.load() > 0 && gpu_alive.load())
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));   // UNet が終わったら、すぐ NPU で始める（20ms 間隔だと最大 20ms 出遅れた）
                if (st->cancel) { stop = true; break; }
            }
            int idx = -1;
            const int got = claim(&idx, !is_main);
            if (got < 0) { if (st->cancel) stop = true; break; }
            if (got == 0) break;
            std::string e;
            Slot* s = take_slot(slots);
            const auto t_start = std::chrono::steady_clock::now();
            if (!run_tile(m, idx, tin, s, &e)) {
                std::lock_guard<std::mutex> lock(mu);
                if (is_main) { werr = e; stop = true; }
                else { retry.push_back(idx); gpu_alive = false; }   // GPU は諦めて、このタイルは NPU に任せる
                break;
            }
            submit(s, idx);
            ++st->done;
            __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "[%ld ms] vae tile %d/%d done on %s (%.0f ms)  fg_busy=%d", LogMs(), st->done.load(), total, is_main ? "NPU" : "GPU",
                                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count(), fg_busy_.load());
            if (!is_main) {
                // GPU 側だけ、使用率の上限に合わせて休む（OS の描画に GPU を空ける）。休み中もキャンセルに反応する
                const int cap = g_gpu_cap.load();
                if (cap < 100) {
                    const double busy_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
                    double rest_ms = busy_ms * (100 - cap) / cap;
                    while (rest_ms > 0 && !st->cancel && !stop) {
                        const double step = std::min(rest_ms, 20.0);
                        std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(step * 1000)));
                        rest_ms -= step;
                    }
                }
                UpdateMs(g_vae_gpu_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count());   // 休んだ分を含む
            } else if (fg_busy_.load() == 0) {   // NPU が UNet と競合していないときの所要だけを使う
                UpdateMs(g_vae_npu_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count());
            }
        }
    };
    std::thread gpu_thread;
    if (gvae) gpu_thread = std::thread(worker, gvae.get(), false);
    worker(vae.get(), true);
    if (gpu_thread.joinable()) gpu_thread.join();
    {   // GPU が失敗したタイルを NPU で処理する
        std::vector<float> tin(4 * 64 * 64);
        Slot slots[2];
        for (int idx : retry) {
            if (stop) break;
            std::string e;
            Slot* s = take_slot(slots);
            if (!run_tile(vae.get(), idx, tin, s, &e)) { werr = e; stop = true; break; }
            submit(s, idx);
            ++st->done;
        }
        drain(slots);
        if (!retry.empty()) gpu_failed_ = true;
    }
    { std::lock_guard<std::mutex> lock(q_mu); q_closed = true; }   // 合成のスレッドは、残りを片付けてから終わる
    q_cv.notify_all();
    acc_thread.join();
    if (st->cancel) { res.err = "キャンセルしました"; return res; }
    if (!werr.empty()) { res.err = werr; return res; }
    // 合成と RGB への変換は、合成のスレッドが終えている（acc_thread.join の時点で全部の行が変換済み）
    __android_log_print(ANDROID_LOG_INFO, "dsxl-time", "[%ld ms] vae all tiles done and converted (%d, %dx%d)", LogMs(), total, W, H);
    res.ok = true;
    return res;
}

bool Pipeline::LaunchDecode(uint64_t ticket, std::shared_ptr<DecodeSource> src, int lh, int lw, bool gpu_vae,
                            std::shared_ptr<DecodeState>* state_out, std::shared_future<DecodeResult>* fut_out, std::string* err) {
    auto vae = Acquire("vae_64.bin", err);
    if (!vae) return false;
    auto gvae = gpu_vae ? AcquireGpuVae() : nullptr;   // GPU が使えるときは NPU と分担する
    // 先に走っている VAE があれば、その終了を待つ（NPU / GPU の VAE を 2 つ同時には動かさない）
    std::shared_future<DecodeResult> prev;
    { std::lock_guard<std::mutex> lock(decode_mu_); prev = last_decode_; }
    if (prev.valid()) prev.wait();
    auto state = std::make_shared<DecodeState>();
    auto fut = std::async(std::launch::async, [this, vae, gvae, src, lh, lw, state]() {
        lib_.Vote();
        DecodeResult r = DecodeImpl(vae, gvae, src, lh, lw, state.get());
        lib_.ReleaseVotes(&fg_busy_);   // VAE が終わった。次のジョブが CLIP / UNet を始めていなければ、投票を解除して待機中は通常のクロックに戻す
        return r;
    }).share();
    {
        std::lock_guard<std::mutex> lock(decode_mu_);
        decodes_[ticket] = {fut, state};
        last_decode_ = fut;
        last_state_ = state;
    }
    if (state_out) *state_out = state;
    if (fut_out) *fut_out = fut;
    return true;
}

// latent が全部できてから VAE を始める（普通の経路）
uint64_t Pipeline::StartDecode(uint64_t ticket, std::vector<float> latent, int lh, int lw, bool gpu_vae, std::string* err) {
    auto src = std::make_shared<DecodeSource>();
    src->latent = std::move(latent);
    src->rows = lh; src->cols = lw;   // 全部確定済み
    return LaunchDecode(ticket, src, lh, lw, gpu_vae, nullptr, nullptr, err) ? ticket : 0;
}

uint64_t Pipeline::StartDraftDecode(uint64_t ticket, std::vector<float> latent, int lh, int lw, bool gpu_vae, std::string* err) {
    if (!prev_ || !prev_->HasFiles(lh, lw)) {
        __android_log_print(ANDROID_LOG_WARN, "dsxl-pipe", "draft mode: TAESD unavailable for %dx%d (GPU 無効 or model missing) -> VAE", lh, lw);
        return StartDecode(ticket, std::move(latent), lh, lw, gpu_vae, err);
    }
    std::shared_future<DecodeResult> prev;
    { std::lock_guard<std::mutex> lock(decode_mu_); prev = last_decode_; }
    auto state = std::make_shared<DecodeState>();
    state->total = 1;
    auto fut = std::async(std::launch::async, [this, prev, state, latent = std::move(latent), lh, lw, gpu_vae]() mutable {
        if (prev.valid()) prev.wait();   // 先に走っている VAE があれば、その終了を待つ（GPU を同時に使わない）
        DecodeResult r;
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        std::string e;
        if (prev_ && prev_->DecodeFull(latent, lh, lw, &rgb, &w, &h, &e)) {
            r.img.w = w; r.img.h = h; r.img.rgb = std::move(rgb);
            r.ok = true;
            state->done = 1;
        } else {
            // TAESD が使えなかった（コンパイルや実行の失敗）: その場で VAE にする
            __android_log_print(ANDROID_LOG_WARN, "dsxl-pipe", "draft mode: TAESD failed (%s) -> VAE", e.c_str());
            std::string e2;
            auto vae = Acquire("vae_64.bin", &e2);
            if (!vae) { r.err = e2; return r; }
            auto gvae = gpu_vae ? AcquireGpuVae() : nullptr;
            auto src = std::make_shared<DecodeSource>();
            src->latent = std::move(latent);
            src->rows = lh; src->cols = lw;
            lib_.Vote();
            r = DecodeImpl(vae, gvae, src, lh, lw, state.get());
        }
        lib_.ReleaseVotes(&fg_busy_);   // 終わった。次のジョブが CLIP / UNet を始めていなければ、投票を解除して待機中は通常のクロックに戻す
        return r;
    }).share();
    {
        std::lock_guard<std::mutex> lock(decode_mu_);
        decodes_[ticket] = {fut, state};
        last_decode_ = fut;
        last_state_ = state;
    }
    return ticket;
}

// タイル Hires の最終ステップの頭で呼ぶ。latent のバッファを確保し（確定した範囲は 0 から始まる）、確定した範囲の VAE を裏で始める。
// 確定の進み方は、縦長は行・横長は列（by_cols）。GPU 用 VAE が使えないときは false（NPU は UNet と競合するだけなので、普通に UNet のあとで始める）。
bool Pipeline::StartStream(Stream& s, int lh, int lw, bool by_cols) {
    if (!s.gpu_vae || !gpu_lib_ok_ || gpu_failed_) return false;
    if (!AcquireGpuVae()) return false;
    auto src = std::make_shared<DecodeSource>();
    src->latent.assign(static_cast<size_t>(4) * lh * lw, 0.f);
    src->rows = by_cols ? lh : 0;
    src->cols = by_cols ? 0 : lw;
    std::string e;
    if (!LaunchDecode(s.ticket, src, lh, lw, true, &s.state, &s.fut, &e)) {
        __android_log_print(ANDROID_LOG_WARN, "dsxl-pipe", "stream decode not started: %s", e.c_str());
        return false;
    }
    s.src = src;
    s.started = true;
    __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "VAE streaming started in the last Hires step (%s)", by_cols ? "by columns" : "by rows");
    return true;
}

void Pipeline::AbortStream(Stream& s) {
    if (!s.started) return;
    s.state->cancel = true;
    s.fut.wait();
    std::lock_guard<std::mutex> lock(decode_mu_);
    decodes_.erase(s.ticket);
    s.started = false;
}

bool Pipeline::Await(uint64_t ticket, const ProgressFn& progress, Image* out, std::string* err) {
    DecodeEntry e;
    {
        std::lock_guard<std::mutex> lock(decode_mu_);
        auto it = decodes_.find(ticket);
        if (it == decodes_.end()) { if (err) *err = "VAE の結果がありません（解放されたか、すでに受け取り済みです）"; return false; }
        e = it->second;
    }
    int last_done = -1, last_total = -1;
    auto report = [&]() {   // 進捗は呼び出しスレッドからだけ流す（JNI のため）
        const int d = e.state->done.load(), t = std::max(e.state->total.load(), 1);
        if (d != last_done || t != last_total) { last_done = d; last_total = t; progress(PStage::kVae, d, t); }
    };
    report();
    while (e.fut.wait_for(std::chrono::milliseconds(30)) != std::future_status::ready) report();
    report();
    const DecodeResult& r = e.fut.get();
    { std::lock_guard<std::mutex> lock(decode_mu_); decodes_.erase(ticket); }
    if (!r.ok) { if (err) *err = r.err; return false; }
    *out = r.img;
    return true;
}

// ---------------------------------------------------------------------------------------------
uint64_t Pipeline::Begin(const GenParams& p_in, const ProgressFn& progress, std::string* err) {
    std::lock_guard<std::mutex> fg(fg_mu_);
    BusyGuard busy(fg_busy_);
    GenParams p = p_in;
    cancel_ = false;
    if (prev_) prev_->Reset();   // 前のジョブの、処理待ちのプレビューを捨てる
    last_ = p;
    have_base_ = false;
    base_ticket_ = 0;
    base_latent_.clear();
    final_latent_.clear(); final_ready_ = false;
    const uint64_t ticket = ++ticket_counter_;
    base_file_ = GraphKey(p.portrait, false);
    tile_file_ = GraphKey(p.portrait, true);
    if (!HasUnet(dir_)) { if (err) *err = "モデルファイルがありません: " + MissingFiles(dir_); return 0; }
    if (!Exists(AdapterPath(p.lora_adapter))) { if (err) *err = "LoRA の adapter がありません: " + AdapterPath(p.lora_adapter); return 0; }
    lib_.Vote();
    progress(PStage::kText, 1, 1);
    // 条件（プロンプト、CLIP と adapter、TE の強さ）が前回と同じなら、CLIP は実行しない（前回の hidden_ / pooled_ をそのまま使う）
    const std::string key = EncodeKey(p);
    const bool cached = !enc_key_.empty() && key == enc_key_ && !hidden_.empty() && !pooled_.empty();
    Prefetch(base_file_);   // UNet（共有 bin）が載っていなければ読み始める。CLIP は別プロセスなので、並行して動く
    bool encoded = true;
    if (cached) {
        __android_log_print(ANDROID_LOG_INFO, "dsxl-pipe", "encode: cached (CLIP skipped)");
    } else {
        enc_key_.clear();
        encoded = Encode(p.prompt, &hidden_, &pooled_, err);
        if (encoded) enc_key_ = key;
    }
    if (!encoded) return 0;
    if (!Denoise(p, progress, err)) return 0;
    base_ticket_ = ticket;
    have_base_ = true;
    if (!p.hires) return p.draft ? StartDraftDecode(ticket, base_latent_, base_lh_, base_lw_, p.gpu_vae, err)
                                 : StartDecode(ticket, base_latent_, base_lh_, base_lw_, p.gpu_vae, err);
    std::vector<float> latent;
    int lh = 0, lw = 0;
    Stream stream;
    stream.ticket = ticket; stream.gpu_vae = p.gpu_vae;
    // 速度優先のときは、最終ステップの途中から VAE（GPU）を始める仕組みは使わない（VAE を省くので）
    if (!Hires(progress, &latent, &lh, &lw, p.draft ? nullptr : &stream, err)) { AbortStream(stream); return 0; }
    if (stream.started) return ticket;   // タイル Hires の最終ステップの途中から、VAE はもう裏で始まっている
    return p.draft ? StartDraftDecode(ticket, std::move(latent), lh, lw, p.gpu_vae, err)
                   : StartDecode(ticket, std::move(latent), lh, lw, p.gpu_vae, err);
}

uint64_t Pipeline::BeginRefine(uint64_t base_ticket, int hires_steps, double hires_denoise, const ProgressFn& progress, std::string* err) {
    std::lock_guard<std::mutex> fg(fg_mu_);
    BusyGuard busy(fg_busy_);
    if (!have_base_ || base_ticket == 0 || base_ticket != base_ticket_) {
        if (err) *err = "Hires の元になる Base がありません（先に生成してください）";
        return 0;
    }
    cancel_ = false;
    if (prev_) prev_->Reset();
    last_.hires_steps = std::max(1, hires_steps);
    last_.hires_denoise = hires_denoise;
    const uint64_t ticket = ++ticket_counter_;
    lib_.Vote();
    std::vector<float> latent;
    int lh = 0, lw = 0;
    Stream stream;
    stream.ticket = ticket; stream.gpu_vae = last_.gpu_vae;
    if (!Hires(progress, &latent, &lh, &lw, last_.draft ? nullptr : &stream, err)) { AbortStream(stream); return 0; }
    if (stream.started) return ticket;
    return last_.draft ? StartDraftDecode(ticket, std::move(latent), lh, lw, last_.gpu_vae, err)
                       : StartDecode(ticket, std::move(latent), lh, lw, last_.gpu_vae, err);
}

bool Pipeline::RefineRegion(int x, int y, int w, int h, double denoise, const ProgressFn& progress, Image* img, std::string* err) {
    std::lock_guard<std::mutex> fg(fg_mu_);
    BusyGuard busy(fg_busy_);
    if (!final_ready_.load() || final_latent_.empty()) {
        if (err) *err = "領域リファインの元になる Hires の latent がありません（先に Hires まで生成してください）";
        return false;
    }
    const int lh = final_lh_, lw = final_lw_;
    if (!img || img->w != lw * 8 || img->h != lh * 8 || img->rgb.size() != static_cast<size_t>(3) * img->w * img->h) {
        if (err) *err = "画像の寸法が、直前の Hires の latent と合いません（直前の生成の画像だけに掛けられます）";
        return false;
    }
    auto tile = Acquire(tile_file_, err);
    if (!tile) return false;
    const TensorDesc* sd = tile->Input("sample");
    if (!sd || sd->dims.size() != 4) { if (err) *err = "タイル UNet の入力 sample が不正です"; return false; }
    const int th = sd->dims[2], tw = sd->dims[3];
    RefineWindow win;
    if (!PlanRefineWindow(x, y, w, h, img->w, img->h, tw * 8, th * 8, &win, err)) return false;
    cancel_ = false;
    if (!tile->ApplyAdapter(AdapterPath(last_.lora_adapter), err)) return false;   // 生成と同じ super adapter
    lib_.Vote();

    // 窓の latent を、最終 latent から切り出す（ズーム 1 倍）。部分 denoise: latent + noise * sigma0
    std::vector<float> xw(static_cast<size_t>(4) * th * tw);
    const int ly0 = win.y0 / 8, lx0 = win.x0 / 8;
    for (int c = 0; c < 4; c++) for (int r = 0; r < th; r++)
        std::copy(final_latent_.begin() + (static_cast<size_t>(c) * lh + ly0 + r) * lw + lx0, final_latent_.begin() + (static_cast<size_t>(c) * lh + ly0 + r) * lw + lx0 + tw,
                  xw.begin() + (static_cast<size_t>(c) * th + r) * tw);
    static const Schedule sched;
    const std::vector<double> sig = sched.SgmUniformTail(std::max(1, last_.hires_steps), denoise);
    std::mt19937_64 rng(static_cast<uint64_t>(last_.seed) ^ 0xD1B54A32D192ED03ULL ^ (++refine_counter_ * 0x9E3779B97F4A7C15ULL));
    std::normal_distribution<float> nd(0.f, 1.f);
    const float s0 = static_cast<float>(sig[0]);
    for (auto& v : xw) v += nd(rng) * s0;

    // 窓は 1 枚のタイル: 重なりも eps の平均も無い（SampleTiled の 1 枚の場合）。プレビューは出さない
    suppress_preview_ = true;
    const bool ok = SampleTiled(*tile, PStage::kHiresUnet, sig, xw, th, tw, rng, progress, nullptr, err);
    suppress_preview_ = false;
    if (!ok) return false;

    const uint64_t ticket = ++ticket_counter_;
    if (!StartDecode(ticket, std::move(xw), th, tw, last_.gpu_vae, err)) return false;
    Image patch;
    if (!Await(ticket, progress, &patch, err)) return false;
    if (patch.w != win.ww || patch.h != win.wh || patch.rgb.size() != static_cast<size_t>(3) * win.ww * win.wh) { if (err) *err = "窓のデコード結果の寸法が合いません"; return false; }

    BlendRefined(img->rgb.data(), img->w, img->h, win, patch.rgb.data(), RefineParams());
    if (!img->argb.empty()) {   // JNI に渡す argb も、窓の範囲だけ作り直す
        for (int r = 0; r < win.wh; r++) for (int q = 0; q < win.ww; q++) {
            const size_t i = (static_cast<size_t>(win.y0 + r) * img->w + win.x0 + q);
            const uint8_t* p = &img->rgb[i * 3];
            img->argb[i] = static_cast<int32_t>(0xFF000000u | (static_cast<uint32_t>(p[0]) << 16) | (static_cast<uint32_t>(p[1]) << 8) | p[2]);
        }
    }
    return true;
}

bool Pipeline::Generate(const GenParams& p, const ProgressFn& progress, Image* out, std::string* err) {
    const uint64_t t = Begin(p, progress, err);
    return t != 0 && Await(t, progress, out, err);
}

bool Pipeline::Refine(int hires_steps, double hires_denoise, const ProgressFn& progress, Image* out, std::string* err) {
    const uint64_t t = BeginRefine(base_ticket_.load(), hires_steps, hires_denoise, progress, err);
    return t != 0 && Await(t, progress, out, err);
}

}  // namespace dsxl
