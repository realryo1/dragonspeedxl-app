#pragma once
// QNN HTP の薄いラッパ。context binary（mmap）を読み込み、float32 / int32 の入出力で graphExecute する。
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"

namespace dsxl {

struct TensorDesc {
    std::string name;
    Qnn_Tensor_t tensor;          // バイナリ内のメタデータの複製（clientBuf だけ実行時に差し替える）
    std::vector<uint32_t> dims;   // tensor.dimensions が指す実体
    Qnn_DataType_t dtype = QNN_DATATYPE_UNDEFINED;
    size_t bytes = 0;
};

class QnnBackendLib {
public:
    ~QnnBackendLib();
    // libQnnHtp.so / libQnnSystem.so を lib_dir から dlopen し、backend と device を作る。burst 相当の性能設定も行う。
    // gpu=true なら libQnnGpu.so（Adreno）を使う。device と burst 設定は無し。
    bool Init(const std::string& lib_dir, std::string* err, bool gpu = false);
    // FP32 の DLC を GPU 用にコンパイル（FP16 精度）し、context binary として out_bin に保存する（初回のみ。数秒かかる）。
    // 以降は out_bin を QnnModel::Load で読み込める。graph 名は DLC のファイル名（拡張子なし）。
    bool BuildGpuContextBinary(const std::string& dlc_path, const std::string& out_bin, std::string* err);

    bool is_gpu = false;             // GPU バックエンドか（GPU のコンテキストは低優先度で作る）
    QnnInterface_t iface{};          // プロバイダ（DLC の compose に渡す）
    QNN_INTERFACE_VER_TYPE qnn{};
    QNN_SYSTEM_INTERFACE_VER_TYPE sys{};
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_LogHandle_t log_ = nullptr;

    // この backend（NPU または GPU）で graphExecute を実行していた累積時間。使用率メーター用（実行中の分も含む）。
    uint64_t BusyNs() const;
    std::atomic<uint64_t> busy_ns{0};
    std::atomic<uint64_t> run_start_ns{0};   // 実行中ならその開始時刻、そうでなければ 0

    // HTP の電力・クロックの投票（DCVS 無効・コーナー最大・スリープ無効・ポーリング）。何度でも呼べる（どのスレッドからでも可）。
    // device 作成直後だけでなく、コンテキスト作成後と生成のたびにも呼び直す（作成前の投票が捨てられる報告があるため）。GPU では何もしない。
    void Vote();
    // 投票を解除する（作った電力設定 ID を全部破棄する。DSP は通常の DCVS に戻る）。待機中に最大クロックのままにしないため、生成が終わったら呼ぶ。
    // busy が渡されて 0 より大きいとき（別のジョブが CLIP / UNet を実行中）は何もしない。Vote と同じ排他で、行き違いは起きない。
    void ReleaseVotes(const std::atomic<int>* busy = nullptr);
    // graphExecute を直列にする（NPU は同時に 1 グラフだけ実行する。裏の VAE と次ジョブの CLIP/UNet が重なっても安全にする）。
    std::mutex exec_mu;

private:
    void PinToPrimeCores();   // 呼び出しスレッドを最高周波数のコアに固定する（Vote から呼ぶ）
    bool hmx_ = false;        // HMX_V2 の投票が通ったか
    void* htp_lib_ = nullptr;
    void* sys_lib_ = nullptr;
    std::mutex vote_mu_;
    std::map<std::thread::id, uint32_t> power_ids_;   // 投票したスレッドごとの電力設定 ID（ID は作ったスレッドからしか使えない）
    bool power_ = false;
};

// context binary（mmap）と、そこから作ったコンテキスト。重みを共有する複数グラフの QnnModel が共有して持ち、最後の 1 つが消えたときに解放する。
struct QnnCtx {
    ~QnnCtx();
    QnnBackendLib* lib = nullptr;
    Qnn_ContextHandle_t ctx = nullptr;
    void* map = nullptr;
    size_t map_size = 0;
};

class QnnModel {
public:
    // 単体のグラフの bin を読む（複数グラフの bin なら最初のグラフ）。
    bool Load(QnnBackendLib* lib, const std::string& bin_path, std::string* err);
    // 複数グラフ（重み共有）の bin を 1 回だけ読み込み、グラフごとに QnnModel を作る（bin 内の順）。コンテキストは全員で共有する。
    // enable_graphs が空でなければ、その名前のグラフだけを読み込む（QNN_CONTEXT_CONFIG_ENABLE_GRAPHS。除いたグラフの実行時バッファが載らない。
    // 4 グラフの UNet で縦 2 つだけにすると DMA-BUF が約 1.4GB 減る。docs/Research.md の 4.4 節）。out にも、その名前のグラフだけが入る。
    static bool LoadAll(QnnBackendLib* lib, const std::string& bin_path, std::vector<std::shared_ptr<QnnModel>>* out, std::string* err,
                        const std::vector<std::string>& enable_graphs = {});
    const std::string& GraphName() const { return graph_name_; }
    // 入力・出力は名前 -> バッファ。サイズは各テンソルの dims×dtype に一致している必要がある。
    bool Run(const std::map<std::string, const void*>& in, const std::map<std::string, void*>& out, std::string* err);
    const TensorDesc* Input(const std::string& n) const;
    const TensorDesc* Output(const std::string& n) const;
    std::string Describe() const;
    // LoRA の adapter（QNN の binary section。UNet は super adapter <名前>.4g.bin、CLIP は <名前>.clip_l.bin など）をこのグラフに適用する。
    // このグラフに同じファイルを適用済みなら何もしない。適用中の adapter は、グラフが消えるか、次の adapter を適用するまで有効。強さは入力 lora_alpha で変える。
    bool ApplyAdapter(const std::string& path, std::string* err);
    ~QnnModel();

private:
    QnnBackendLib* lib_ = nullptr;
    std::shared_ptr<QnnCtx> ctx_;
    Qnn_GraphHandle_t graph_ = nullptr;
    std::string graph_name_;
    std::vector<TensorDesc> in_, out_;
    std::string adapter_;            // 適用中の adapter のファイル
    void* adapter_map_ = nullptr;    // その mmap（適用後も、バックエンドが参照するかもしれないので残す）
    size_t adapter_map_size_ = 0;
};

}  // namespace dsxl
