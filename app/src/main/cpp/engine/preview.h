#pragma once
// TAESDXL のデコーダ（GPU = Adreno、FP16）。2 つの使い方:
//  1. 生成中のプレビュー: UNet の各ステップで予測した x0 の latent を絵にして返す。専用スレッドで行い、UNet を待たせない（最新のものだけ処理する）。
//  2. 速度優先モード: 最終の latent を、VAE の代わりに TAESD でフルサイズにデコードする（DecodeFull。同期）。
// モデル: models/gpu/taesd_<lh>x<lw>_fp32.dlc（latent の寸法ごと。縦 160x90 / 240x135、横 90x160 / 135x240）。初回だけ GPU 用にコンパイルして taesd_<lh>x<lw>_gpu.bin に保存する。
// 無い・失敗したときは、プレビューが出ないだけで、生成には影響しない。
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "engine/qnn_model.h"

namespace dsxl {

class Previewer {
public:
    // rgb は w*h*3（8bit）。ワーカースレッドから呼ばれる。
    using Fn = std::function<void(int w, int h, const std::vector<uint8_t>& rgb)>;
    // cap: GPU の使用率の上限（%）を返す関数。デコード 1 回ごとに、処理時間 ×（100−上限）÷上限 だけ休む（VAE の GPU と同じ考え方）。
    Previewer(QnnBackendLib* gpu_lib, std::string model_dir, std::function<int()> cap);
    ~Previewer();

    void SetCallback(Fn fn);          // 空にすると無効（Submit は何もしない）
    bool Enabled() const { return enabled_.load(); }
    // latent（NCHW、4×lh×lw。sampler 空間の x0 のまま）。前のものがまだ処理されていなければ、これで置き換える。
    void Submit(const std::vector<float>& latent, int lh, int lw);
    void Reset();                     // 待機中の分を捨てる（新しいジョブの前）
    void DropModels();                // 読み込んだ GPU のモデルを手放す（モデルの解放ボタン用。次の Submit で読み直す）

    // その寸法のモデルのファイル（DLC か、コンパイル済みの bin）があるか。待たない。
    bool HasFiles(int lh, int lw) const;
    // 速度優先モード: latent をフルサイズ（8lh × 8lw）の RGB（8bit）にする。同期。初回はコンパイルで数秒かかる。失敗は false（理由は err）。
    bool DecodeFull(const std::vector<float>& latent, int lh, int lw, std::vector<uint8_t>* rgb, int* w, int* h, std::string* err);

private:
    void Loop();
    // run_mu_ を持った状態で呼ぶ。縮小率 f（1 で等倍、2 で 2×2 の平均）で RGB にする。
    bool DecodeLocked(const std::vector<float>& latent, int lh, int lw, int f, std::vector<uint8_t>* rgb, int* w, int* h, double* ms, std::string* err);
    std::shared_ptr<QnnModel> Model(int lh, int lw, std::string* err);   // run_mu_ を持った状態で。失敗した寸法は以降試さない

    QnnBackendLib* lib_;
    std::string dir_;
    std::function<int()> cap_;

    std::mutex mu_;                   // 待機中の latent・コールバック・停止の守り
    std::condition_variable cv_;
    Fn fn_;
    std::atomic<bool> enabled_{false};
    bool stop_ = false, has_ = false;
    std::vector<float> pending_;
    int plh_ = 0, plw_ = 0;
    std::atomic<bool> drop_{false};

    std::mutex run_mu_;               // モデルの読み込み・実行・共有バッファの守り（ワーカーと DecodeFull が使う）
    std::map<int, std::shared_ptr<QnnModel>> models_;   // キーは lh * 10000 + lw
    std::set<int> failed_;
    std::vector<float> out_;          // 3 × 8lh × 8lw
    std::vector<uint8_t> rgb_;
    std::thread th_;                  // 最後に宣言する（他のメンバが先に初期化されてから動き出す）
};

}  // namespace dsxl
