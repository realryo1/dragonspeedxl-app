#pragma once
// テキストエンコーダ（CLIP-L / CLIP-bigG）を、UNet とは別のプロセスで動かすための部品。
// cDSP は 32 ビットで、1 プロセス（FastRPC のセッション = PD）が使える仮想アドレス空間は約 3.75GB。4 グラフ共有の UNet（3.74GB）だけでほぼ埋まり、
// 同じプロセスに CLIP（約 1.7GB）を載せると context の作成が失敗する（code 1007 / 1002。失敗すると載っていた context も無効になる。2026-10-18 に実機で踏んだ）。
// セッションはプロセスごとに別なので、CLIP は別プロセス（Android の Service `:clip`）に置き、UNet 側とは UNIX ドメインソケットで話す。
//   サーバー（CLIP のプロセス）: ClipServeAsync(model_dir, lib_dir)。CLIP の context（通常 / LoRA つき）を保持し、要求のたびにエンコードする。
//   クライアント（UNet のプロセス）: ClipClient。プロンプトと LoRA の名前を送り、hidden（77×2048）と pooled（1280）を受け取る。
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace dsxl {

// CLIP の context のファイル名（通常 clip_*.bin、または LoRA つき clip_*_lora.bin）。LoRA 名が空でなく、clip_*_lora.bin と loras/<名前>.clip_*.bin が
// 両方あれば LoRA つきを返し、adapter のファイルを *adapter に入れる。which は "clip_l" か "clip_g"。
std::string ClipFileFor(const std::string& model_dir, const char* which, const std::string& lora, std::string* adapter);

// サーバーを別スレッドで始める（1 プロセスで 1 回だけ。2 回目以降は何もしない）。戻り値は、すでに始まっていたか始められたか。
bool ClipServeAsync(const std::string& model_dir, const std::string& lib_dir);
// サーバーを現在のスレッドで動かす（戻らない）。CLI の clipd 用。
void ClipServe(const std::string& model_dir, const std::string& lib_dir);

class ClipClient {
public:
    ~ClipClient();
    // プロンプトを CLIP-L / bigG でエンコードする。lora は UNet の LoRA 名（空なら通常の CLIP）。te_alpha は CLIP の LoRA の強さ。
    bool Encode(const std::string& prompt, const std::string& lora, float te_alpha, std::vector<float>* hidden, std::vector<float>* pooled, std::string* err);
    // 先に context を読み込んでおく（lora に合う組）。
    bool Warm(const std::string& lora, std::string* err);
    // CLIP の context を手放させる。つながっていなければ何もしない（待たない）。
    void Unload();
    bool Connected() const { return fd_ >= 0; }

private:
    bool Request(int type, const std::string& prompt, const std::string& lora, float te_alpha, std::vector<float>* out, std::string* err);
    bool Connect(int wait_ms);
    void Close();
    std::mutex mu_;
    int fd_ = -1;
};

}  // namespace dsxl
