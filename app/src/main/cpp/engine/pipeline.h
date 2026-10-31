#pragma once
// SDXL（Illustrious）の生成パイプライン: CLIP-L / CLIP-bigG → LCM(sgm_uniform) UNet（Base 720p）→ [latent ×1.5 → タイル Hires 1080p] → VAE（タイル）。すべて QNN HTP。
// UNet は縦横 4 グラフの LoRA つき共有 bin（unet_lora_4g.bin。縦 Base 720x1280・縦タイル 576x1024・横 Base 1280x720・横タイル 1024x576、extended_udma）だけ。
// LoRA を使わないときも同じ bin に、重みが 0 の default の super adapter を適用する（2026-10-18 に、個別ファイル・全体 Hires・LoRA の別 context などを外した。
// 外した方式は docs/Sagyou.md の 2026-10-18 の付録）。
#include <atomic>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "engine/clip_service.h"
#include "engine/preview.h"
#include "engine/qnn_model.h"

namespace dsxl {

// Kotlin の Stage 列挙の順（TEXT, BASE_UNET, UPSCALE, HIRES_UNET, VAE, LOAD）に合わせる
enum class PStage : int { kText = 0, kBaseUnet = 1, kUpscale = 2, kHiresUnet = 3, kVae = 4, kLoad = 5 };
using ProgressFn = std::function<void(PStage stage, int step, int total)>;

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;  // w*h*3
    std::vector<int32_t> argb; // w*h（0xFFRRGGBB）。VAE が rgb と一緒に作る。空でなければ JNI はこれをそのまま Java に渡す（TAESD の経路は空）
};

struct GenParams {
    std::string prompt;
    int64_t seed = 0;
    int steps = 2;  // Base のステップ数
    bool hires = false;  // true なら Base の後に続けて Hires（タイル、1920x1080）まで実行する
    int hires_steps = 4;
    double hires_denoise = 0.39;
    bool portrait = false;       // 縦長（720x1280 → 1080x1920）。false は横長（1280x720 → 1920x1080）
    bool gpu_vae = true;  // VAE のタイルを NPU と GPU（Adreno）で分担する。GPU 用モデルが無い・失敗したときは NPU のみ
    // 速度優先: 最終の latent を VAE ではなく TAESD（GPU）でデコードする（約 2 秒速い。細部は少し甘い）。TAESD のモデルが無い・使えないときは VAE。
    bool draft = false;
    // LoRA: lora_adapter（loras/<名前>.4g.bin の <名前>）が空でなければ、その super adapter を適用する。空なら default の super adapter（重みが 0）。
    // lora_alpha は実行時の alpha 入力（強さ w × その LoRA の係数 c）。
    std::string lora_adapter;
    float lora_alpha = 0.f;
    // テキストエンコーダ（CLIP）の LoRA の強さ w（lora_alpha の入力。UNet と違って係数 c は掛けない。adapter の B に alpha/rank が入っている）。
    // models/clip_l_lora.bin と loras/<名前>.clip_l.bin（bigG は clip_g_lora.bin と .clip_g.bin）があるときだけ、その CLIP に適用する。無ければ通常の CLIP。
    float lora_te_alpha = 0.f;
};

// ModelState() の戻り値のビット（1 << 4 は以前の全体 Hires UNet。空き番）
enum ModelBit : int {
    kBitClipL = 1 << 0, kBitClipG = 1 << 1, kBitVae = 1 << 2, kBitBaseUnet = 1 << 3, kBitGpuVae = 1 << 5,
    kBitLoading = 1 << 6,   // どれかのロードが進行中
    kBitTile = 1 << 7,      // タイル Hires 用の UNet（グラフ）
};

class Pipeline {
public:
    ~Pipeline() { DropModels(); }
    // model_dir に次を置く: clip_l.bin clip_g.bin vae_64_w8a16.bin（または vae_64.bin）text_projection.f32 clip_vocab.tsv clip_merges.txt、
    // UNet の unet_lora_4g.bin と、LoRA を使わないときの adapter unet_lora_4g_default.bin。LoRA の adapter は loras/<名前>.4g.bin
    bool Init(const std::string& model_dir, const std::string& lib_dir, std::string* err);

    // ---- 生成（VAE を裏で実行する形）----
    // CLIP → Base UNet →（p.hires なら）Hires UNet まで実行し、VAE のデコードは裏のスレッドで始めて ticket を返す（失敗は 0）。
    // 返った時点で NPU は空いているので、次の Begin を続けて呼べる（前の VAE は次のジョブの計算と重なって進む）。
    // 画像は Await() で受け取る。Base の latent と条件付けを保持するので、後から BeginRefine() で Hires だけ実行できる。
    uint64_t Begin(const GenParams& p, const ProgressFn& progress, std::string* err);
    // base_ticket（Begin が返した、Hires を掛けていない Base の ticket）の latent に Hires（latent ×1.5 → hires_steps 段）を掛け、VAE を裏で始める。
    uint64_t BeginRefine(uint64_t base_ticket, int hires_steps, double hires_denoise, const ProgressFn& progress, std::string* err);
    // ticket の VAE が終わるまで待ち、画像を受け取る。待っている間 progress(kVae, 済み, 全体) を呼ぶ（呼び出しスレッドから）。
    bool Await(uint64_t ticket, const ProgressFn& progress, Image* out, std::string* err);

    // ---- 領域リファイン（細部・線の明瞭化。docs/refine.md）----
    // 直前に生成した Hires の画像 img（Await が返した最終画像。この関数が書き換える）の矩形 (x, y, w, h)（画素）を、窓（タイルのグラフと同じ大きさ。
    // 縦なら 576x1024）で描き直し（LCM、hires_steps 段、denoise。重なりなし）、VAE で戻して、高周波だけを元の画像へ移植して貼る。
    // 窓は領域を余白つきで収める位置に決める。領域が大きすぎる（縦なら 536x984 を超える）と失敗する。UNet はタイルのグラフ 1 つ、同じ LoRA の adapter。
    // 直前の生成の画像でなければ（latent の寸法が合わなければ）失敗する。同じ画像に何度でも続けて呼べる（領域ごとに）。
    // 窓 1 枚で約 2 秒（推定。UNet 約 1 秒、VAE 約 1 秒）。生成中のプレビューは出さない。失敗のあとは ReleaseVotes を呼ぶ。
    bool RefineRegion(int x, int y, int w, int h, double denoise, const ProgressFn& progress, Image* img, std::string* err);
    // 領域リファインができるか（直前の生成が Hires まで終わっていて、最終 latent を持っている）
    bool CanRefineRegion() const { return final_ready_.load(); }
    // 同期版（Begin + Await）。CLI 用。
    bool Generate(const GenParams& p, const ProgressFn& progress, Image* out, std::string* err);
    bool Refine(int hires_steps, double hires_denoise, const ProgressFn& progress, Image* out, std::string* err);
    // base_ticket の Base latent をまだ保持していて、Hires を掛けられるか
    // 電力の投票を解除する（失敗・キャンセルで VAE まで行かなかったとき用。他のジョブが実行中なら何もしない）
    void ReleaseVotes() { lib_.ReleaseVotes(&fg_busy_); }
    bool CanRefine(uint64_t base_ticket) const { return base_ticket != 0 && base_ticket == base_ticket_.load() && have_base_.load(); }
    // 実行中のジョブ（CLIP / UNet）を中断する。裏の VAE は止めない。
    void Cancel() { cancel_ = true; }

    // ---- 生成中のプレビュー（TAESD。GPU で別スレッド）----
    // UNet の各ステップで予測した x0 を絵にして fn に渡す（速度優先のときだけ、最終ステップは、すぐ最終画像が出るので除く）。fn はワーカースレッドから呼ばれる。
    // 空の fn で無効。GPU（libQnnGpu.so と vae_64_fp32.dlc）が使えないとき、または TAESD のモデルが無いときは、何も出ない（生成には影響しない）。
    void SetPreviewFn(Previewer::Fn fn);
    // 直前の Begin の条件の draft（速度優先）だけを差し替える。BeginRefine の前に呼ぶ（Hires 化のデコードの方式を、今の設定にするため）。
    void SetDraft(bool draft) { std::lock_guard<std::mutex> lock(fg_mu_); last_.draft = draft; }

    // ---- モデルの常駐 ----
    // CLIP・VAE・UNet（共有 bin。縦横の全グラフが 1 つの context）・GPU 用 VAE を、すべて読み込んで常駐させる。progress(kLoad, 済み, 全体)。
    // 戻り値が空なら成功。失敗は "!" で始まる文章。それ以外は注記（GPU 用 VAE が使えない、など）。
    std::string LoadModels(bool portrait, const ProgressFn& progress);
    // ジョブの完了後に呼ぶ先読み（SSXL のプリフェッチ）。足りないモデルを裏で読み込み始めるだけで、待たない。
    void PrefetchNext(bool portrait);
    // 常駐モデルの状態（ModelBit の論理和）。待たない。
    int ModelState(bool portrait);
    // 常駐しているモデルをすべて解放する（解放ボタン・アプリ終了用）。裏の VAE が終わるまで待つ。
    void DropModels();

    // VAE を GPU で分担するとき、GPU の使用率の上限（10〜100 %。100 は制限なし）。GPU でタイルを 1 つ処理するたびに、
    // 処理時間 ×（100−上限）÷上限 だけ休んで、OS の描画などに GPU を空ける。生成中でも変更できる。
    static void SetGpuCap(int percent);
    // メモリ節約モード。オンのとき UNet は今の向きの 2 グラフだけを読み（DMA-BUF が約 5.4GB → 4.0GB）、向きが変わるとき（次の生成の始めに）読み直す（約 2 秒）。
    // 切り替えは、次の生成で UNet を読み込むとき（先読みを含む）から効き、そのとき載っている UNet を手放して読み直す。
    static void SetMemorySaver(bool on);
    // 使用率メーター用: NPU / GPU で実行していた累積時間（ns）と現在時刻（ns, steady clock）
    void BusyNs(uint64_t* npu, uint64_t* gpu, uint64_t* now) const;
    // 足りない必須ファイル（UNet と default の adapter を含む）。全部あれば空
    static std::string MissingFiles(const std::string& model_dir);
    static const char* UnetFile() { return "unet_lora_4g.bin"; }               // 縦横 4 グラフの LoRA つき共有 bin
    static const char* DefaultAdapter() { return "unet_lora_4g_default.bin"; }  // LoRA を使わないときの super adapter（重みが 0）
    static bool HasUnet(const std::string& model_dir);

private:
    bool Encode(const std::string& prompt, std::vector<float>* hidden, std::vector<float>* pooled, std::string* err);
    bool Denoise(const GenParams& p, const ProgressFn& progress, std::string* err);   // Base。結果は base_latent_ に入る
    // ---- VAE（裏で実行）----
    struct DecodeState {            // 進捗の共有（Await がポーリングして progress に流す）
        std::atomic<int> done{0}, total{0};
        std::atomic<bool> cancel{false};
    };
    struct DecodeResult { bool ok = false; Image img; std::string err; };
    struct DecodeEntry { std::shared_future<DecodeResult> fut; std::shared_ptr<DecodeState> state; };
    // VAE が読む latent（lh×lw×4）と、確定した範囲。rows / cols は「先頭から何行 / 何列が確定済みか」（全部確定なら lh / lw）。
    // 確定した範囲に収まる VAE タイル（64×64）から順に、裏のスレッドがデコードを始める（タイル Hires の最終ステップの途中から）。
    struct DecodeSource {
        std::vector<float> latent;
        std::atomic<int> rows{0}, cols{0};
    };
    // タイル Hires の最終ステップで、終わった範囲から VAE を始めるための状態（Begin / BeginRefine が持つ）
    struct Stream {
        bool started = false;
        uint64_t ticket = 0;
        bool gpu_vae = true;
        std::shared_ptr<DecodeSource> src;
        std::shared_ptr<DecodeState> state;
        std::shared_future<DecodeResult> fut;
    };

    bool Hires(const ProgressFn& progress, std::vector<float>* latent, int* lh, int* lw, Stream* stream, std::string* err);
    // 初期値 x（ノイズ済み）を sigma 列に沿って LCM でデノイズする。latent の寸法は UNet の sample 入力から決まる。
    bool Sample(QnnModel& unet, PStage stage, const std::vector<double>& sig, std::vector<float>& x, std::mt19937_64& rng,
                const ProgressFn& progress, std::string* err);
    // タイル Hires（MultiDiffusion）。x（latent lh×lw）を、タイル用 UNet（sample 入力の寸法がタイル）を重ねて並べて覆い、
    // 1 ステップごとに各タイルの eps を線形フェザーで重み付き平均して合成する。time_ids は全体の寸法（最終画像のサイズ）。
    // stream を渡すと、最終ステップは「タイル 1 列（縦長は 1 行、横長は 1 列）」ずつ確定させ、確定した範囲の VAE を GPU で先に始める。
    bool SampleTiled(QnnModel& unet, PStage stage, const std::vector<double>& sig, std::vector<float>& x, int lh, int lw, std::mt19937_64& rng,
                     const ProgressFn& progress, Stream* stream, std::string* err);
    // 条件を整えて VAE を裏のスレッドで始める（先に走っている VAE があれば、それが終わるのを待つ。同時に動かすのは 1 つだけ）。
    uint64_t StartDecode(uint64_t ticket, std::vector<float> latent, int lh, int lw, bool gpu_vae, std::string* err);
    // 速度優先: VAE の代わりに TAESD（GPU）でフルサイズにデコードする（裏のスレッド）。モデルのファイルが無い・実行に失敗したときは、その場で VAE にする。
    uint64_t StartDraftDecode(uint64_t ticket, std::vector<float> latent, int lh, int lw, bool gpu_vae, std::string* err);
    bool LaunchDecode(uint64_t ticket, std::shared_ptr<DecodeSource> src, int lh, int lw, bool gpu_vae,
                      std::shared_ptr<DecodeState>* state, std::shared_future<DecodeResult>* fut, std::string* err);
    // ストリームを始める。GPU 用 VAE が使えないときは false（普通に UNet のあとで始める）
    bool StartStream(Stream& s, int lh, int lw, bool by_cols);
    void AbortStream(Stream& s);   // UNet が失敗・キャンセルしたとき、始めてしまった VAE を止めて片付ける
    DecodeResult DecodeImpl(std::shared_ptr<QnnModel> vae, std::shared_ptr<QnnModel> gvae, std::shared_ptr<DecodeSource> src, int lh, int lw, DecodeState* st);
    void StopDecodes();
    std::mutex decode_mu_;
    std::map<uint64_t, DecodeEntry> decodes_;
    std::shared_future<DecodeResult> last_decode_;   // 直近に始めた VAE（次の VAE が待つ相手）
    std::shared_ptr<DecodeState> last_state_;
    std::atomic<int> fg_busy_{0};   // CLIP / UNet を実行中のジョブ数。>0 の間、裏の VAE は NPU を使わず GPU に任せる（GPU があるとき）

    // ---- モデルの常駐・先読み ----
    // CLIP / VAE / UNet（共有 bin 3.74GB。縦横の 4 グラフが 1 つの context）はすべて常駐。向きを切り替えても読み直さない。
    // ロードは別スレッドで先に始められる（直前の段階の NPU 計算と重ねる）。cache_ は cache_mu_ で守る（待つときはロックを手放す）。
    struct LoadResult { std::shared_ptr<QnnModel> model; std::string err; };
    void Prefetch(const std::string& file);
    std::shared_future<LoadResult> FutureOf(const std::string& file);   // 無ければ空（valid() == false）
    std::shared_ptr<QnnModel> Acquire(const std::string& file, std::string* err);
    bool WaitLoaded(const std::string& file, std::string* err);         // 失敗したらキャッシュから外す（他は手放さない）
    void Evict(const std::string& file);
    void DropCache();   // キャッシュのモデルをすべて手放す（ロード中のものは終わるまで待つ。裏の VAE は止めない）
    // CLIP（CLIP-L・bigG。通常 / LoRA つき）は別プロセスの :clip サービスが持つ。UNet（3.74GB）と同じプロセスには載らない（DSP の空間が約 3.75GB。
    // 載せようとするとあとから作る方が失敗し、DSP が作り直されて、載っていた context が無効になる。2026-10-18 に実機で踏んだ）。clip_service.h を見る。
    // 使うなら CLIP の LoRA のファイル（clip_*_lora.bin と adapter）の有無でエンコードの条件が変わるので、EncodeKey に入れる。
    std::string ClipFile(const char* which, const GenParams& p, std::string* adapter) const;   // which は "clip_l" か "clip_g"
    ClipClient clip_;
    // UNet に適用する super adapter のファイル（LoRA なら loras/<名前>.4g.bin、無ければ unet_lora_4g_default.bin）
    std::string AdapterPath(const std::string& lora) const;
    // UNet のグラフのキャッシュのキー（"unet:base:p" など）と、共有 bin の中でそのグラフを見分ける latent の寸法
    static std::string GraphKey(bool portrait, bool tile);
    static bool GraphDims(const std::string& key, int* lh, int* lw);
    // 共有 bin（複数グラフ）の読み込み結果。UNet のグラフのキーの読み込みは、ここからグラフを取り出す形になる。
    struct GroupResult { std::vector<std::shared_ptr<QnnModel>> models; std::string err; };
    // 共有 bin から、latent の寸法 lh×lw のグラフを key で取り出す
    void PrefetchGraph(const std::string& key, int lh, int lw);
    std::mutex cache_mu_;
    std::map<std::string, std::shared_future<LoadResult>> cache_;
    std::map<std::string, std::shared_future<GroupResult>> groups_;   // キーは共有 bin のファイル名
    // GPU（QNN GPU バックエンド）で動かす VAE。vae_64_fp32.dlc を初回だけ GPU 用にコンパイルして vae_64_gpu.bin に保存する。
    void PrefetchGpuVae();
    std::shared_ptr<QnnModel> AcquireGpuVae();   // 使えなければ nullptr（以降は試さない）
    QnnBackendLib gpu_lib_;
    bool gpu_lib_ok_ = false;
    std::atomic<bool> gpu_failed_{false};
    // プレビュー（TAESD）。gpu_lib_ のあとに宣言する（先に壊れて、ワーカースレッドが止まってから gpu_lib_ が壊れる）
    std::unique_ptr<Previewer> prev_;
    std::atomic<bool> suppress_preview_{false};   // 領域リファイン（窓の latent）のあいだは、生成中のプレビューを出さない
    void EmitPreview(const std::vector<float>& denoised, int lh, int lw) { if (!suppress_preview_ && prev_ && prev_->Enabled()) prev_->Submit(denoised, lh, lw); }

    std::string dir_;
    QnnBackendLib lib_;
    std::atomic<bool> cancel_{false};

    // 直前の Begin の状態（BeginRefine 用）。fg_mu_ の中でだけ触る（have_base_ は読むだけ他スレッドからも）。
    std::mutex fg_mu_;
    std::atomic<uint64_t> ticket_counter_{0};
    std::atomic<uint64_t> base_ticket_{0};
    std::atomic<bool> have_base_{false};
    GenParams last_;
    std::string base_file_, tile_file_;   // 直前の Begin の向きの UNet のグラフのキー
    std::vector<float> hidden_, pooled_, base_latent_;
    // 直前の Hires の最終 latent（4 x final_lh_ x final_lw_。縦 1080x1920 で約 0.5MB）。領域リファインが窓を切り出す。新しい Begin で消える。
    std::vector<float> final_latent_;
    int final_lh_ = 0, final_lw_ = 0;
    std::atomic<bool> final_ready_{false};
    uint64_t refine_counter_ = 0;   // 領域リファインの乱数を、呼ぶたびに変える
    // hidden_ / pooled_ を作った条件（プロンプト、使った CLIP と adapter、TE の強さ）。同じなら CLIP を実行しない（LoRA のときは CLIP の読み込みごと省ける）
    std::string enc_key_;
    std::string EncodeKey(const GenParams& p) const;
    int base_lh_ = 0, base_lw_ = 0;
};

}  // namespace dsxl
