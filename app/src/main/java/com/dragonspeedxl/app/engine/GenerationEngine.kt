package com.dragonspeedxl.app.engine

import android.graphics.Bitmap
import com.dragonspeedxl.app.lora.LoraUse

/** 解像度の向き。Base は 1280×720 / 720×1280、Hires は ×1.5（1920×1080 / 1080×1920）。 */
enum class Orientation(val label: String, val baseW: Int, val baseH: Int) {
    LANDSCAPE("横 1280×720", 1280, 720),
    PORTRAIT("縦 720×1280", 720, 1280);

    val hiresW: Int get() = baseW * 3 / 2
    val hiresH: Int get() = baseH * 3 / 2
}

enum class Stage(val label: String) {
    TEXT("CLIP"),
    BASE_UNET("Base UNet"),
    UPSCALE("Upscale"),
    HIRES_UNET("Hires UNet"),
    VAE("VAE"),
    LOAD("モデルのロード"),   // ネイティブの PStage::kLoad と順を合わせる（生成の段階ではなく、ロードボタンの進捗）
    MIX("LoRA の合成"),       // アプリ側だけの段階（LoRA を 2 つ以上使うときの adapter の合成。step は 0〜100 の %）。ネイティブからは来ない
}

data class GenerationRequest(
    val prompt: String,
    val orientation: Orientation,
    val seed: Long,
    /** true なら Base の後にそのまま Hires まで実行する（hiresSteps が 0 のときは false） */
    val hires: Boolean,
    /** Base のステップ数（LCM）。既定 2（SSXL の base steps） */
    val steps: Int = 2,
    /** Hires のステップ数。既定 4（SSXL の hires steps）。0 なら Base のみ */
    val hiresSteps: Int = 4,
    /** Hires の denoise。既定 0.39 */
    val hiresDenoise: Double = 0.39,
    /** 速度優先: 最終の latent を VAE ではなく TAESD（GPU）でデコードする（約 2 秒速い。細部は少し甘い）。TAESD のモデルが無いときは VAE。 */
    val draft: Boolean = false,
    /**
     * エンジンに渡す LoRA の adapter: 名前（models/loras/<名前>.4g.bin）。空なら LoRA なし（default の adapter）。縦横・Hires のどれでも使える。
     * LoRA を 2 つ以上使うときは、合成した adapter（"mix/<鍵>" = loras/mix/<鍵>.4g.bin）。生成の直前に loras から決める。
     */
    val lora: String = "",
    /** LoRA の実行時の alpha 入力（強さ w × その LoRA の係数 c。合成した adapter は、その係数 C）。 */
    val loraAlpha: Float = 0f,
    /** テキストエンコーダ（CLIP）の LoRA の強さ w（係数 c は掛けない。合成した adapter は 1）。CLIP の LoRA（clip_*_lora.bin と loras/<名前>.clip_*.bin）があるときだけ使われる。 */
    val loraTeAlpha: Float = 0f,
    /** 使う LoRA と強さ（選んだ順。強さ 0 は入れない）。履歴と生成情報に残すもの。lora / loraAlpha / loraTeAlpha はこれから決める。 */
    val loras: List<LoraUse> = emptyList(),
)

data class StageTiming(val stage: Stage, val millis: Long)

data class Progress(val stage: Stage, val step: Int, val totalSteps: Int)

class GenerationResult(
    val request: GenerationRequest,
    val bitmap: Bitmap,
    val timings: List<StageTiming>,
    /** この画像が Hires 済みか */
    val hiresDone: Boolean,
    /** Base の latent をエンジン側に保持している場合のハンドル（後から Hires 化するため）。0 は無し */
    val baseHandle: Long = 0L,
) {
    val totalMillis: Long get() = timings.sumOf { it.millis }
}

/**
 * 生成の途中経過。CLIP・UNet（Hires まで）は終わっていて、VAE は裏で動いている。
 * 画像は GenerationEngine.await で受け取る。この間に次の begin を呼べる。
 */
class PendingGeneration(
    val request: GenerationRequest,
    /** ネイティブ側のジョブ番号 */
    val ticket: Long,
    val hiresDone: Boolean,
    /** この結果に引き継ぐ段階時間（Hires 化のときの、Base 側の CLIP / Base UNet の時間） */
    val carriedTimings: List<StageTiming>,
    /** Hires を掛けていない Base のとき、後から Hires 化するための Base の ticket（0 は掛けられない） */
    val baseTicket: Long,
    internal val tracker: Any,
)

/** 生成エンジンの共通インターフェース。実装は NativeEngine（QNN HTP / GPU）。 */
interface GenerationEngine {
    val name: String

    /** 準備（エンジンの初期化）。成功なら null、失敗なら理由を返す。 */
    suspend fun prepare(): String?

    /** Base（と、request.hires なら Hires まで）の CLIP・UNet を実行する。VAE は裏で始まり、結果は await で受け取る。 */
    suspend fun begin(request: GenerationRequest, onProgress: (Progress) -> Unit): PendingGeneration

    /** Base のみの結果から、同じ seed・プロンプトで Hires の UNet だけを実行する。VAE は裏で始まる。 */
    suspend fun beginRefine(base: GenerationResult, hiresSteps: Int, hiresDenoise: Double, draft: Boolean, onProgress: (Progress) -> Unit): PendingGeneration

    /** begin / beginRefine の VAE が終わるまで待って、結果を受け取る（待つ間 VAE の進捗を onProgress に流す）。 */
    suspend fun await(pending: PendingGeneration, onProgress: (Progress) -> Unit): GenerationResult

    /** 実行中の生成（CLIP・UNet）を中断する。裏の VAE は止めない。 */
    fun cancel() {}

    fun release()
}
