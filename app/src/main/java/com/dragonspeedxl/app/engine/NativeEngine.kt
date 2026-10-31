package com.dragonspeedxl.app.engine

import android.content.Context
import android.graphics.Bitmap
import android.system.Os
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * C++ ネイティブエンジン（QNN HTP）への JNI 橋渡し。CLIP-L / CLIP-bigG / UNet / VAE をすべて NPU で実行する。
 * 横 1280×720 / 縦 720×1280 の Base と、×1.5 のタイル Hires（1920×1080 / 1080×1920）。
 * UNet は縦横 4 グラフの LoRA つき共有 bin（unet_lora_4g.bin）1 つ。LoRA なしのときは default の adapter（unet_lora_4g_default.bin）を適用する。
 */
class NativeEngine(private val context: Context) : GenerationEngine {
    override val name = "NPU（QNN HTP）"

    /** ネイティブからの進捗通知（JNI が onProgress(III)V を呼ぶ）。stage は Stage の ordinal。 */
    fun interface Listener {
        fun onProgress(stage: Int, step: Int, total: Int)
    }

    /** 生成中のプレビュー（TAESD）の通知（JNI が onPreview(II[I)V を、ネイティブのワーカースレッドから呼ぶ）。argb は w×h の ARGB。 */
    fun interface PreviewListener {
        fun onPreview(w: Int, h: Int, argb: IntArray)
    }

    /**
     * 生成中のプレビューの受け口（null ならプレビューなし）。ネイティブのスレッドから呼ばれるので、UI の状態はメインスレッドへ渡し直すこと。
     * begin / beginRefine のたびに、その時点の値が登録される。
     */
    @Volatile var previewSink: ((Bitmap) -> Unit)? = null

    /** モデル置き場: /sdcard/Android/data/<pkg>/files/models/ （adb push で配置。APK には入れない） */
    val modelDir: File get() = File(context.getExternalFilesDir(null), "models")

    private val libDir: String get() = context.applicationInfo.nativeLibraryDir
    private var ready = false

    override suspend fun prepare(): String? = withContext(Dispatchers.Default) {
        if (!loaded) return@withContext "libdsxl_native.so を読み込めません: $loadError"
        if (ready) return@withContext null
        modelDir.mkdirs()
        // Skel を探せるように、QNN 初期化より前に ADSP_LIBRARY_PATH を設定する
        Os.setenv("ADSP_LIBRARY_PATH", libDir, true)
        ClipService.start(context, modelDir.absolutePath, libDir)   // CLIP は別プロセス（UNet と同じプロセスには載らない）
        val err = nativeInit(modelDir.absolutePath, libDir)
        ready = err.isEmpty()
        err.ifEmpty { null }
    }

    /** 設定画面向けの状態表示（モデル配置・QNN ライブラリ・ネイティブ版数）。 */
    fun statusText(): String =
        if (!loaded) "ネイティブライブラリ未ロード: $loadError"
        else nativeBackendInfo(modelDir.absolutePath, libDir)

    /** [NPU の累積実行 ns, GPU の累積実行 ns, 現在時刻 ns]（使用率メーター用）。エンジン未初期化なら null。 */
    fun busy(): LongArray? = if (loaded && ready) nativeBusy() else null

    /** VAE で GPU を NPU と分担するときの GPU 使用率の上限（10〜100 %）。生成中でも反映される。 */
    fun setGpuCap(percent: Int) {
        if (loaded) nativeSetGpuCap(percent)
    }

    /** メモリ節約モード。オンなら UNet を今の向きの 2 グラフだけ読む（DMA-BUF が約 1.4GB 減る。向きが変わると読み直す）。次の生成から効く。 */
    fun setMemorySaver(on: Boolean) {
        if (loaded) nativeSetMemorySaver(on)
    }

    /** UNet（unet_lora_4g.bin）と default の adapter（unet_lora_4g_default.bin）が配置されているか。縦横・Hires はすべてこの 1 つで動く */
    fun hasUnet(): Boolean = loaded && nativeHasUnet(modelDir.absolutePath)

    override suspend fun begin(request: GenerationRequest, onProgress: (Progress) -> Unit): PendingGeneration =
        withContext(Dispatchers.Default) {
            check(ready) { "エンジンが初期化されていません" }
            ClipService.start(context, modelDir.absolutePath, libDir)   // 落とされていたら再開する（動いていれば何もしない）
            check(hasUnet()) { "UNet（unet_lora_4g.bin と unet_lora_4g_default.bin）が配置されていません" }
            val tracker = StageTracker(onProgress)
            registerPreview()
            val ticket = nativeBegin(
                request.prompt, request.seed, request.steps, request.hires, request.hiresSteps, request.hiresDenoise,
                request.orientation == Orientation.PORTRAIT, request.draft, request.lora, request.loraAlpha, request.loraTeAlpha, tracker.listener,
            )
            if (ticket == 0L) throw IllegalStateException(nativeLastError().ifEmpty { "生成に失敗しました" })
            // Hires を掛けていない Base は、あとから Hires 化できる（ネイティブが latent を保持している間）
            PendingGeneration(request, ticket, hiresDone = request.hires, carriedTimings = emptyList(), baseTicket = if (request.hires) 0L else ticket, tracker = tracker)
        }

    /** 直前に生成した Base の latent に Hires を掛ける（エンジン側が latent を保持している間のみ有効）。 */
    override suspend fun beginRefine(base: GenerationResult, hiresSteps: Int, hiresDenoise: Double, draft: Boolean, onProgress: (Progress) -> Unit): PendingGeneration =
        withContext(Dispatchers.Default) {
            check(ready) { "エンジンが初期化されていません" }
            val tracker = StageTracker(onProgress)
            registerPreview()
            val ticket = nativeBeginRefine(base.baseHandle, hiresSteps, hiresDenoise, draft, tracker.listener)
            if (ticket == 0L) throw IllegalStateException(nativeLastError().ifEmpty { "Hires に失敗しました" })
            // Base 側の CLIP / Base UNet の時間は引き継ぎ、VAE は Hires 後のものに置き換える
            val request = base.request.copy(hires = true, hiresSteps = hiresSteps, hiresDenoise = hiresDenoise, draft = draft)
            PendingGeneration(request, ticket, hiresDone = true, carriedTimings = base.timings.filter { it.stage != Stage.VAE }, baseTicket = 0L, tracker = tracker)
        }

    override suspend fun await(pending: PendingGeneration, onProgress: (Progress) -> Unit): GenerationResult =
        withContext(Dispatchers.Default) {
            val tracker = pending.tracker as StageTracker
            tracker.sink = onProgress   // ここからの進捗（VAE）は await 側へ流す
            val arr = nativeAwait(pending.ticket, tracker.listener)
                ?: throw IllegalStateException(nativeLastError().ifEmpty { "VAE に失敗しました" })
            GenerationResult(
                pending.request, toBitmap(arr), pending.carriedTimings + tracker.finish(),
                hiresDone = pending.hiresDone, baseHandle = pending.baseTicket,
            )
        }

    /** ticket の Base latent にまだ Hires を掛けられるか（次のジョブが始まると、Base latent は置き換わる）。 */
    fun canRefine(baseTicket: Long): Boolean = loaded && ready && baseTicket != 0L && nativeCanRefine(baseTicket)

    // ---- モデルの常駐（ロードボタン・先読み・解放）----
    /** 常駐モデルの状態のビット（pipeline.h の ModelBit と同じ）。待たない。 */
    fun modelState(orientation: Orientation): Int =
        if (loaded && ready) nativeModelState(orientation == Orientation.PORTRAIT) else 0

    /**
     * モデルをすべて読み込む（UNet は共有 bin 1 つなので、縦横どちらのグラフも同時に載る）。進捗は onProgress(済み, 全体)。
     * 戻り値は注記（空なら注記なし）。失敗は例外。
     */
    suspend fun loadModels(orientation: Orientation, onProgress: (Int, Int) -> Unit): String = withContext(Dispatchers.Default) {
        check(ready) { "エンジンが初期化されていません" }
        val r = nativeLoadModels(orientation == Orientation.PORTRAIT) { _, step, total -> onProgress(step, total) }
        if (r.startsWith("!")) throw IllegalStateException(r.drop(1))
        r
    }

    /** ジョブ完了後の先読み（SSXL のプリフェッチ）。足りないモデルの読み込みを裏で始めるだけで、待たない。 */
    suspend fun prefetch(orientation: Orientation) = withContext(Dispatchers.Default) {
        if (loaded && ready) nativePrefetch(orientation == Orientation.PORTRAIT)
    }

    /** 常駐モデルをすべて解放する（裏の VAE が終わるまで待つ）。 */
    suspend fun unloadModels() = withContext(Dispatchers.Default) {
        if (loaded && ready) nativeUnload()
    }

    private fun toBitmap(arr: IntArray): Bitmap = Bitmap.createBitmap(arr, 2, arr[0], arr[0], arr[1], Bitmap.Config.ARGB_8888)

    /** 今の previewSink をネイティブに登録する（null なら無効にする）。新しいジョブを始める直前に呼ぶ。 */
    private fun registerPreview() {
        val sink = previewSink
        nativeSetPreviewListener(
            if (sink == null) null else PreviewListener { w, h, argb -> sink(Bitmap.createBitmap(argb, w, h, Bitmap.Config.ARGB_8888)) },
        )
    }

    /** ネイティブからの進捗通知を UI の Progress に中継しつつ、ステージごとの所要時間を集計する。begin から await まで 1 つを使い回す。 */
    private inner class StageTracker(@Volatile var sink: (Progress) -> Unit) {
        private val millis = LongArray(Stage.entries.size)
        private var cur = -1
        private var start = System.nanoTime()
        val listener = Listener { stage, step, total ->
            val now = System.nanoTime()
            if (stage != cur) {
                if (cur >= 0) millis[cur] += (now - start) / 1_000_000
                cur = stage; start = now
            }
            sink(Progress(Stage.entries[stage], step, total))
        }

        fun finish(): List<StageTiming> {
            if (cur >= 0) millis[cur] += (System.nanoTime() - start) / 1_000_000
            return Stage.entries.filter { millis[it.ordinal] > 0 }.map { StageTiming(it, millis[it.ordinal]) }
        }
    }

    override fun cancel() {
        if (loaded && ready) nativeCancel()
    }

    override fun release() {
        if (loaded && ready) { nativeSetPreviewListener(null); nativeRelease() }
        ready = false
    }

    private external fun nativeInit(modelDir: String, nativeLibDir: String): String
    private external fun nativeBackendInfo(modelDir: String, nativeLibDir: String): String
    private external fun nativeHasUnet(modelDir: String): Boolean
    private external fun nativeBegin(
        prompt: String, seed: Long, steps: Int, hires: Boolean, hiresSteps: Int, hiresDenoise: Double,
        portrait: Boolean, draft: Boolean, lora: String, loraAlpha: Float, loraTeAlpha: Float, listener: Listener,
    ): Long
    private external fun nativeBeginRefine(baseTicket: Long, hiresSteps: Int, hiresDenoise: Double, draft: Boolean, listener: Listener): Long
    private external fun nativeAwait(ticket: Long, listener: Listener): IntArray?
    private external fun nativeCanRefine(baseTicket: Long): Boolean
    private external fun nativeLoadModels(portrait: Boolean, listener: Listener): String
    private external fun nativePrefetch(portrait: Boolean)
    private external fun nativeModelState(portrait: Boolean): Int
    private external fun nativeUnload()
    private external fun nativeBusy(): LongArray?
    private external fun nativeSetGpuCap(percent: Int)
    private external fun nativeSetMemorySaver(on: Boolean)
    private external fun nativeSetPreviewListener(listener: PreviewListener?)
    private external fun nativeCancel()
    private external fun nativeLastError(): String
    private external fun nativeRelease()

    private companion object {
        var loaded = false
        var loadError = ""

        init {
            try {
                System.loadLibrary("dsxl_native")
                loaded = true
            } catch (t: Throwable) {
                loadError = t.message ?: t.javaClass.simpleName
            }
        }
    }
}
