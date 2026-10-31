package com.dragonspeedxl.app

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.provider.OpenableColumns
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.dragonspeedxl.app.data.HistoryItem
import com.dragonspeedxl.app.data.HistoryStore
import com.dragonspeedxl.app.engine.DeviceMetrics
import com.dragonspeedxl.app.engine.GenerationEngine
import com.dragonspeedxl.app.engine.GenerationRequest
import com.dragonspeedxl.app.engine.GenerationResult
import com.dragonspeedxl.app.engine.Metrics
import com.dragonspeedxl.app.engine.MetricsWindowMs
import com.dragonspeedxl.app.engine.NativeEngine
import com.dragonspeedxl.app.engine.Orientation
import com.dragonspeedxl.app.engine.PendingGeneration
import com.dragonspeedxl.app.engine.Progress
import com.dragonspeedxl.app.engine.RootMetrics
import com.dragonspeedxl.app.engine.Stage
import com.dragonspeedxl.app.prompt.DanbooruClient
import com.dragonspeedxl.app.prompt.PromptChip
import com.dragonspeedxl.app.prompt.PromptChips
import com.dragonspeedxl.app.security.AppLock
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max
import kotlin.random.Random

/** 履歴 1 ページの列数（横）と行数（縦）。既定は 4×4。設定の数値入力で 1〜8。表示枚数は積。変えると 1 ページ目に戻る。 */
const val HISTORY_COLS_DEFAULT = 4
const val HISTORY_ROWS_DEFAULT = 4
const val HISTORY_GRID_MIN = 1
const val HISTORY_GRID_MAX = 8

/** 除外ワードの既定。カンマ区切り。この語句を含む履歴を隠す（スイッチの既定はオン）。 */
const val DEFAULT_IMAGE_BLACKLIST =
    "nsfw, sex, nude, penis, pussy, nipple, uncensored, handjob, fellatio, lactation, futanari, open clothes"

/**
 * 画面の状態と操作。名前は姉妹プロジェクト SSXL の webui（app.js / index.html）に揃えている:
 * seed / hiresDenoise / baseSteps / hiresSteps / hiresScale / historySearch / historyPage / historyDeleteMode /
 * historySelectedIds / blurBlacklist / activeId / statusline、操作は generate / refine（hiresActionBtn）/ useAsBase（流用）/ copyPrompt。
 */
class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val store = HistoryStore(app)
    private val native = NativeEngine(app)
    private val prefs = app.getSharedPreferences("ui", Context.MODE_PRIVATE)

    // ---- 入力（空欄 = 既定値。placeholder に既定値を出す）----
    /** 起動時に入るプロンプト（設定で変えられる。既定は "hatsune miku" だけ）。 */
    var defaultPrompt by mutableStateOf(prefs.getString("defaultPrompt", null) ?: DEFAULT_PROMPT)
        private set
    var prompt by mutableStateOf(defaultPrompt)
    var orientation by mutableStateOf(Orientation.PORTRAIT)
    var seed by mutableStateOf("")            // 空欄ならランダム
    var hiresDenoise by mutableStateOf("")    // 既定 0.39
    var baseSteps by mutableStateOf("")       // 既定 2
    var hiresSteps by mutableStateOf("")      // 既定 4。0 なら Base のみ
    /** Hires の倍率。グラフが固定なので ×1.5 のみ（SSXL の hiresScale）。 */
    val hiresScale = 1.5

    private val baseStepsValue get() = baseSteps.trim().toIntOrNull()?.coerceIn(1, 8) ?: 2
    private val hiresStepsValue get() = hiresSteps.trim().toIntOrNull()?.coerceIn(0, 8) ?: 4
    private val hiresDenoiseValue get() = hiresDenoise.trim().toDoubleOrNull()?.coerceIn(0.05, 1.0) ?: 0.39

    // ---- プロンプトの「戻る」（SSXL の promptUndoBtn）----
    private val promptUndo = ArrayDeque<String>()
    var canUndoPrompt by mutableStateOf(false)
        private set

    private fun pushPromptUndo(old: String) {
        if (old.isBlank() || promptUndo.lastOrNull() == old) return
        promptUndo.addLast(old)
        while (promptUndo.size > 30) promptUndo.removeFirst()
        canUndoPrompt = true
    }

    fun undoPrompt() {
        val old = promptUndo.removeLastOrNull() ?: return
        prompt = old
        reloadChipsFromPrompt()
        canUndoPrompt = promptUndo.isNotEmpty()
    }

    // ---- プロンプトチップ（SSXL の prompt_chips.js / prompt_chip_ui.js と同じ仕様。Danbooru の補完つき）----
    /** "chip"（チップ編集）/ "text"（テキスト編集）。どちらも最終的に prompt（文字列）が生成に使われる。 */
    var promptMode by mutableStateOf("chip")
        private set
    var positiveChips by mutableStateOf<List<PromptChip>>(PromptChips.parsePrompt(prompt).chips)
        private set
    var selectedChipId by mutableStateOf<String?>(null)
        private set
    var positiveChipInput by mutableStateOf("")
        private set
    var tagSuggestions by mutableStateOf<List<DanbooruClient.Tag>>(emptyList())
        private set
    /** 補完を引いた結果が 0 件だった（そのまま追加の案内を出す）。 */
    var tagNoHit by mutableStateOf(false)
        private set
    var chipToast by mutableStateOf<String?>(null)
        private set
    private val danbooru = DanbooruClient()
    private var tagSearchJob: Job? = null
    private var toastJob: Job? = null

    private fun reloadChipsFromPrompt() {
        positiveChips = PromptChips.parsePrompt(prompt).chips
        selectedChipId = null
    }

    /** チップを確定する。「戻る」に積み、prompt（文字列）も作り直す。 */
    private fun commitChips(chips: List<PromptChip>) {
        pushPromptUndo(prompt)
        positiveChips = chips
        prompt = PromptChips.serializeChips(chips)
    }

    private fun showChipToast(message: String) {
        chipToast = message
        toastJob?.cancel()
        toastJob = viewModelScope.launch { delay(2800); chipToast = null }
    }

    fun changePromptMode(mode: String) {
        if (mode == promptMode) return
        if (mode == "chip") reloadChipsFromPrompt()   // テキストで編集した内容を取り込む
        hideSuggestions()
        promptMode = mode
    }

    fun hideSuggestions() { tagSearchJob?.cancel(); tagSuggestions = emptyList(); tagNoHit = false }

    /** タグ入力欄の変更。カンマが入ったら確定、そうでなければ 180ms 後に Danbooru の補完を引く。 */
    fun onChipInputChange(text: String) {
        positiveChipInput = text
        tagSearchJob?.cancel()
        tagNoHit = false
        val query = text.trim()
        if (text.contains(',')) { commitTyped(text); return }
        if (query.isEmpty()) { tagSuggestions = emptyList(); return }
        tagSearchJob = viewModelScope.launch {
            delay(180)
            try {
                val tags = danbooru.autocomplete(query)
                if (positiveChipInput.trim() == query) {
                    tagSuggestions = tags
                    tagNoHit = tags.isEmpty()
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                android.util.Log.w("dsxl-danbooru", "autocomplete failed: $e", e)
                tagSuggestions = emptyList()    // 通信できなくても入力は続けられる
                showChipToast("タグ補完に失敗: ${e.javaClass.simpleName} ${e.message.orEmpty()}")
            }
        }
    }

    fun addDanbooruTag(name: String) {
        commitChips(positiveChips + PromptChips.chipFromDanbooru(name))
        positiveChipInput = ""
        hideSuggestions()
    }

    /** 入力欄の文字をチップにする（カンマ区切り・`(本文:1.2)` の重みつきの貼り付けも解析する）。 */
    fun commitTyped(text: String) {
        val value = text.trim()
        if (value.isEmpty()) return
        val parsed = PromptChips.parsePrompt(value)
        positiveChipInput = ""
        hideSuggestions()
        if (parsed.chips.isEmpty()) return
        commitChips(positiveChips + parsed.chips)
        if (parsed.added > 1 || parsed.rawCount > 0 || parsed.weightCount > 0) showChipToast(PromptChips.pasteSummary(parsed.added, parsed.weightCount, parsed.rawCount))
    }

    fun selectChip(id: String) { selectedChipId = if (selectedChipId == id) null else id }

    fun deleteChip(id: String) {
        commitChips(positiveChips.filter { it.id != id })
        if (selectedChipId == id) selectedChipId = null
    }

    fun applyChipWeight(id: String, weight: Double) {
        val chip = positiveChips.firstOrNull { it.id == id } ?: return
        if (chip.raw || chip.weight == weight) return
        commitChips(positiveChips.map { if (it.id == id) it.copy(weight = weight) else it })
    }

    fun renameChip(id: String, label: String) {
        val chip = positiveChips.firstOrNull { it.id == id } ?: return
        val next = PromptChips.updateChipLabel(chip, label)
        if (next.label != chip.label) commitChips(positiveChips.map { if (it.id == id) next else it })
    }

    /** ドラッグで並べ替えた結果（チップの id の並び）を確定する。「戻る」に積む。 */
    fun reorderChips(ids: List<String>) {
        val byId = positiveChips.associateBy { it.id }
        val list = ids.mapNotNull { byId[it] }
        if (list.size != positiveChips.size || list.map { it.id } == positiveChips.map { it.id }) return
        commitChips(list)
    }

    // ---- 状態 ----
    var running by mutableStateOf(false)
        private set
    var progress by mutableStateOf<Progress?>(null)
        private set
    var error by mutableStateOf<String?>(null)
        private set
    var history by mutableStateOf(store.load())
        private set
    var nativeStatus by mutableStateOf(native.statusText())
        private set
    private var modelsTick by mutableStateOf(0)   // モデルの配置を調べ直す合図

    /** Hires が使えるか（UNet の共有 bin に、縦横のタイル用グラフも入っている） */
    val hiresAvailable: Boolean get() { modelsTick; return native.hasUnet() }

    /** 表示中のジョブ（履歴の選択 / 直近の生成）。SSXL の state.activeId。 */
    var activeId by mutableStateOf<String?>(null)
        private set
    var previewBitmap by mutableStateOf<Bitmap?>(null)
        private set
    /** 実行中のジョブ ID（生成開始時に払い出す。ステータス行の「Job #…」に出す）。 */
    private var runningId: String? by mutableStateOf(null)
    /** 完了直後のステータス（"完了 · Job #… · 12.3s"）。 */
    private var doneNote by mutableStateOf<String?>(null)
    /** 直近の Base のみの生成結果（Hires 化できるのはこれだけ。latent をエンジンが保持しているため）。 */
    private var lastBase: Pair<String, GenerationResult>? = null

    private var job: Job? = null

    /** 画像は出ているが、履歴への保存がまだ終わっていないジョブ（保存は裏で行う。終わると履歴の一覧に入る）。 */
    private var pendingItem by mutableStateOf<HistoryItem?>(null)

    val activeItem: HistoryItem? get() = history.firstOrNull { it.id == activeId } ?: pendingItem?.takeIf { it.id == activeId }

    // ---- 裏の VAE と、モデルの常駐・先読み ----
    /** 裏で VAE を実行中のジョブ数。CLIP・UNet が終わったジョブの VAE は、次のジョブの CLIP・UNet と重ねて進む。 */
    var decodingJobs by mutableStateOf(0)
        private set
    private var decodeProgress by mutableStateOf<Progress?>(null)
    private var decodeId: String? by mutableStateOf(null)
    /** ロード・解放ボタンの実行中。 */
    var modelBusy by mutableStateOf(false)
        private set
    private var loadProgress by mutableStateOf<Pair<Int, Int>?>(null)
    /** 常駐モデルの状態（pipeline.h の ModelBit）。1 秒おきに更新する。 */
    private var modelMask by mutableStateOf(0)

    /** ジョブの完了後に、次のジョブの足りないモデルを裏で読み込んでいる最中か（SSXL のプリフェッチ）。 */
    private val prefetching: Boolean get() = !modelBusy && !running && (modelMask and BIT_LOADING) != 0

    /** 生成ボタンの上の「モデル」行に出す状態。 */
    val modelStateText: String
        get() {
            if (modelBusy) return loadProgress?.let { (d, t) -> "モデル: ロード中 $d/$t" } ?: "モデル: 処理中…"
            val m = modelMask
            val missing = buildList {
                if ((m and (BIT_CLIP_L or BIT_CLIP_G)) != (BIT_CLIP_L or BIT_CLIP_G)) add("CLIP")
                if ((m and BIT_BASE_UNET) == 0) add("UNet")
                if (hiresAvailable && (m and BIT_TILE) == 0) add("Hires（タイル）")
                if ((m and BIT_VAE) == 0) add("VAE")
            }
            return when {
                (m and BIT_LOADING) != 0 -> "モデル: 読み込み中…"
                missing.isEmpty() -> "モデル: ロード済み" + if ((m and BIT_GPU_VAE) != 0) "（GPU-VAE あり）" else ""
                missing.size >= 3 -> "モデル: 未ロード"
                else -> "モデル: 未ロード（${missing.joinToString("・")}）"
            }
        }

    /** プレビューの進捗バー用: CLIP・UNet の間はその進捗、VAE の間は VAE の進捗。 */
    val activeProgress: Progress? get() = if (running) progress else decodeProgress

    /** 生成・VAE・ロード・解放のどれかが動いている間は、次の生成もロードも受け付けない（最終画像が上がってから次へ）。 */
    val busy: Boolean get() = running || decodingJobs > 0 || modelBusy
    /** ロードできる状態か。 */
    val modelIdle: Boolean get() = !busy
    /** 1 つでもモデルが常駐しているか（解放ボタンを押せるか）。 */
    val anyModelLoaded: Boolean get() = (modelMask and (BIT_CLIP_L or BIT_CLIP_G or BIT_VAE or BIT_BASE_UNET or BIT_TILE or BIT_GPU_VAE)) != 0

    /** SSXL の statusline（"Progress: …"）。 */
    val statusline: String
        get() {
            val p = progress
            val id = runningId?.let { jobIdOf(it) }
            val dp = decodeProgress
            return when {
                running && p != null -> "Progress: ${stageLabel(p)} · Job #$id"
                running -> "Progress: 準備中 · Job #$id"
                decodingJobs > 0 -> "Progress: ${dp?.let { stageLabel(it) } ?: "VAE"} · Job #${decodeId?.let { jobIdOf(it) }}"
                modelBusy -> "Progress: ${loadProgress?.let { (d, t) -> "モデルをロード中 $d/$t" } ?: "モデルを処理中"}"
                doneNote != null -> "Progress: $doneNote${if (prefetching) " · 次ジョブ準備中" else ""}"
                prefetching -> "Progress: 次ジョブ準備中"
                else -> "Progress: —"
            }
        }

    private fun stageLabel(p: Progress): String {
        val n = if (p.totalSteps > 1) " ${p.step}/${p.totalSteps}" else ""
        return when (p.stage) {
            Stage.TEXT -> "CLIP"
            Stage.BASE_UNET -> "Base sampling$n"
            Stage.UPSCALE -> "Upscale"
            Stage.HIRES_UNET -> "Hires sampling$n"
            Stage.VAE -> "VAE$n"
            Stage.LOAD -> "モデルをロード中$n"
        }
    }

    private fun jobIdOf(id: String) = "%010x".format((id.toLongOrNull() ?: 0L) and 0xFFFFFFFFFFL)

    // ---- アプリのロック（パスワード / 指紋）----
    val lock = AppLock(app)
    var hasPassword by mutableStateOf(lock.hasPassword)
        private set
    var biometricOn by mutableStateOf(lock.biometricEnabled)
        private set
    /** ロック中か。起動時に、ロックが有効なら最初からロック状態にする。 */
    var locked by mutableStateOf(lock.enabled)
        private set
    var showSettings by mutableStateOf(false)

    fun setPassword(pw: String) { lock.setPassword(pw); hasPassword = true }
    fun clearPassword() { lock.clearPassword(); hasPassword = false }
    fun setBiometric(on: Boolean) { lock.biometricEnabled = on; biometricOn = on }
    fun unlock() { locked = false }
    /** アプリが背面に回ったとき（MainActivity.onStop）など。ロックが有効ならロックする。 */
    fun lockNow() { if (lock.enabled) locked = true }

    // ---- メトリクス（生成中と、終わってから 2 秒は 0.25 秒、それ以外は 1 秒おき。グラフは直近 MetricsWindowMs（DeviceMetrics.kt）の時間軸）----
    private val root = RootMetrics(app)
    private val sampler = DeviceMetrics(app, { native.busy() }, root)
    var metrics by mutableStateOf<Metrics?>(null)
        private set
    var metricsHistory by mutableStateOf<List<Metrics>>(emptyList())
        private set
    /** 詳細モニターの展開状態。ヘッダの CPU / MEM / NPU / GPU の行をタップで開閉（SSXL の sys-metrics に相当）。 */
    /** 履歴の開閉（設定として覚える）。 */
    var historyExpanded by mutableStateOf(prefs.getBoolean("historyExpanded", true))
        private set

    fun toggleHistoryExpanded() {
        historyExpanded = !historyExpanded
        prefs.edit().putBoolean("historyExpanded", historyExpanded).apply()
    }

    var monitorExpanded by mutableStateOf(false)

    init {
        viewModelScope.launch {
            var lastActive = 0L   // 最後に「生成中」だった時刻。終わってからも 2 秒は高頻度を続ける（終わり際の変化を細かく見る）
            while (true) {
                root.start()
                val m = sampler.sample()
                metrics = m
                // 履歴は時刻で切る（直近 MetricsWindowMs。線がつながるよう、窓の外の 1 点は残す）。ポーリング間隔が変わっても、グラフの横軸は時間に比例する
                val all = metricsHistory + m
                val firstIn = all.indexOfFirst { m.timeMs - it.timeMs <= MetricsWindowMs }.coerceAtLeast(0)
                metricsHistory = all.drop((firstIn - 1).coerceAtLeast(0))
                modelMask = native.modelState(orientation)
                val now = android.os.SystemClock.elapsedRealtime()
                if (running || decodingJobs > 0) lastActive = now   // 生成中 = CLIP・UNet と、裏の VAE の間
                delay(if (now - lastActive < 2000L) 250L else 1000L)
            }
        }
    }

    // ---- 履歴（検索・除外・ページ・削除モード）----
    /** ヘッダの「除外」。オンのとき、除外ワードに当たるジョブを履歴から隠す（SSXL の blurBlacklist）。 */
    var blurBlacklist by mutableStateOf(prefs.getBoolean("blurBlacklist", true))
        private set
    /** 除外ワード（空白・カンマ区切り。ジョブ ID も指定できる）。設定で編集する。未保存なら既定の語句。 */
    var imageBlacklistTags by mutableStateOf(prefs.getString("imageBlacklistTags", null) ?: DEFAULT_IMAGE_BLACKLIST)
        private set
    var historySearch by mutableStateOf("")
        private set
    var historyPage by mutableStateOf(1)
        private set
    var historyDeleteMode by mutableStateOf(false)
        private set
    var historySelectedIds by mutableStateOf<Set<String>>(emptySet())
        private set
    /** 🎲 ランダム表示中の一覧（null なら通常のページ表示）。 */
    private var historyRandomJobs by mutableStateOf<List<HistoryItem>?>(null)

    /** 履歴 1 ページの列数（横）。未保存なら 4。以前の件数だけの保存は、列 4 で割り切れるとき行数へ移す。 */
    var historyCols by mutableStateOf(initialHistoryCols())
        private set

    /** 履歴 1 ページの行数（縦）。 */
    var historyRows by mutableStateOf(initialHistoryRows())
        private set

    /** 1 ページの表示枚数（横 × 縦）。 */
    val historyPageSize: Int get() = historyCols * historyRows

    /** GPU 使用率の上限（%）。既定 90。VAE で GPU を NPU と分担するとき、OS の描画まで重くならないよう余白を残す。 */
    var gpuCap by mutableStateOf(prefs.getInt("gpuCap", 90).coerceIn(10, 100))
        private set

    /** メモリ節約モード。UNet を今の向きの 2 グラフだけ読み、端末のメモリ（DMA-BUF）を約 1.4GB 減らす。向きを変えるたびに UNet を読み直す（約 2 秒）。既定はオフ。 */
    var memorySaver by mutableStateOf(prefs.getBoolean("memorySaver", false))
        private set

    init { native.setGpuCap(gpuCap); native.setMemorySaver(memorySaver) }

    /** 生成中のプレビュー（TAESD。UNet の各ステップの予測を、GPU で絵にして途中経過として出す）。設定で切り替え、覚える。 */
    var previewTaesd by mutableStateOf(prefs.getBoolean("previewTaesd", true))
        private set

    fun updatePreviewTaesd(on: Boolean) {
        previewTaesd = on
        prefs.edit().putBoolean("previewTaesd", on).apply()
    }

    /**
     * 高速デコード: 最終の latent を VAE ではなく TAESD（GPU）でデコードする。数秒速くなるが、細部はやや甘い。
     * TAESD のモデル（gpu/taesd_*_fp32.dlc、またはコンパイル済みの *_gpu.bin）が無いときは、設定で選べない。選んでいても実行時は VAE に戻る。
     * 設定で切り替え、覚える（既定オフ）。履歴には draft の印が付く。
     */
    var draftMode by mutableStateOf(prefs.getBoolean("draftMode", false))
        private set

    fun updateDraftMode(on: Boolean) {
        draftMode = on
        prefs.edit().putBoolean("draftMode", on).apply()
    }

    /** 起動時に見た、欠けている TAESD のファイル名（全部あれば null）。プレビューと高速デコードが使えるかの判定の元。 */
    val taesdMissingFile: String? = missingTaesdFile()

    /** 保存してあるオンのまま、TAESD が無くて使えないとき（生成は VAE で動く）。設定の表示と生成画面の案内に使う。 */
    val previewBlocked: Boolean get() = taesdMissingFile != null && (previewTaesd || draftMode)

    /** 高速デコードが実際に有効か（オンで、TAESD が揃っている）。GPU 使用率の上限の欄を出すかの判定に使う。 */
    val draftActive: Boolean get() = draftMode && taesdMissingFile == null

    // ---- LoRA。UNet は LoRA の有無によらず同じ共有 bin（unet_lora_4g.bin）。LoRA ごとの super adapter は models/loras/<名前>.4g.bin（縦横・Hires 共通）----
    /** models/loras/ にある LoRA の名前（<名前>.4g.bin）。LoRA の欄を開くたびに読み直す。 */
    var loraNames by mutableStateOf(listLoras())
        private set
    /** 使う LoRA（空なら使わない）。起動のたびに「なし」から始める（前回の選択は覚えない）。 */
    var loraName by mutableStateOf("")
        private set
    /** UNet（共有 bin）が置かれているか。LoRA もこれで動く。 */
    var loraReady by mutableStateOf(checkLoraReady())
        private set
    /** 強さ w（A1111 の <lora:名前:w> の w と同じ意味。既定 1.0）。実行時の alpha 入力は w × c（c は loras/<名前>.json の "c"。無ければ 1）。 */
    var loraStrength by mutableStateOf(prefs.getFloat("loraStrength", 1.0f))
        private set

    private fun lorasDir() = File(native.modelDir, "loras")
    private fun listLoras(): List<String> =
        lorasDir().listFiles { f -> f.isFile && f.name.endsWith(LORA_SUFFIX) }?.map { it.name.removeSuffix(LORA_SUFFIX) }?.sorted() ?: emptyList()

    private fun checkLoraReady(): Boolean = native.hasUnet()

    fun refreshLoras() {
        loraNames = listLoras()
        loraReady = checkLoraReady()
        if (loraName.isNotEmpty() && loraName !in loraNames) updateLoraName("")   // 選んでいた LoRA のファイルが消えた
    }

    fun updateLoraName(name: String) {
        loraName = name
    }

    fun updateLoraStrength(w: Float) {
        loraStrength = w
        prefs.edit().putFloat("loraStrength", w).apply()
    }

    // ---- LoRA の追加（Civitai の URL から取って、端末の中で super adapter に変換する。変換は lora/LoraConverter.kt）----
    // ひな型は models/unet_lora_4g_default.bin（LoRA なしでも使う default の super adapter）。窓の対応表は APK の assets の super_4g_windows.json（loraconverter/data と同じもの）。
    /** Civitai の API トークン（設定で登録する。アプリ専用の領域に保存する。公開されているモデルなら空でもよい）。 */
    var civitaiToken by mutableStateOf(prefs.getString("civitaiToken", "") ?: "")
    fun updateCivitaiToken(token: String) {
        civitaiToken = token.trim()
        prefs.edit().putString("civitaiToken", civitaiToken).apply()
    }
    var loraImporting by mutableStateOf(false)
    var loraImportMessage by mutableStateOf("")
    var loraImportProgress by mutableStateOf(0f)
    var loraImportFailed by mutableStateOf(false)
    /** ひな型が置かれているか。無ければ、足りないファイル名を返す。 */
    fun missingLoraTemplate(): String? =
        com.dragonspeedxl.app.lora.LoraConverter.DEFAULT_ADAPTER.takeIf { !File(native.modelDir, it).isFile }

    fun importLora(url: String) = startLoraImport { converter, work, progress ->
        com.dragonspeedxl.app.lora.LoraImporter.fromCivitai(url, civitaiToken, converter, native.modelDir, lorasDir(), work, progress)
    }

    /** 端末から選んだ safetensors を取り込む。名前と activation text は入力欄の値（空の activation text も可）。 */
    fun importLoraFile(uri: Uri, name: String, activation: String) = startLoraImport { converter, work, progress ->
        com.dragonspeedxl.app.lora.LoraConverter.checkName(name)
        val src = File(work, "$name.safetensors")
        try {
            progress("コピー中", 0f)
            val input = getApplication<Application>().contentResolver.openInputStream(uri) ?: error("選んだファイルを開けない")
            input.use { i -> src.outputStream().use { i.copyTo(it) } }
            com.dragonspeedxl.app.lora.LoraImporter.fromFile(src, name, activation, converter, native.modelDir, lorasDir(), progress)
        } finally {
            src.delete()
        }
    }

    /** 選んだファイルの表示名（拡張子つき）。取れなければ空。 */
    fun loraDisplayName(uri: Uri): String = try {
        getApplication<Application>().contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { c -> if (c.moveToFirst()) c.getString(0) else null } ?: ""
    } catch (_: Exception) { "" }

    /** LoRA 取り込みの共通の流れ。作業場所（cache）で変換し、一覧を更新する。 */
    private fun startLoraImport(run: (com.dragonspeedxl.app.lora.LoraConverter, File, (String, Float) -> Unit) -> com.dragonspeedxl.app.lora.LoraImporter.Outcome) {
        if (loraImporting || busy) return
        missingLoraTemplate()?.let { loraImportFailed = true; loraImportMessage = "ひな型 $it が models/ にありません。"; return }
        loraImporting = true
        loraImportFailed = false
        loraImportProgress = 0f
        loraImportMessage = "開始"
        viewModelScope.launch {
            try {
                val outcome = withContext(Dispatchers.IO) {
                    val assets = getApplication<Application>().assets
                    val converter = com.dragonspeedxl.app.lora.LoraConverter(
                        assets.open("layout.json").bufferedReader().use { it.readText() },
                        assets.open("attach_map.json").bufferedReader().use { it.readText() },
                        assets.open("super_4g_windows.json").bufferedReader().use { it.readText() },
                        assets.open("clip_layout.json").bufferedReader().use { it.readText() },
                    )
                    run(converter, File(getApplication<Application>().cacheDir, "lora_download")) { msg, f -> loraImportMessage = msg; loraImportProgress = f }
                }
                refreshLoras()
                loraImportProgress = 1f
                loraImportMessage = "「${outcome.name}」を追加しました（はみ出し ${"%.4f".format(outcome.result.overRatio * 100)}%）。" +
                    if (outcome.activation.isEmpty()) "" else "\nactivation text: ${outcome.activation}"
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                loraImportFailed = true
                loraImportMessage = e.message ?: e.toString()
            } finally {
                loraImporting = false
            }
        }
    }

    /** この LoRA の係数 c（none モードで B を c で割って焼いてあるので、実行時に掛け戻す）。loras/<名前>.json の {"c": 1.518}。無ければ 1。 */
    private fun loraCoefficient(name: String): Float = try {
        val t = File(lorasDir(), "$name.json").readText()
        Regex("\"c\"\\s*:\\s*([0-9.eE+-]+)").find(t)?.groupValues?.get(1)?.toFloat() ?: 1f
    } catch (_: Exception) { 1f }

    /** この LoRA の activation text（トリガーの文。loras/<名前>.json の "activation"）。無ければ空。LoRA 欄の [コピー] でクリップボードに入れる。 */
    fun loraActivation(name: String): String = try {
        org.json.JSONObject(File(lorasDir(), "$name.json").readText()).optString("activation", "")
    } catch (_: Exception) { "" }

    /** 今の設定で LoRA を使えるか: 選んでいて、UNet と adapter（<名前>.4g.bin）がある。縦横どちらでも使える。 */
    private fun activeLora(): String =
        if (loraName.isNotEmpty() && checkLoraReady() && File(lorasDir(), loraName + LORA_SUFFIX).exists()) loraName else ""

    /** 生成中の途中経過の絵（TAESD）。最終画像が出る・失敗する・キャンセルすると消える。 */
    var livePreview by mutableStateOf<Bitmap?>(null)
        private set
    private var previewEpoch = 0   // ジョブごとに増やす。遅れて届いた前のジョブのプレビューを捨てるため

    /** プレビュー欄に出す絵: 生成中は途中経過、無ければ表示中のジョブの画像。 */
    val displayBitmap: Bitmap? get() = livePreview ?: previewBitmap

    fun updateDefaultPrompt(text: String) {
        defaultPrompt = text
        prefs.edit().putString("defaultPrompt", text).apply()
    }

    fun updateMemorySaver(on: Boolean) {
        memorySaver = on
        prefs.edit().putBoolean("memorySaver", on).apply()
        native.setMemorySaver(on)
    }

    fun updateGpuCap(percent: Int) {
        gpuCap = percent.coerceIn(10, 100)
        prefs.edit().putInt("gpuCap", gpuCap).apply()
        native.setGpuCap(gpuCap)
    }

    private fun initialHistoryCols(): Int =
        if (prefs.contains("historyCols")) prefs.getInt("historyCols", HISTORY_COLS_DEFAULT).coerceIn(HISTORY_GRID_MIN, HISTORY_GRID_MAX)
        else HISTORY_COLS_DEFAULT

    private fun initialHistoryRows(): Int {
        if (prefs.contains("historyRows")) {
            return prefs.getInt("historyRows", HISTORY_ROWS_DEFAULT).coerceIn(HISTORY_GRID_MIN, HISTORY_GRID_MAX)
        }
        // 以前は件数だけを保存していた（既定 16 = 4 列 × 4 行）。列は 4 のまま、割り切れるときだけ行に移す
        val page = prefs.getInt("historyPageSize", HISTORY_COLS_DEFAULT * HISTORY_ROWS_DEFAULT)
        val rows = page / HISTORY_COLS_DEFAULT
        return if (rows * HISTORY_COLS_DEFAULT == page) rows.coerceIn(HISTORY_GRID_MIN, HISTORY_GRID_MAX) else HISTORY_ROWS_DEFAULT
    }

    /** 履歴の列数（横）を変える。ページの区切りが変わるので、1 ページ目に戻す。 */
    fun updateHistoryCols(n: Int) {
        val v = n.coerceIn(HISTORY_GRID_MIN, HISTORY_GRID_MAX)
        if (v == historyCols) return
        historyCols = v
        prefs.edit().putInt("historyCols", v).apply()
        historyPage = 1
        historyRandomJobs = null
    }

    /** 履歴の行数（縦）を変える。ページの区切りが変わるので、1 ページ目に戻す。 */
    fun updateHistoryRows(n: Int) {
        val v = n.coerceIn(HISTORY_GRID_MIN, HISTORY_GRID_MAX)
        if (v == historyRows) return
        historyRows = v
        prefs.edit().putInt("historyRows", v).apply()
        historyPage = 1
        historyRandomJobs = null
    }

    /**
     * 生成中のプレビューと高速デコードに必要な TAESD のうち、無いファイル名。
     * gpu/taesd_<寸法>_gpu.bin か _fp32.dlc のどちらかがあれば、その寸法は使える。全部あれば null。
     */
    fun missingTaesdFile(): String? {
        val gpu = File(native.modelDir, "gpu")
        for (size in listOf("160x90", "240x135", "90x160", "135x240")) {
            val dlc = "taesd_${size}_fp32.dlc"
            if (!File(gpu, "taesd_${size}_gpu.bin").exists() && !File(gpu, dlc).exists()) return dlc
        }
        return null
    }


    fun updateBlurBlacklist(on: Boolean) { blurBlacklist = on; prefs.edit().putBoolean("blurBlacklist", on).apply(); historyPage = 1 }
    fun updateImageBlacklistTags(v: String) { imageBlacklistTags = v; prefs.edit().putString("imageBlacklistTags", v).apply() }
    fun updateHistorySearch(v: String) { historySearch = v; historyPage = 1; historyRandomJobs = null }

    private fun searchText(it: HistoryItem) = "${it.prompt} ${it.jobId} ${it.seed} ${it.kind}".lowercase()

    private fun isBlacklisted(it: HistoryItem): Boolean {
        val words = imageBlacklistTags.split(Regex("[,\\s]+")).map { w -> w.trim().lowercase() }.filter { w -> w.isNotEmpty() }
        if (words.isEmpty()) return false
        val text = searchText(it)
        return words.any { w -> text.contains(w) }
    }

    /** 検索語（"-語句" は除外）と「除外」を適用した履歴。 */
    val filteredHistory: List<HistoryItem>
        get() {
            val terms = historySearch.trim().lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }
            val include = terms.filter { !it.startsWith("-") }
            val exclude = terms.filter { it.startsWith("-") && it.length > 1 }.map { it.drop(1) }
            return history.filter { h ->
                val t = searchText(h)
                !(blurBlacklist && isBlacklisted(h)) && include.all { t.contains(it) } && exclude.none { t.contains(it) }
            }
        }

    val historyPageCount: Int get() = max(1, (filteredHistory.size + historyPageSize - 1) / historyPageSize)

    /** いま表示する履歴（ランダム表示中はその一覧、通常はページ分）。 */
    val historyPageItems: List<HistoryItem>
        get() = historyRandomJobs ?: filteredHistory.drop((historyPage - 1) * historyPageSize).take(historyPageSize)

    fun historyPrev() { historyRandomJobs = null; historyPage = (historyPage - 1).coerceAtLeast(1) }
    fun historyNext() { historyRandomJobs = null; historyPage = (historyPage + 1).coerceAtMost(historyPageCount) }
    fun historyJump(page: Int) { historyRandomJobs = null; historyPage = page.coerceIn(1, historyPageCount) }
    fun historyRandom() { historyRandomJobs = filteredHistory.shuffled().take(historyPageSize) }

    fun toggleHistoryDeleteMode() { historyDeleteMode = !historyDeleteMode; historySelectedIds = emptySet() }
    fun toggleHistorySelected(id: String) {
        historySelectedIds = if (id in historySelectedIds) historySelectedIds - id else historySelectedIds + id
    }

    fun deleteSelected() {
        val items = history.filter { it.id in historySelectedIds }
        if (items.isEmpty()) return
        store.deleteAll(items)
        if (activeId in historySelectedIds) { activeId = null; previewBitmap = null }
        if (lastBase?.first in historySelectedIds) lastBase = null
        historySelectedIds = emptySet()
        historyDeleteMode = false
        historyRandomJobs = null
        history = store.load()
        historyPage = historyPage.coerceIn(1, historyPageCount)
    }

    // ---- 操作 ----
    fun refreshNativeStatus() {
        nativeStatus = native.statusText()
        modelsTick++
    }

    /** 生成。hires steps が 0（または Hires 用モデルなし）なら Base のみ。 */
    fun generate() {
        if (busy) return
        pushPromptUndo(prompt.trim())   // 「戻る」で、生成に使ったプロンプトへ戻れるようにする
        val s = seed.trim().toLongOrNull() ?: Random.nextLong(0, 4_294_967_295L)
        val lora = activeLora()   // 空なら LoRA なし（エンジンが default の adapter を使う）
        val hires = hiresStepsValue > 0 && hiresAvailable
        val req = GenerationRequest(prompt.trim(), orientation, s, hires, baseStepsValue, hiresStepsValue.coerceAtLeast(1), hiresDenoiseValue, draftMode,
            lora, if (lora.isEmpty()) 0f else loraStrength * loraCoefficient(lora), if (lora.isEmpty()) 0f else loraStrength)
        launchJob { engine.begin(req, ::onProgress) }
    }

    /** ✨ Hires（SSXL の hiresActionBtn）。直近の Base のみの結果にだけ掛けられる。 */
    fun refine() {
        val (baseId, base) = lastBase ?: return
        if (busy || activeId != baseId) return
        launchJob(refining = true) { engine.beginRefine(base, hiresStepsValue.coerceAtLeast(1), hiresDenoiseValue, draftMode, ::onProgress) }
    }

    /** このジョブを ✨ Hires 化できるか。 */
    fun canRefine(item: HistoryItem): Boolean = !busy && hiresAvailable && !item.hires && lastBase?.first == item.id

    /** 実行中のジョブ（CLIP・UNet）を中断する。裏で動いている前のジョブの VAE は止めない。 */
    fun cancel() {
        if (!running) return
        engine.cancel()
        job?.cancel()
    }

    private val engine: GenerationEngine get() = native

    /**
     * ジョブを起動する。流れ: start（CLIP・UNet。ここまでは 1 度に 1 ジョブ）→ 次のジョブを受け付ける → await（VAE。裏で進み、次のジョブと重なる）。
     * 画像は VAE が終わってから履歴とプレビューに入る。
     */
    private fun launchJob(refining: Boolean = false, start: suspend () -> PendingGeneration) {
        running = true
        error = null
        progress = null
        doneNote = null
        val id = store.newId()
        runningId = id
        if (!refining) lastBase = null   // 新しい Base を作り始めると、エンジンが保持していた Base の latent は置き換わる
        // 生成中のプレビュー（TAESD）。届くのはネイティブのスレッドなので、メインスレッドへ渡し直す。古いジョブのものは epoch で捨てる
        val epoch = ++previewEpoch
        livePreview = null
        native.previewSink = if (previewTaesd) { bmp ->
            viewModelScope.launch { if (epoch == previewEpoch && (running || decodingJobs > 0)) livePreview = bmp }
        } else null
        job = viewModelScope.launch {
            var foregroundDone = false
            try {
                engine.prepare()?.let { throw IllegalStateException(it) }
                val pending = start()
                // CLIP・UNet は終わった。NPU は空いたので次の生成を受け付け、VAE は裏で続ける
                foregroundDone = true
                running = false
                runningId = null
                progress = null
                decodingJobs++
                decodeId = id
                decodeProgress = null
                val r = try {
                    engine.await(pending, ::onDecodeProgress)
                } finally {
                    decodingJobs--
                    if (decodingJobs == 0) { decodeProgress = null; decodeId = null }
                }
                // 画像を先に出す。履歴への保存（PNG 圧縮。1080p で重い）は裏で行い、終わったら履歴の一覧に加える
                val tShow = System.nanoTime()
                val item = store.itemFor(r, id)
                pendingItem = item
                activeId = item.id
                previewBitmap = r.bitmap
                livePreview = null   // 最終画像が出た
                // Hires を掛けていない Base は、まだエンジンが latent を保持していれば ✨ Hires 化できる
                lastBase = if (!r.hiresDone && native.canRefine(r.baseHandle)) item.id to r else null
                historyRandomJobs = null
                doneNote = "完了 · Job #${item.jobId} · ${"%.1f".format(r.totalMillis / 1000.0)}s"
                android.util.Log.i("dsxl-time", "preview shown: ${(System.nanoTime() - tShow) / 1_000_000} ms after the image arrived on the main thread")
                saveHistoryInBackground(r, item)
                // 次のジョブがすぐ始まるように、足りないモデルを裏で読み込んでおく（SSXL のプリフェッチ）。ほかに動いているジョブがあるときは、そちらに任せる
                if (!running && decodingJobs == 0) native.prefetch(r.request.orientation)
            } catch (e: CancellationException) {
                error = "キャンセルしました"
                livePreview = null
                throw e
            } catch (e: Throwable) {
                error = "生成に失敗: ${e.message ?: e.javaClass.simpleName}"
                livePreview = null
            } finally {
                if (!foregroundDone) {
                    running = false
                    runningId = null
                    progress = null
                }
                refreshNativeStatus()
            }
        }
    }

    /** 履歴への保存（PNG 圧縮とファイル書き込み）を裏のスレッドで行い、終わったら履歴の一覧を更新する。保存が終わるまでは、表示中のジョブは pendingItem で出す。 */
    private fun saveHistoryInBackground(r: GenerationResult, item: HistoryItem) {
        viewModelScope.launch {
            val t0 = System.nanoTime()
            val items = withContext(Dispatchers.IO) {
                store.save(r, item)
                store.load()
            }
            android.util.Log.i("dsxl-time", "history save (PNG + index): ${(System.nanoTime() - t0) / 1_000_000} ms (background)")
            history = items
            historyRandomJobs = null
            if (pendingItem?.id == item.id) pendingItem = null
        }
    }

    private fun onProgress(p: Progress) {
        progress = p
    }

    private fun onDecodeProgress(p: Progress) {
        decodeProgress = p
    }

    // ---- モデルのロード・解放（生成ボタンの上の行）----
    /** 今の向きのモデル（CLIP・VAE・UNet・Hires UNet・GPU 用 VAE）をすべて読み込んで常駐させる。 */
    fun loadModels() {
        if (!modelIdle) return
        val o = orientation
        modelBusy = true
        error = null
        doneNote = null
        loadProgress = null
        viewModelScope.launch {
            try {
                engine.prepare()?.let { throw IllegalStateException(it) }
                val note = native.loadModels(o) { done, total -> loadProgress = done to total }
                doneNote = "モデルをロードしました" + if (note.isNotEmpty()) "（$note）" else ""
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                error = "ロードに失敗: ${e.message ?: e.javaClass.simpleName}"
            } finally {
                modelBusy = false
                loadProgress = null
                modelMask = native.modelState(orientation)
                refreshNativeStatus()
            }
        }
    }

    /** 常駐しているモデルをすべて解放する（メモリを空ける）。次の生成・ロードで読み込み直す。 */
    fun unloadModels() {
        if (!modelIdle) return
        modelBusy = true
        error = null
        doneNote = null
        viewModelScope.launch {
            try {
                native.unloadModels()
                doneNote = "モデルを解放しました"
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                error = "解放に失敗: ${e.message ?: e.javaClass.simpleName}"
            } finally {
                modelBusy = false
                modelMask = native.modelState(orientation)
                refreshNativeStatus()
            }
        }
    }

    /** 履歴のジョブを表示する（SSXL では履歴のカードをタップ）。 */
    fun selectHistory(item: HistoryItem) {
        if (busy) return
        val bmp = store.loadBitmap(item) ?: return
        activeId = item.id
        previewBitmap = bmp
        doneNote = null
    }

    /**
     * ↙流用（SSXL の useAsBaseBtn）: 表示中のジョブの条件を入力欄に戻す。LoRA も戻す（LoRA なしの履歴なら LoRA も外す）。
     * 履歴の LoRA の super adapter が今は無いときは、LoRA なしにして知らせる。
     */
    fun useAsBase() {
        val a = activeItem ?: return
        if (busy) return
        pushPromptUndo(prompt)
        prompt = a.prompt
        reloadChipsFromPrompt()
        orientation = a.orientation
        seed = a.seed.toString()
        baseSteps = a.baseSteps.toString()
        hiresSteps = if (a.hires) a.hiresSteps.toString() else "0"
        hiresDenoise = a.denoise.toString()
        if (a.lora.isEmpty()) {
            updateLoraName("")
        } else if (File(lorasDir(), a.lora + LORA_SUFFIX).isFile) {
            updateLoraName(a.lora)
            updateLoraStrength(a.loraStrength)
        } else {
            updateLoraName("")
            doneNote = "LoRA「${a.lora}」が無いので、LoRA なしにしました"
        }
    }

    fun thumbnail(item: HistoryItem): Bitmap? = store.loadBitmap(item, maxSide = 360)

    fun saveToGallery() {
        val bmp = previewBitmap ?: return
        val uri = store.saveToGallery(bmp, "dsxl_${System.currentTimeMillis()}", activeItem)
        error = if (uri != null) null else "保存に失敗しました"
        if (uri != null) doneNote = "ギャラリーに保存しました（Pictures/dragonspeedXL）"
    }

    override fun onCleared() {
        root.stop()
        native.release()
    }

    companion object {
        const val DEFAULT_PROMPT = "hatsune miku"

        // ネイティブ（pipeline.h の ModelBit）の常駐モデルの状態のビット
        private const val BIT_CLIP_L = 1 shl 0
        private const val BIT_CLIP_G = 1 shl 1
        private const val BIT_VAE = 1 shl 2
        private const val BIT_BASE_UNET = 1 shl 3
        private const val BIT_GPU_VAE = 1 shl 5
        private const val LORA_SUFFIX = com.dragonspeedxl.app.lora.LoraConverter.SUFFIX
        private const val BIT_LOADING = 1 shl 6
        private const val BIT_TILE = 1 shl 7
    }
}
