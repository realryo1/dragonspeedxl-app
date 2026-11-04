package com.dragonspeedxl.app

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.provider.OpenableColumns
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.dragonspeedxl.app.data.HistoryItem
import com.dragonspeedxl.app.data.HistoryStore
import com.dragonspeedxl.app.download.DownloadService
import com.dragonspeedxl.app.download.DownloadStatus
import com.dragonspeedxl.app.download.ModelDownloader
import com.dragonspeedxl.app.download.ModelManifest
import com.dragonspeedxl.app.download.toHex
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
import com.dragonspeedxl.app.lora.LoraConverter
import com.dragonspeedxl.app.lora.LoraMixer
import com.dragonspeedxl.app.lora.LoraUse
import com.dragonspeedxl.app.prompt.DanbooruClient
import com.dragonspeedxl.app.prompt.TagDictionary
import com.dragonspeedxl.app.prompt.PromptChip
import com.dragonspeedxl.app.prompt.PromptChips
import com.dragonspeedxl.app.security.AppLock
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
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
    private val downloader = ModelDownloader(native.modelDir)
    /** APK に同梱した、モデルの一覧（サイズと sha256）。足りないファイルを調べるのと、取るのに使う。 */
    private val modelManifest = ModelManifest.parse(app.assets.open(ModelManifest.ASSET).bufferedReader().use { it.readText() })
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
    private val danbooru = DanbooruClient(TagDictionary(java.io.File(app.filesDir, "tag_dictionary")))
    /** 端末内の辞書に切り替えたことを知らせ済み（Danbooru に繋がったら戻す）。 */
    private var tagLocalNotified = false
    /** 端末内の辞書の取得を、このプロセスで 1 回試した。 */
    private var tagDictPrefetched = false

    /** ローカル辞書モード: Danbooru に問い合わせず、端末内の辞書（dskjal のタグ一覧）だけで補完する。設定で切り替え、覚える（既定オフ）。 */
    var localTagMode by mutableStateOf(prefs.getBoolean("localTagMode", false))
        private set
    /** 置いてある辞書のファイル名（例 "tags-2026-03-23.csv"。無ければ null）。設定に出す。 */
    var tagDictionaryName by mutableStateOf<String?>(null)
        private set
    /** 辞書を取得中。 */
    var tagDictionaryLoading by mutableStateOf(false)
        private set
    /** 最後に辞書の取得に失敗した理由（設定に出す。成功したら null）。 */
    var tagDictionaryError by mutableStateOf<String?>(null)
        private set

    /** 設定を開いたときに、辞書の有無を見直す（通信しない）。 */
    fun refreshTagDictionary() {
        viewModelScope.launch { tagDictionaryName = withContext(Dispatchers.IO) { danbooru.localFile()?.name } }
    }

    fun updateLocalTagMode(on: Boolean) {
        localTagMode = on
        prefs.edit().putBoolean("localTagMode", on).apply()
        hideSuggestions()
        if (on) fetchTagDictionary()
    }

    /** 辞書が無ければ取る（あれば何もしない）。失敗は設定に出す。 */
    fun fetchTagDictionary() {
        if (tagDictionaryLoading) return
        tagDictPrefetched = true
        viewModelScope.launch {
            tagDictionaryLoading = true
            tagDictionaryError = null
            try {
                tagDictionaryName = danbooru.prefetchLocal().name
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                android.util.Log.w("dsxl-danbooru", "tag dictionary fetch failed: $e")
                tagDictionaryError = "${e.javaClass.simpleName} ${e.message.orEmpty()}".trim()
            } finally {
                tagDictionaryLoading = false
            }
        }
    }
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

    /** タグ入力欄の変更。カンマが入ったら確定、そうでなければ 180ms 後に Danbooru の補完を引く（繋がらなければ端末内の辞書）。 */
    fun onChipInputChange(text: String) {
        positiveChipInput = text
        tagSearchJob?.cancel()
        tagNoHit = false
        val query = text.trim()
        if (text.contains(',')) { commitTyped(text); return }
        if (query.isEmpty()) { tagSuggestions = emptyList(); return }
        if (!tagDictPrefetched) fetchTagDictionary()
        tagSearchJob = viewModelScope.launch {
            delay(180)
            try {
                val r = danbooru.autocomplete(query, localOnly = localTagMode)
                r.onlineError?.let { android.util.Log.w("dsxl-danbooru", "online failed, using local tags: $it") }
                if (!r.local) tagLocalNotified = false
                else if (r.onlineError != null && !tagLocalNotified) {
                    tagLocalNotified = true
                    showChipToast("Danbooru に繋がらないため、端末内の辞書で補完します")
                }
                if (positiveChipInput.trim() == query) {
                    tagSuggestions = r.tags
                    tagNoHit = r.tags.isEmpty()
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

    // ---- モデルのダウンロード（初回は、足りないモデルを取るまで本体の画面を出さない）----
    /** 足りないモデルがあるか（同梱の manifest と端末の中身で決める）。true の間は、案内画面を出す。 */
    var modelsNeedDownload by mutableStateOf(computeNeedDownload())
        private set
    var downloadProgress by mutableStateOf<ModelDownloader.Progress?>(null)
        private set
    /** 取得の完了・失敗のメッセージ。 */
    var downloadMessage by mutableStateOf<String?>(null)
        private set
    var downloadFailed by mutableStateOf(false)
        private set
    private var downloadJob: Job? = null
    val downloading: Boolean get() = downloadJob?.isActive == true

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
            Stage.MIX -> "LoRA を合成中 ${p.step}%"
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

    /** GPU 使用率の上限（%）。既定は制限なし（100）。描画が重くなるときは設定で下げる（VAE で GPU を NPU と分担するときの余白）。 */
    var gpuCap by mutableStateOf(prefs.getInt("gpuCap", 100).coerceIn(10, 100))
        private set

    /** メモリ節約モード。UNet を今の向きの 2 グラフだけ読み、端末のメモリ（DMA-BUF）を約 1.4GB 減らす。向きを変えるたびに UNet を読み直す（約 2 秒）。既定はオン。 */
    var memorySaver by mutableStateOf(prefs.getBoolean("memorySaver", true))
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
     * 設定で切り替え、覚える（既定オン）。履歴には draft の印が付く。
     */
    var draftMode by mutableStateOf(prefs.getBoolean("draftMode", true))
        private set

    fun updateDraftMode(on: Boolean) {
        draftMode = on
        prefs.edit().putBoolean("draftMode", on).apply()
    }

    /**
     * 欠けている TAESD のファイル名（全部あれば null）。プレビューと高速デコードが使えるかの判定の元。
     * 起動時と、モデルの取得が終わったとき（refreshNativeStatus）に数え直す。起動時の値を持ち続けると、初回の取得のあとも「無い」と出る。
     */
    var taesdMissingFile: String? by mutableStateOf(missingTaesdFile())
        private set

    /** 保存してあるオンのまま、TAESD が無くて使えないとき（生成は VAE で動く）。設定の表示と生成画面の案内に使う。 */
    val previewBlocked: Boolean get() = taesdMissingFile != null && (previewTaesd || draftMode)

    // ---- LoRA。UNet は LoRA の有無によらず同じ共有 bin（unet_lora_4g.bin）。LoRA ごとの super adapter は models/loras/<名前>.4g.bin（縦横・Hires 共通）----
    /** models/loras/ にある LoRA の名前（<名前>.4g.bin）。LoRA の欄を開くたびに読み直す。 */
    var loraNames by mutableStateOf(listLoras())
        private set
    /**
     * 使う LoRA と強さ w（選んだ順。空なら使わない）。いくつでも選べる（多重適用。ComfyUI の LoraLoader を重ねるのと同じく、重みの差分を足す）。
     * 強さ w は A1111 の <lora:名前:w> の w と同じ意味（既定 1.0。UNet と CLIP に同じ w）。起動のたびに「なし」から始める（前回の選択は覚えない）。
     * 2 つ以上のときは、生成の直前に adapter を 1 つに合成する（lora/LoraMixer.kt。組み合わせと強さごとに loras/mix/ に作って使い回す）。
     */
    var loraSelection by mutableStateOf<List<LoraUse>>(emptyList())
        private set
    /** UNet（共有 bin）が置かれているか。LoRA もこれで動く。 */
    var loraReady by mutableStateOf(checkLoraReady())
        private set

    private fun lorasDir() = File(native.modelDir, "loras")
    private fun listLoras(): List<String> =
        lorasDir().listFiles { f -> f.isFile && f.name.endsWith(LORA_SUFFIX) }?.map { it.name.removeSuffix(LORA_SUFFIX) }?.sorted() ?: emptyList()

    private fun checkLoraReady(): Boolean = native.hasUnet()

    fun refreshLoras() {
        loraNames = listLoras()
        loraReady = checkLoraReady()
        loraSelection = loraSelection.filter { it.name in loraNames }   // 選んでいた LoRA のファイルが消えた
    }

    /** LoRA を選ぶ・外す（チェックボックス）。選ぶときの強さは 1.0。 */
    fun toggleLora(name: String) {
        loraSelection = if (loraSelection.any { it.name == name }) loraSelection.filter { it.name != name } else loraSelection + LoraUse(name, 1.0f)
    }

    fun clearLoras() {
        loraSelection = emptyList()
    }

    fun updateLoraStrength(name: String, w: Float) {
        loraSelection = loraSelection.map { if (it.name == name) it.copy(strength = w) else it }
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
    /** 取り込みに成功するたびに 1 増える。LoRA 追加のダイアログは、これが変わったら入力欄を空に戻す（同じ入力での二重の追加を防ぐ）。 */
    var loraImportedCount by mutableStateOf(0)
        private set

    /** ダイアログを開くとき、前回の結果（メッセージ・失敗の印）を消す。 */
    fun clearLoraImportMessage() {
        loraImportMessage = ""
        loraImportFailed = false
        loraImportProgress = 0f
    }
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
        DownloadStatus.update(LORA_TITLE, "準備中", 0f)   // 通知に出す（アプリを閉じても、取り込みは続く）
        DownloadService.start(getApplication())
        viewModelScope.launch {
            try {
                val outcome = withContext(Dispatchers.IO) {
                    run(newConverter(), File(getApplication<Application>().cacheDir, "lora_download")) { msg, f ->
                        loraImportMessage = msg
                        loraImportProgress = f
                        DownloadStatus.update(LORA_TITLE, msg, f)
                    }
                }
                refreshLoras()
                loraImportProgress = 1f
                loraImportMessage = "「${outcome.name}」を追加しました（はみ出し ${"%.4f".format(outcome.result.overRatio * 100)}%）。" +
                    if (outcome.activation.isEmpty()) "" else "\nactivation text: ${outcome.activation}"
                loraImportedCount++
                DownloadStatus.finish(LORA_TITLE, "「${outcome.name}」を追加しました", failed = false)
            } catch (e: CancellationException) {
                DownloadStatus.finish(LORA_TITLE, "中断しました", failed = true)
                throw e
            } catch (e: Exception) {
                loraImportFailed = true
                loraImportMessage = e.message ?: e.toString()
                DownloadStatus.finish(LORA_TITLE, "失敗しました。アプリを開いて確かめてください", failed = true)
            } finally {
                loraImporting = false
            }
        }
    }

    /** 変換・合成の道具（形の表は APK の assets。loraconverter/data と同じもの）。重いので IO のスレッドで作る。 */
    private fun newConverter(): LoraConverter {
        val assets = getApplication<Application>().assets
        fun read(n: String) = assets.open(n).bufferedReader().use { it.readText() }
        return LoraConverter(read("layout.json"), read("attach_map.json"), read("super_4g_windows.json"), read("clip_layout.json"))
    }

    /** この LoRA の係数 c（none モードで B を c で割って焼いてあるので、実行時に掛け戻す）。loras/<名前>.json の {"c": 1.518}。無ければ 1。 */
    private fun loraCoefficient(name: String): Float = try {
        val t = File(lorasDir(), "$name.json").readText()
        Regex("\"c\"\\s*:\\s*([0-9.eE+-]+)").find(t)?.groupValues?.get(1)?.toFloat() ?: 1f
    } catch (_: Exception) { 1f }

    /** 生成に使う LoRA: 選んだもののうち、adapter があって強さが 0 でないもの（選んだ順）。UNet が無ければ空。 */
    private fun activeLoras(): List<LoraUse> =
        if (!checkLoraReady()) emptyList() else loraSelection.filter { it.strength != 0f && File(lorasDir(), it.name + LORA_SUFFIX).isFile }

    /**
     * req.loras から、エンジンに渡す adapter と alpha を決める（IO のスレッドで呼ぶ）。
     * 1 つなら、その LoRA の adapter（alpha = w × c、CLIP は w）。2 つ以上なら、合成した adapter（loras/mix/<鍵>。alpha = 合成の係数 C、CLIP は 1）。
     * 合成した adapter が無ければ作る（数十秒。進捗は Stage.MIX）。鍵は、名前・強さ・各 adapter のファイルの大きさと時刻から決める。
     */
    private fun resolveLora(req: GenerationRequest, cancelled: () -> Boolean): GenerationRequest {
        val uses = req.loras
        if (uses.isEmpty()) return req.copy(lora = "", loraAlpha = 0f, loraTeAlpha = 0f)
        if (uses.size == 1) {
            val u = uses[0]
            return req.copy(lora = u.name, loraAlpha = u.strength * loraCoefficient(u.name), loraTeAlpha = u.strength)
        }
        val sorted = uses.sortedBy { it.name }   // 順によらず同じ鍵・同じ中身にする（足し算なので順は効かない）
        val key = mixKey(sorted)
        val dir = File(lorasDir(), MIX_DIR)
        val meta = File(dir, "$key.json")
        if (meta.isFile && File(dir, key + LORA_SUFFIX).isFile) {
            meta.setLastModified(System.currentTimeMillis())   // 最近使ったものとして残す
        } else {
            pruneMixes(dir, keep = MIX_KEEP - 1)
            onProgress(Progress(Stage.MIX, 0, 100))
            val parts = sorted.map { LoraMixer.Part(it.name, it.strength, loraCoefficient(it.name).toDouble()) }
            val r = LoraMixer(newConverter()).mix(parts, lorasDir(), native.modelDir, dir, key,
                progress = { f -> onProgress(Progress(Stage.MIX, (f * 100).toInt().coerceIn(0, 100), 100)) }, cancelled = cancelled)
            android.util.Log.i("dsxl-lora", "mix $key: c=${r.c} truncated=${r.truncated}/${r.tensors} over=${r.overRatio} clip=${r.clipFiles}")
        }
        val c = org.json.JSONObject(meta.readText()).getDouble("c").toFloat()
        return req.copy(lora = "$MIX_DIR/$key", loraAlpha = c, loraTeAlpha = 1f)
    }

    /** 合成の鍵（16 進 16 桁）。名前・強さ・各 adapter（UNet と CLIP）のファイルの大きさと時刻。合成の作り方を変えたら版（v1）を上げる。 */
    private fun mixKey(uses: List<LoraUse>): String {
        val s = buildString {
            append("v1")
            for (u in uses) {
                append('|').append(u.name).append(':').append("%.3f".format(java.util.Locale.US, u.strength))
                for (suffix in listOf(LORA_SUFFIX, ".clip_l.bin", ".clip_g.bin")) {
                    val f = File(lorasDir(), u.name + suffix)
                    append(':').append(if (f.isFile) "${f.length()}@${f.lastModified()}" else "-")
                }
            }
        }
        return java.security.MessageDigest.getInstance("SHA-1").digest(s.toByteArray()).toHex().take(16)
    }

    /** loras/mix/ の合成した adapter を、最近使った keep 組だけ残して消す（1 組 約 310MB）。作りかけ（.json の無いもの）も消す。 */
    private fun pruneMixes(dir: File, keep: Int) {
        val files = dir.listFiles() ?: return
        val recent = files.filter { it.name.endsWith(".json") }.sortedByDescending { it.lastModified() }.take(keep).map { it.name.removeSuffix(".json") }.toSet()
        for (f in files) if (f.name.substringBefore('.') !in recent) f.delete()
    }

    /** loraActivation の覚え（名前 → ファイルの時刻・大きさと、読んだ値）。LoRA 欄は画面の更新のたびに呼ぶので、ファイルが変わったときだけ読み直す。 */
    private val activationCache = HashMap<String, Triple<Long, Long, String>>()

    /** この LoRA の activation text（トリガーの文。loras/<名前>.json の "activation"）。無ければ空。LoRA 欄の [コピー] でクリップボードに入れる。 */
    fun loraActivation(name: String): String {
        val f = File(lorasDir(), "$name.json")
        val modified = f.lastModified()
        val length = f.length()
        activationCache[name]?.let { (m, l, v) -> if (m == modified && l == length) return v }
        val v = try { org.json.JSONObject(f.readText()).optString("activation", "") } catch (_: Exception) { "" }
        activationCache[name] = Triple(modified, length, v)
        return v
    }

    /** 生成中の途中経過の絵（TAESD）。最終画像が出る・失敗する・キャンセルすると消える。 */
    var livePreview by mutableStateOf<Bitmap?>(null)
        private set
    /** 途中経過の絵を作っているジョブの向き。選択中の向き（orientation）は次の生成の向きなので、絵の枠はこちらで決める。 */
    var previewOrientation by mutableStateOf(Orientation.PORTRAIT)
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

    /**
     * 検索語（"-語句" は除外）と「除外」を適用した履歴。履歴・検索語・除外の設定が変わったときだけ作り直す
     * （画面の更新のたびに、全件の照合と除外ワードの分解をやり直さない）。
     */
    val filteredHistory: List<HistoryItem> by derivedStateOf {
        val terms = historySearch.trim().lowercase().split(SPACES).filter { it.isNotEmpty() }
        val include = terms.filter { !it.startsWith("-") }
        val exclude = terms.filter { it.startsWith("-") && it.length > 1 }.map { it.drop(1) }
        val blocked = if (blurBlacklist) imageBlacklistTags.split(BLACKLIST_SEPARATORS).map { w -> w.trim().lowercase() }.filter { w -> w.isNotEmpty() } else emptyList()
        history.filter { h ->
            val t = searchText(h)
            blocked.none { t.contains(it) } && include.all { t.contains(it) } && exclude.none { t.contains(it) }
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
        modelsNeedDownload = computeNeedDownload()
        taesdMissingFile = missingTaesdFile()
    }

    private fun computeNeedDownload(): Boolean = downloader.missing(modelManifest).isNotEmpty()

    /** 足りないモデルを取る。途中で止まっても（アプリを閉じても）次回は続きから。取り終えると、本体の画面に切り替わる。 */
    fun startModelDownload() {
        if (downloading || !modelIdle) return
        downloadMessage = null
        downloadFailed = false
        downloadProgress = null
        DownloadStatus.update(MODEL_TITLE, "準備中", 0f)   // 通知に出す（アプリを閉じても、取得は続く）
        DownloadService.start(getApplication())
        downloadJob = viewModelScope.launch {
            try {
                downloader.download(modelManifest) { p ->
                    downloadProgress = p
                    val fraction = if (p.totalBytes > 0) p.doneBytes.toFloat() / p.totalBytes else 0f
                    DownloadStatus.update(
                        MODEL_TITLE,
                        "${p.fileIndex + 1} / ${p.fileCount}  ${ModelDownloader.gb(p.doneBytes)} / ${ModelDownloader.gb(p.totalBytes)} GB",
                        fraction,
                    )
                }
                downloadMessage = "取得が終わりました。"
                DownloadStatus.finish(MODEL_TITLE, "完了しました。アプリでモデルを読み込めます", failed = false)
            } catch (e: CancellationException) {
                DownloadStatus.finish(MODEL_TITLE, "中断しました（次回は続きから）", failed = true)
                throw e
            } catch (e: Exception) {
                downloadFailed = true
                downloadMessage = "取得に失敗しました: ${e.message ?: e.javaClass.simpleName}（通信を確かめて、もう一度押してください）"
                DownloadStatus.finish(MODEL_TITLE, "失敗しました。アプリを開いて、もう一度押してください", failed = true)
            } finally {
                refreshNativeStatus()
            }
        }
    }

    /** 生成。hires steps が 0（または Hires 用モデルなし）なら Base のみ。 */
    fun generate() {
        if (busy) return
        pushPromptUndo(prompt.trim())   // 「戻る」で、生成に使ったプロンプトへ戻れるようにする
        val s = seed.trim().toLongOrNull() ?: Random.nextLong(0, 4_294_967_295L)
        val hires = hiresStepsValue > 0 && hiresAvailable
        // LoRA は空なら無し（エンジンが default の adapter を使う）。adapter と alpha は、ジョブの中で決める（2 つ以上なら合成するので重い）
        val req = GenerationRequest(prompt.trim(), orientation, s, hires, baseStepsValue, hiresStepsValue.coerceAtLeast(1), hiresDenoiseValue, draftMode,
            loras = activeLoras())
        launchJob(orientation) {
            val ready = withContext(Dispatchers.IO) {
                val ctx = coroutineContext
                resolveLora(req) { !ctx.isActive }
            }
            engine.begin(ready, ::onProgress)
        }
    }

    /** ✨ Hires（SSXL の hiresActionBtn）。直近の Base のみの結果にだけ掛けられる。 */
    fun refine() {
        val (baseId, base) = lastBase ?: return
        if (busy || activeId != baseId) return
        launchJob(base.request.orientation, refining = true) { engine.beginRefine(base, hiresStepsValue.coerceAtLeast(1), hiresDenoiseValue, draftMode, ::onProgress) }
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
    private fun launchJob(orientation: Orientation, refining: Boolean = false, start: suspend () -> PendingGeneration) {
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
        previewOrientation = orientation
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
        val (have, missing) = a.loras.partition { File(lorasDir(), it.name + LORA_SUFFIX).isFile }
        loraSelection = have
        if (missing.isNotEmpty()) doneNote = "LoRA「${missing.joinToString("」「") { it.name }}」が無いので、外しました"
    }

    fun thumbnail(item: HistoryItem): Bitmap? = store.thumbnail(item, maxSide = 360)

    fun saveToGallery() {
        val bmp = previewBitmap ?: return
        val uri = store.saveToGallery(bmp, "dsxl_${System.currentTimeMillis()}", activeItem)
        error = if (uri != null) null else "保存に失敗しました"
        if (uri != null) doneNote = "ギャラリーに保存しました（Pictures/dragonspeedXL）"
    }

    /**
     * 書き出した zip を読み込んで、履歴に足す。同じ id の項目は飛ばす。裏で進み、通知に出す。
     * 読み込めたら、履歴の一覧を読み直す（追加された分が出る）。
     */
    fun importHistory(uri: Uri) {
        DownloadStatus.update(IMPORT_TITLE, "準備中", 0f)
        DownloadService.start(getApplication())
        viewModelScope.launch {
            try {
                val r = withContext(Dispatchers.IO) {
                    store.importZip({ getApplication<Application>().contentResolver.openInputStream(uri) ?: error("読み込み元を開けない") }) { done, total ->
                        DownloadStatus.update(IMPORT_TITLE, "$done / $total 枚", done.toFloat() / total)
                    }
                }
                history = store.load()
                historyPage = historyPage.coerceIn(1, historyPageCount)
                doneNote = "履歴を読み込みました（追加 ${r.added} 件、同じ id で飛ばした ${r.skipped} 件、画像なし ${r.noImage} 件）"
                DownloadStatus.finish(IMPORT_TITLE, "${r.added} 件を追加しました", failed = false)
            } catch (e: CancellationException) {
                DownloadStatus.finish(IMPORT_TITLE, "中断しました", failed = true)
                throw e
            } catch (e: Exception) {
                error = "読み込みに失敗: ${e.message ?: e.javaClass.simpleName}"
                DownloadStatus.finish(IMPORT_TITLE, "読み込みに失敗しました（zip の形式を確かめて）", failed = true)
            }
        }
    }

    /** 書き出しのファイル名（dragonspeedXL-history-yyyyMMdd-HHmm.zip）。 */
    fun historyExportName(): String =
        "dragonspeedXL-history-" + java.text.SimpleDateFormat("yyyyMMdd-HHmm", java.util.Locale.JAPAN).format(java.util.Date()) + ".zip"

    /**
     * 履歴を zip に書き出す（画像と index.json）。書き出し先は、ユーザーが選んだ場所（uri）。
     * 裏で進み、通知に進み具合を出す（アプリを閉じても続く）。
     */
    fun exportHistory(uri: Uri) {
        val items = history.toList()
        if (items.isEmpty()) {
            error = "書き出す履歴がありません"
            return
        }
        DownloadStatus.update(EXPORT_TITLE, "準備中", 0f)
        DownloadService.start(getApplication())
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val out = getApplication<Application>().contentResolver.openOutputStream(uri) ?: error("書き出し先を開けない")
                    out.use { o ->
                        store.exportZip(o, items) { done, total ->
                            DownloadStatus.update(EXPORT_TITLE, "$done / $total 枚", done.toFloat() / total)
                        }
                    }
                }
                doneNote = "履歴を書き出しました（${items.size} 件）"
                DownloadStatus.finish(EXPORT_TITLE, "${items.size} 件を書き出しました", failed = false)
            } catch (e: CancellationException) {
                DownloadStatus.finish(EXPORT_TITLE, "中断しました", failed = true)
                throw e
            } catch (e: Exception) {
                error = "書き出しに失敗: ${e.message ?: e.javaClass.simpleName}"
                DownloadStatus.finish(EXPORT_TITLE, "書き出しに失敗しました", failed = true)
            }
        }
    }

    override fun onCleared() {
        root.stop()
        native.release()
    }

    companion object {
        const val DEFAULT_PROMPT = "hatsune miku"

        // 裏の取得（通知の見出し）
        private const val MODEL_TITLE = "モデルをダウンロード中"
        private const val LORA_TITLE = "LoRA を取り込み中"
        private const val EXPORT_TITLE = "履歴を書き出し中"
        private const val IMPORT_TITLE = "履歴を読み込み中"

        // ネイティブ（pipeline.h の ModelBit）の常駐モデルの状態のビット
        private const val BIT_CLIP_L = 1 shl 0
        private const val BIT_CLIP_G = 1 shl 1
        private const val BIT_VAE = 1 shl 2
        private const val BIT_BASE_UNET = 1 shl 3
        private const val BIT_GPU_VAE = 1 shl 5
        private const val LORA_SUFFIX = com.dragonspeedxl.app.lora.LoraConverter.SUFFIX
        /** 合成した adapter の置き場（models/loras/mix/。LoRA の一覧には出ない）。エンジンには "mix/<鍵>" として渡す */
        private const val MIX_DIR = "mix"
        /** 残しておく合成の組の数（最近使ったもの） */
        private const val MIX_KEEP = 2
        private const val BIT_LOADING = 1 shl 6
        private const val BIT_TILE = 1 shl 7

        private val SPACES = Regex("\\s+")
        private val BLACKLIST_SEPARATORS = Regex("[,\\s]+")
    }
}
