package com.dragonspeedxl.app.data

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import com.dragonspeedxl.app.engine.GenerationResult
import com.dragonspeedxl.app.engine.Orientation
import com.dragonspeedxl.app.lora.LoraConverter
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.io.IOException
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

data class HistoryItem(
    val id: String,
    val prompt: String,
    val seed: Long,
    val orientation: Orientation,
    val hires: Boolean,
    val totalMillis: Long,
    val createdAt: Long,
    val baseSteps: Int = 2,
    val hiresSteps: Int = 4,
    val denoise: Double = 0.39,
    /** 表示用の印。"_half" は旧機能（UNet の変種）、"_draft" は速度優先（TAESD を最終画像にした）で生成した履歴。ふつうは空。 */
    val variant: String = "",
    /** 使った LoRA の名前（models/loras/<名前>.4g.bin）。空なら LoRA なし。流用のとき、LoRA の欄をこれに合わせる。 */
    val lora: String = "",
    /** そのときの LoRA の強さ w（LoRA なしなら 0）。流用のとき、LoRA の強さをこれに合わせる。 */
    val loraStrength: Float = 0f,
) {
    /** SSXL の webui と同じ形式のジョブ ID（16 進 10 桁。例 58e078900f）。id（作成時刻ミリ秒）の下位 40 ビット。 */
    val jobId: String get() = "%010x".format((id.toLongOrNull() ?: 0L) and 0xFFFFFFFFFFL)

    /** SSXL の kindBadge と同じ表記（BASE / HIRES） */
    val kind: String get() = if (hires) "HIRES" else "BASE"

    /** 履歴タイルのラベル。縦長・変種のときは付け足す（例: HIRES·縦·half）。 */
    val tileLabel: String get() = kind + (if (orientation == Orientation.PORTRAIT) "·縦" else "") + (if (variant.isEmpty()) "" else "·" + variant.trimStart('_'))
}

/** 履歴の取り込みの結果。added は足した件数、skipped は同じ id が既にあって飛ばした件数、noImage は画像が zip に無くて足さなかった件数。 */
data class ImportResult(val added: Int, val skipped: Int, val noImage: Int)

/** 生成履歴。画像は filesDir/history/<id>.png、メタデータは index.json（新しい順）。 */
class HistoryStore(context: Context) {
    private val appContext = context.applicationContext
    private val dir = File(appContext.filesDir, "history").apply { mkdirs() }
    private val index = File(dir, "index.json")
    private val lock = Any()   // index.json の読み書き（保存を裏で行うので、次のジョブの保存や削除と重なっても壊れないようにする）

    fun load(): List<HistoryItem> = synchronized(lock) {
        if (!index.exists()) return@synchronized emptyList()
        runCatching { parseItems(index.readText()) }.getOrDefault(emptyList())
    }

    /** index.json（一覧の JSON）を項目に読む。形が違えば例外。 */
    private fun parseItems(text: String): List<HistoryItem> {
        val arr = JSONArray(text)
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            HistoryItem(
                id = o.getString("id"),
                prompt = o.getString("prompt"),
                seed = o.getLong("seed"),
                orientation = Orientation.valueOf(o.getString("orientation")),
                hires = o.getBoolean("hires"),
                totalMillis = o.getLong("totalMillis"),
                createdAt = o.getLong("createdAt"),
                baseSteps = o.optInt("baseSteps", 2),
                hiresSteps = o.optInt("hiresSteps", 4),
                denoise = o.optDouble("denoise", 0.39),
                variant = o.optString("variant", ""),
                lora = o.optString("lora", ""),
                loraStrength = o.optDouble("loraStrength", 0.0).toFloat(),
            )
        }
    }

    /** 生成開始時に決めるジョブ ID の元（作成時刻ミリ秒）。ステータス行の「Job #…」に先に出すため、add より前に払い出す。 */
    fun newId(): String = System.currentTimeMillis().toString()

    /** 履歴の項目をメモリ上で作るだけ（ディスクには書かない）。画像を先に表示し、保存は save で裏で行うため。 */
    fun itemFor(result: GenerationResult, id: String = newId()): HistoryItem = HistoryItem(
        id, result.request.prompt, result.request.seed, result.request.orientation,
        result.hiresDone, result.totalMillis, System.currentTimeMillis(),
        result.request.steps, if (result.hiresDone) result.request.hiresSteps else 0, result.request.hiresDenoise,   // Hires なしは 0 と記録する
        variant = if (result.request.draft) "_draft" else "",   // 速度優先（TAESD が最終画像）は、バッジとタイルのラベルに draft と出す
        lora = result.request.lora,
        loraStrength = if (result.request.lora.isEmpty()) 0f else result.request.loraTeAlpha,   // loraTeAlpha は LoRA の強さ w（係数 c は掛けない）
    )

    /** 画像（PNG。1080p で重い）と index.json を書く。画像には生成情報（A1111 互換の parameters）を入れる。別スレッドで呼んでよい（排他あり）。 */
    fun save(result: GenerationResult, item: HistoryItem) {
        File(dir, "${item.id}.png").writeBytes(PngInfo.embed(encodePng(result.bitmap), PngInfo.infotext(item)))   // 圧縮は排他の外（長いので、他の操作を待たせない）
        synchronized(lock) { write(listOf(item) + load()) }
    }

    fun add(result: GenerationResult, id: String = newId()): HistoryItem = itemFor(result, id).also { save(result, it) }

    fun delete(item: HistoryItem) {
        File(dir, "${item.id}.png").delete()
        synchronized(lock) { write(load().filterNot { it.id == item.id }) }
    }

    fun deleteAll(items: Collection<HistoryItem>) {
        val ids = items.map { it.id }.toSet()
        ids.forEach { File(dir, "$it.png").delete() }
        synchronized(lock) { write(load().filterNot { it.id in ids }) }
    }

    fun loadBitmap(item: HistoryItem, maxSide: Int = 0): Bitmap? {
        val f = File(dir, "${item.id}.png")
        if (!f.exists()) return null
        val opts = BitmapFactory.Options()
        if (maxSide > 0) {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(f.path, bounds)
            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
            opts.inSampleSize = sample
        }
        return BitmapFactory.decodeFile(f.path, opts)
    }

    /** ギャラリー（Pictures/dragonspeedXL）へ保存。権限不要（MediaStore）。item があれば、履歴と同じ生成情報を入れる。 */
    fun saveToGallery(bitmap: Bitmap, name: String, item: HistoryItem?): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "$name.png")
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/dragonspeedXL")
        }
        val resolver = appContext.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        val png = encodePng(bitmap).let { if (item == null) it else PngInfo.embed(it, PngInfo.infotext(item)) }
        resolver.openOutputStream(uri)?.use { it.write(png) }
        return uri
    }

    private fun encodePng(bitmap: Bitmap): ByteArray = ByteArrayOutputStream().also { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray()

    private fun write(items: List<HistoryItem>) {
        index.writeText(toJson(items, imagePath = false).toString())
    }

    /** 一覧の JSON。書き出しのときは、zip の中の画像の場所（images/<id>.png）も入れる。 */
    private fun toJson(items: List<HistoryItem>, imagePath: Boolean): JSONArray {
        val arr = JSONArray()
        items.forEach {
            val o = JSONObject()
                .put("id", it.id).put("prompt", it.prompt).put("seed", it.seed)
                .put("orientation", it.orientation.name).put("hires", it.hires)
                .put("totalMillis", it.totalMillis).put("createdAt", it.createdAt)
                .put("baseSteps", it.baseSteps).put("hiresSteps", it.hiresSteps).put("denoise", it.denoise).put("variant", it.variant)
                .put("lora", it.lora).put("loraStrength", it.loraStrength.toDouble())
            if (imagePath && File(dir, "${it.id}.png").isFile) o.put("image", "images/${it.id}.png")
            arr.put(o)
        }
        return arr
    }

    /**
     * 履歴を zip に書き出す（画像と一覧）。
     * 中身: index.json（メタデータ。新しい順）と、images/<id>.png（各画像。PNG の中に生成情報が入っている）。
     * 画像が無い項目は、一覧には入れるが画像は入れない。onItem(済んだ枚数, 全体)。
     */
    fun exportZip(out: OutputStream, items: List<HistoryItem>, onItem: (Int, Int) -> Unit) {
        ZipOutputStream(out).use { zip ->
            zip.putNextEntry(ZipEntry("index.json"))
            zip.write(toJson(items, imagePath = true).toString(2).toByteArray(Charsets.UTF_8))
            zip.closeEntry()
            items.forEachIndexed { i, it ->
                val f = File(dir, "${it.id}.png")
                if (f.isFile) {
                    zip.putNextEntry(ZipEntry("images/${it.id}.png"))
                    f.inputStream().use { input -> input.copyTo(zip) }
                    zip.closeEntry()
                }
                onItem(i + 1, items.size)
            }
        }
    }

    /**
     * 書き出した zip（exportZip の形式）を読み込んで、履歴に足す。
     * 同じ id の項目は飛ばす（上書きしない）。画像が zip に無い項目は足さない。open は読み直せる入力（index.json を読んでから、画像を取る）。
     */
    fun importZip(open: () -> InputStream, onItem: (Int, Int) -> Unit): ImportResult {
        val items = open().use { raw ->
            ZipInputStream(raw).use { zip ->
                var found: List<HistoryItem>? = null
                generateSequence { zip.nextEntry }.forEach { e ->
                    if (e.name == "index.json") found = parseItems(readLimited(zip, MAX_INDEX_BYTES).toString(Charsets.UTF_8)).map(::safeLora)
                }
                found
            }
        } ?: throw IOException("index.json が zip に無い")
        val existing = load().map { it.id }.toSet()
        val candidates = items.filter { ID.matches(it.id) && it.id !in existing }
        val wantedIds = candidates.map { it.id }.toSet()
        val written = HashSet<String>()
        open().use { raw ->
            ZipInputStream(raw).use { zip ->
                generateSequence { zip.nextEntry }.forEach { e ->
                    val id = IMAGE.matchEntire(e.name)?.groupValues?.get(1) ?: return@forEach
                    if (id !in wantedIds) return@forEach
                    val f = File(dir, "$id.png")
                    val tmp = File(dir, "$id.png.part")
                    if (!copyLimited(zip, tmp, MAX_IMAGE_BYTES)) {   // 上限を超える画像は足さない（noImage に数える）
                        tmp.delete()
                        return@forEach
                    }
                    f.delete()
                    if (tmp.renameTo(f)) {
                        written.add(id)
                        onItem(written.size, wantedIds.size)
                    }
                }
            }
        }
        val added = candidates.filter { it.id in written }
        synchronized(lock) { write((added + load()).sortedByDescending { it.createdAt }) }
        return ImportResult(added = added.size, skipped = items.size - candidates.size, noImage = candidates.size - added.size)
    }

    /** 取り込んだ項目の LoRA 名を確かめる。LoRA の名前として使えない（区切り文字などを含む）なら、LoRA なしの項目にする。 */
    private fun safeLora(item: HistoryItem): HistoryItem {
        if (item.lora.isEmpty()) return item
        val ok = runCatching { LoraConverter.checkName(item.lora) }.isSuccess
        return if (ok) item else item.copy(lora = "", loraStrength = 0f)
    }

    /** zip の今のエントリを、上限まで読む。超えたら例外。 */
    private fun readLimited(input: InputStream, max: Int): ByteArray {
        val out = ByteArrayOutputStream()
        val buf = ByteArray(1 shl 16)
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            if (out.size() + n > max) throw IOException("index.json が大きすぎる（上限 ${max shr 20} MB）")
            out.write(buf, 0, n)
        }
        return out.toByteArray()
    }

    /** 入力を file へ書く。上限を超えたら止めて false（書きかけは呼び出し側が消す）。 */
    private fun copyLimited(input: InputStream, file: File, max: Long): Boolean {
        var total = 0L
        var ok = true
        val buf = ByteArray(1 shl 16)
        file.outputStream().use { out ->
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                total += n
                if (total > max) {
                    ok = false
                    break
                }
                out.write(buf, 0, n)
            }
        }
        return ok
    }

    private companion object {
        val ID = Regex("\\d+")
        val IMAGE = Regex("images/(\\d+)\\.png")
        /** 取り込む zip の上限（一覧は数 MB もあれば十分。画像は 1080p の PNG より十分大きく取る）。 */
        const val MAX_INDEX_BYTES = 8 shl 20
        const val MAX_IMAGE_BYTES = 64L shl 20
    }
}
