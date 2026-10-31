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
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File

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

/** 生成履歴。画像は filesDir/history/<id>.png、メタデータは index.json（新しい順）。 */
class HistoryStore(context: Context) {
    private val appContext = context.applicationContext
    private val dir = File(appContext.filesDir, "history").apply { mkdirs() }
    private val index = File(dir, "index.json")
    private val lock = Any()   // index.json の読み書き（保存を裏で行うので、次のジョブの保存や削除と重なっても壊れないようにする）

    fun load(): List<HistoryItem> = synchronized(lock) {
        if (!index.exists()) return@synchronized emptyList()
        runCatching {
            val arr = JSONArray(index.readText())
            (0 until arr.length()).map { i ->
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
        }.getOrDefault(emptyList())
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
        val arr = JSONArray()
        items.forEach {
            arr.put(
                JSONObject()
                    .put("id", it.id).put("prompt", it.prompt).put("seed", it.seed)
                    .put("orientation", it.orientation.name).put("hires", it.hires)
                    .put("totalMillis", it.totalMillis).put("createdAt", it.createdAt)
                    .put("baseSteps", it.baseSteps).put("hiresSteps", it.hiresSteps).put("denoise", it.denoise).put("variant", it.variant)
                    .put("lora", it.lora).put("loraStrength", it.loraStrength.toDouble()),
            )
        }
        index.writeText(arr.toString())
    }
}
