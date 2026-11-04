package com.dragonspeedxl.app.download

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * APK に同梱した manifest（ModelManifest）にあるモデルを、端末の models/ へ取る。
 * - 途中で止まっても、`.part` から続きを取る（HTTP の Range）。
 * - 取り終えたら sha256 を確かめてから、本来の名前に置く（壊れたものは置かない）。
 * - 端末にあって、サイズが manifest と同じファイルは取らない（sha256 は取り直したときだけ確かめる）。
 */
class ModelDownloader(private val modelDir: File) {

    /** 進み具合。doneBytes / totalBytes は全体、fileDone / fileTotal は今のファイル。 */
    data class Progress(
        val fileIndex: Int,
        val fileCount: Int,
        val file: String,
        val fileDone: Long,
        val fileTotal: Long,
        val doneBytes: Long,
        val totalBytes: Long,
    )

    /** 足りない（無い、またはサイズが違う）ファイル。 */
    fun missing(m: ModelManifest): List<ModelEntry> = m.entries.filter { e ->
        val f = File(modelDir, e.localPath)
        !(f.isFile && f.length() == e.size)
    }

    /**
     * 足りないファイルを取る。キャンセルされたら、`.part` を残して止まる（次回は続きから）。
     * 失敗（通信の切断など）は、ファイルごとに数回やり直す。
     */
    suspend fun download(m: ModelManifest, onProgress: (Progress) -> Unit) = withContext(Dispatchers.IO) {
        val todo = missing(m)
        modelDir.mkdirs()
        val need = todo.sumOf { e -> e.size - partLength(e) }
        check(modelDir.usableSpace >= need + SPACE_MARGIN) {
            "空き容量が足りない（必要 ${gb(need)} GB、空き ${gb(modelDir.usableSpace)} GB）"
        }
        var doneAll = m.totalBytes - todo.sumOf { it.size }
        todo.forEachIndexed { i, e ->
            val base = doneAll
            fetchOne(m, e) { fileDone ->
                onProgress(Progress(i, todo.size, e.localPath, fileDone, e.size, base + fileDone, m.totalBytes))
            }
            doneAll += e.size
        }
    }

    private suspend fun fetchOne(m: ModelManifest, e: ModelEntry, onFile: (Long) -> Unit) {
        val dest = File(modelDir, e.localPath)
        val part = File(dest.path + PART)
        dest.parentFile?.mkdirs()
        var tries = 0
        while (true) {
            try {
                if (part.length() != e.size) httpRange(m.url(e), part, e.size, onFile)
                if (part.length() != e.size) throw IOException("途中で切れた: ${e.localPath}")
                if (!sha256Of(part).equals(e.sha256, ignoreCase = true)) {
                    part.delete()
                    throw IOException("sha256 が合わない: ${e.localPath}（取り直す）")
                }
                if (dest.exists()) dest.delete()
                check(part.renameTo(dest)) { "置けない: ${dest.path}" }
                onFile(e.size)
                return
            } catch (c: CancellationException) {
                throw c
            } catch (x: IOException) {
                tries++
                if (tries >= MAX_TRIES) throw x
                delay(2_000L * tries)
            }
        }
    }

    /** `.part` の続きから（Range）取る。Range が無視されたら最初から取り直す。 */
    private suspend fun httpRange(url: String, part: File, size: Long, onFile: (Long) -> Unit) {
        var have = if (part.isFile) part.length() else 0L
        if (have > size) {
            part.delete()
            have = 0L
        }
        val conn = URL(url).openConnection() as HttpURLConnection
        try {
            conn.connectTimeout = 20_000
            conn.readTimeout = 60_000
            if (have > 0) conn.setRequestProperty("Range", "bytes=$have-")
            val code = conn.responseCode
            if (code !in 200..299) throw IOException("HTTP $code（${part.name}）")
            val resume = have > 0 && code == HttpURLConnection.HTTP_PARTIAL
            if (!resume) have = 0L
            conn.inputStream.use { input ->
                FileOutputStream(part, resume).use { out ->
                    val buf = ByteArray(1 shl 16)
                    var done = have
                    var last = done
                    onFile(done)
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        if (done - last >= (1 shl 20)) {
                            onFile(done)
                            last = done
                        }
                    }
                    onFile(done)
                }
            }
        } finally {
            conn.disconnect()
        }
    }

    private fun partLength(e: ModelEntry): Long =
        File(modelDir, e.localPath + PART).let { if (it.isFile) minOf(it.length(), e.size) else 0L }

    companion object {
        private const val PART = ".part"
        private const val MAX_TRIES = 3
        private const val SPACE_MARGIN = 256L shl 20

        fun gb(bytes: Long): String = "%.1f".format(bytes / 1e9)
    }
}

/** ファイルの sha256（16 進の小文字）。 */
internal fun sha256Of(file: File): String {
    val md = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buf = ByteArray(1 shl 20)
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            md.update(buf, 0, n)
        }
    }
    return md.digest().toHex()
}

/** 16 進の小文字（ハッシュの表記。sha256 の照合と LoRA の合成の鍵に使う）。 */
internal fun ByteArray.toHex(): String {
    val s = StringBuilder(size * 2)
    for (b in this) { val v = b.toInt() and 0xFF; s.append(HEX[v ushr 4]).append(HEX[v and 0xF]) }
    return s.toString()
}

private const val HEX = "0123456789abcdef"
