package com.dragonspeedxl.app.lora

import com.dragonspeedxl.app.download.toHex
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Civitai のモデルページの URL（modelVersionId 付きでも、無しでも）から、LoRA の safetensors と activation text（trainedWords）を取る。
 * トークンは Civitai のホストにだけ送る（リダイレクト先の CDN へは送らない）。
 */
class CivitaiClient(private val token: String) {
    /** sha256 は Civitai が返す、ファイルの SHA256（小文字の 16 進）。無ければ null（その LoRA は取り込まない）。 */
    class Version(val id: Long, val fileName: String, val sizeKb: Double, val trainedWords: List<String>, val sha256: String?)

    fun fetchVersion(url: String): Version {
        val (modelId, versionId0) = parseUrl(url)
        var versionId = versionId0
        if (versionId == null) {
            val model = JSONObject(getText("$API/models/$modelId"))
            versionId = model.getJSONArray("modelVersions").getJSONObject(0).getLong("id")   // 指定が無ければ最新版
        }
        return parseVersion(JSONObject(getText("$API/model-versions/$versionId")))
    }

    /**
     * safetensors を dest に保存する（途中は .part。サイズが分かれば progress(取得済み, 全体) を呼ぶ）。
     * Civitai のハッシュ（SHA256）と合わなければ、置かずに失敗する。ハッシュが無いファイルも取り込まない。
     */
    fun download(v: Version, dest: File, progress: (Long, Long) -> Unit = { _, _ -> }) {
        val expected = v.sha256 ?: error("Civitai がハッシュを返さないので取り込まない（中身を確かめられない）")
        val part = File(dest.path + ".part")
        part.parentFile?.mkdirs()
        val md = MessageDigest.getInstance("SHA-256")
        open("$DOWNLOAD/${v.id}").use { (conn, stream) ->
            val total = conn.contentLengthLong.takeIf { it > 0 } ?: (v.sizeKb * 1024).toLong()
            part.outputStream().use { out ->
                val buf = ByteArray(1 shl 16)
                var done = 0L
                var last = 0L
                while (true) {
                    val n = stream.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    md.update(buf, 0, n)
                    done += n
                    if (done - last >= (1 shl 20)) { progress(done, total); last = done }
                }
                progress(done, total)
            }
        }
        val got = md.digest().toHex()
        if (got != expected) {
            part.delete()
            error("ダウンロードした LoRA のハッシュが合わない（取り込みを止めた）")
        }
        dest.delete()
        check(part.renameTo(dest)) { "${dest.name} に名前を変えられない" }
    }

    private fun getText(url: String): String = open(url).use { (_, s) -> s.readBytes().toString(Charsets.UTF_8) }

    private class Opened(val conn: HttpURLConnection, val stream: java.io.InputStream) : java.io.Closeable {
        operator fun component1() = conn
        operator fun component2() = stream
        override fun close() { stream.close(); conn.disconnect() }
    }

    /** GET。リダイレクトは自分で辿り、トークンは Civitai のホスト（https）にだけ付ける。https 以外へは接続しない。 */
    private fun open(url0: String): Opened {
        var url = url0
        for (hop in 0..5) {
            require(URL(url).protocol == "https") { "https 以外への接続は許さない" }
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.instanceFollowRedirects = false
            conn.connectTimeout = 20_000
            conn.readTimeout = 60_000
            conn.setRequestProperty("User-Agent", "dragonspeedXL/1.0")
            if (token.isNotEmpty() && isCivitaiHost(url)) conn.setRequestProperty("Authorization", "Bearer $token")
            val code = conn.responseCode
            if (code in 300..399) {
                val loc = conn.getHeaderField("Location") ?: error("リダイレクト先が無い（HTTP $code）")
                conn.disconnect()
                url = URL(URL(url), loc).toString()
                continue
            }
            if (code !in 200..299) {
                conn.disconnect()
                val hint = if (code == 401 || code == 403) "（トークンが無効か、ログインが要るモデル）" else ""
                error("Civitai の通信が失敗: HTTP $code$hint")
            }
            return Opened(conn, conn.inputStream)
        }
        error("リダイレクトが多すぎる")
    }

    companion object {
        private const val API = "https://civitai.com/api/v1"
        private const val DOWNLOAD = "https://civitai.com/api/download/models"

        /** API の JSON から Version を作る。safetensors のうち、主のファイルを選ぶ。 */
        internal fun parseVersion(v: JSONObject): Version {
            val files = v.getJSONArray("files")
            val cands = (0 until files.length()).map { files.getJSONObject(it) }.filter { it.optString("name").endsWith(".safetensors") }
            require(cands.isNotEmpty()) { "このバージョンに safetensors が無い" }
            val f = cands.sortedWith(compareBy({ it.optString("type") != "Model" }, { !it.optBoolean("primary", false) })).first()
            val words = v.optJSONArray("trainedWords")
            val sha = f.optJSONObject("hashes")?.optString("SHA256")?.takeIf { it.isNotEmpty() }?.lowercase()
            return Version(
                v.getLong("id"), f.getString("name"), f.optDouble("sizeKB", 0.0),
                words?.let { w -> List(w.length()) { w.getString(it) } } ?: emptyList(), sha,
            )
        }

        /** トークンを送ってよい URL か。https で、ホストが civitai.com と完全に一致するときだけ（似た名前や CDN のサブドメインは含めない）。 */
        internal fun isCivitaiHost(url: String): Boolean {
            val u = runCatching { URL(url) }.getOrNull() ?: return false
            return u.protocol == "https" && u.host.equals("civitai.com", ignoreCase = true)
        }

        /** (モデル ID, バージョン ID)。無いほうは null。 */
        fun parseUrl(url: String): Pair<Long?, Long?> {
            val u = url.trim()
            val version = Regex("[?&]modelVersionId=(\\d+)").find(u)?.groupValues?.get(1)?.toLong()
            Regex("/api/download/models/(\\d+)").find(u)?.let { return null to it.groupValues[1].toLong() }
            Regex("/models/(\\d+)").find(u)?.let { return it.groupValues[1].toLong() to version }
            throw IllegalArgumentException("Civitai のモデルの URL ではありません: $url")
        }
    }
}
