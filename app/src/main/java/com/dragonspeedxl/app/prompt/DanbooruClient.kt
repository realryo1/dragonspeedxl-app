package com.dragonspeedxl.app.prompt

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Danbooru のタグ補完。SSXL のサーバー（`/api/danbooru/autocomplete`）と同じ問い合わせを、アプリから直接行う:
 *   GET https://danbooru.donmai.us/autocomplete.json?search[query]=<q>&search[type]=tag_query&version=1&limit=20
 * 同じ語は 30 秒キャッシュする。入力した語は danbooru.donmai.us に送られる（INTERNET 権限が必要）。
 *
 * Danbooru に繋がらないとき（メンテナンス、通信の拒否、HTML が返るなど）は、端末内の辞書（[LocalTagIndex]。
 * ファイルは [TagDictionary] が dskjal から落としたもの）で補完する。失敗のあと [OFFLINE_MS] の間は Danbooru に
 * 問い合わせず、端末内だけで引く（打つたびに通信の待ちが出ないように）。
 * 辞書のファイルは、補完を初めて使ったときに裏で取っておく（[prefetchLocal]）。
 * 設定の「ローカル辞書モード」では、Danbooru に問い合わせず端末内の辞書だけで引く（`autocomplete(…, localOnly = true)`）。
 */
class DanbooruClient(private val dictionary: TagDictionary) {
    data class Tag(val name: String, val label: String, val antecedent: String, val postCount: Long)

    /** 補完の結果。[local] は端末内の辞書で引いたとき true。[onlineError] は Danbooru に失敗した直後だけ入る。 */
    data class Result(val tags: List<Tag>, val local: Boolean, val onlineError: Throwable? = null)

    private val cache = HashMap<String, Pair<Long, List<Tag>>>()
    @Volatile private var offlineUntil = 0L
    @Volatile private var local: LocalTagIndex? = null
    private val localLock = Any()

    /**
     * Danbooru が失敗しても例外にしない（端末内で引く）。端末内の辞書も無く、取れもしないときだけ例外を投げる。
     * @param localOnly true なら Danbooru に問い合わせず、端末内の辞書だけで引く（設定の「ローカル辞書モード」）。
     */
    suspend fun autocomplete(query: String, localOnly: Boolean = false): Result = withContext(Dispatchers.IO) {
        val q = query.trim()
        if (q.isEmpty()) return@withContext Result(emptyList(), local = localOnly)
        val key = q.lowercase()
        if (!localOnly) synchronized(cache) {
            cache[key]?.takeIf { System.currentTimeMillis() - it.first < CACHE_MS }?.let { return@withContext Result(it.second, local = false) }
        }
        var error: Throwable? = null
        if (!localOnly && System.currentTimeMillis() >= offlineUntil) {
            try {
                val tags = fetchOnline(q)
                synchronized(cache) { cache[key] = System.currentTimeMillis() to tags }
                return@withContext Result(tags, local = false)
            } catch (e: Exception) {
                offlineUntil = System.currentTimeMillis() + OFFLINE_MS
                error = e
            }
        }
        val index = try {
            localIndex()
        } catch (e: Exception) {
            error?.let { e.addSuppressed(it) }
            val why = if (localOnly) "端末内の辞書が取れない" else "Danbooru に繋がらず、端末内の辞書も取れない"
            throw IllegalStateException("$why: ${e.message}", e)
        }
        Result(index.search(q, LIMIT), local = true, onlineError = error)
    }

    /** 辞書のファイルが無ければ取っておく。置いてある（取れた）辞書のファイルを返す。失敗は例外（次に要るときにまた取りにいく）。 */
    suspend fun prefetchLocal(): java.io.File = withContext(Dispatchers.IO) { dictionary.ensure() }

    /** 置いてある辞書のファイル（無ければ null。通信しない）。 */
    fun localFile(): java.io.File? = dictionary.file()

    /** 端末内の辞書（初回だけ読む。無ければ取る）。メモリを食うので、Danbooru が使えている間は作らない。 */
    private fun localIndex(): LocalTagIndex =
        local ?: synchronized(localLock) {
            local ?: LocalTagIndex.loadCsv(dictionary.ensure().reader(Charsets.UTF_8)).also { local = it }
        }

    private fun fetchOnline(q: String): List<Tag> {
        val url = URL(
            "https://danbooru.donmai.us/autocomplete.json" +
                "?search%5Bquery%5D=${URLEncoder.encode(q, "UTF-8")}&search%5Btype%5D=tag_query&version=1&limit=$LIMIT",
        )
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 4000
            readTimeout = 4000
            setRequestProperty("User-Agent", "dragonspeedXL/1.0")
            setRequestProperty("Accept", "application/json")
        }
        try {
            if (conn.responseCode !in 200..299) throw IllegalStateException("Danbooru ${conn.responseCode}")
            val arr = JSONArray(conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() })
            val tags = ArrayList<Tag>()
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                val name = o.optString("value").ifEmpty { o.optJSONObject("tag")?.optString("name").orEmpty() }
                if (name.isEmpty()) continue
                tags.add(Tag(name, o.optString("label").ifEmpty { name }, o.optString("antecedent"), o.optLong("post_count", 0L)))
            }
            return tags
        } finally {
            conn.disconnect()
        }
    }

    companion object {
        private const val CACHE_MS = 30_000L
        /** Danbooru に失敗したあと、端末内だけで引く時間。 */
        private const val OFFLINE_MS = 5 * 60_000L
        private const val LIMIT = 20
        fun wikiUrl(name: String) = "https://danbooru.donmai.us/wiki_pages/${URLEncoder.encode(name.trim(), "UTF-8").replace("+", "%20")}"
    }
}
