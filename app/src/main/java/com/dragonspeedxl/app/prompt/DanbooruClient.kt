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
 */
class DanbooruClient {
    data class Tag(val name: String, val label: String, val antecedent: String, val postCount: Long)

    private val cache = HashMap<String, Pair<Long, List<Tag>>>()

    /** 失敗したら例外を投げる（呼び出し側で補完を閉じる）。 */
    suspend fun autocomplete(query: String): List<Tag> = withContext(Dispatchers.IO) {
        val q = query.trim()
        if (q.isEmpty()) return@withContext emptyList()
        val key = q.lowercase()
        synchronized(cache) {
            cache[key]?.takeIf { System.currentTimeMillis() - it.first < CACHE_MS }?.let { return@withContext it.second }
        }
        val url = URL(
            "https://danbooru.donmai.us/autocomplete.json" +
                "?search%5Bquery%5D=${URLEncoder.encode(q, "UTF-8")}&search%5Btype%5D=tag_query&version=1&limit=20",
        )
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 5000
            readTimeout = 5000
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
            synchronized(cache) { cache[key] = System.currentTimeMillis() to tags }
            tags
        } finally {
            conn.disconnect()
        }
    }

    companion object {
        private const val CACHE_MS = 30_000L
        fun wikiUrl(name: String) = "https://danbooru.donmai.us/wiki_pages/${URLEncoder.encode(name.trim(), "UTF-8").replace("+", "%20")}"
    }
}
