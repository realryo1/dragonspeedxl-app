package com.dragonspeedxl.app.prompt

import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * 端末内のタグ辞書のファイル（dskjal の「Danbooru タグ検索」の CSV）を取得して置いておく。
 * 単語の一覧はリポジトリにも APK にも入れず、アプリが初めて要るときに dskjal から落とす（約 3MB、1 回だけ）。
 * dskjal は Danbooru とは別のサーバーなので、Danbooru がメンテナンス中でも取れる。
 *
 *   一覧: https://dskjal.com/spa/danbooru-tag-search/dbs/db_list.txt（1 行 1 ファイル名。日付の名前で、最大が最新）
 *   本体: https://dskjal.com/spa/danbooru-tag-search/dbs/tags-YYYY-MM-DD.csv
 * 置き場: `<dir>/tags-YYYY-MM-DD.csv`（新しい版を取ったら古い版は消す）。
 */
class TagDictionary(private val dir: File) {

    /** 置いてある辞書（無ければ null）。 */
    fun file(): File? =
        dir.listFiles { f -> f.isFile && NAME.matches(f.name) }?.maxByOrNull { it.name }

    /** 無ければ dskjal から取る。取れなければ例外。 */
    @Synchronized
    fun ensure(): File {
        file()?.let { return it }
        dir.mkdirs()
        val name = String(get(BASE + "db_list.txt"), Charsets.UTF_8).lineSequence()
            .map { it.trim() }
            .filter { NAME.matches(it) }
            .maxOrNull() ?: throw IllegalStateException("db_list.txt に辞書が見つからない")
        val bytes = get(BASE + name)
        // 中身が CSV でない（エラーページなど）ものを置かない
        var tags = 0
        LocalTagIndex.parseCsv(String(bytes, Charsets.UTF_8).reader()) { f -> if (f.size == 4 && f[1].trim().toLongOrNull() != null) tags++ }
        if (tags < MIN_TAGS) throw IllegalStateException("辞書が小さすぎる（$tags タグ）")
        val tmp = File(dir, "$name.part")
        tmp.writeBytes(bytes)
        val dst = File(dir, name)
        if (!tmp.renameTo(dst)) { tmp.delete(); throw IllegalStateException("辞書を保存できない") }
        dir.listFiles { f -> f.isFile && f.name != name && (NAME.matches(f.name) || f.name.endsWith(".part")) }?.forEach { it.delete() }
        return dst
    }

    private fun get(url: String): ByteArray {
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 8000
            readTimeout = 30000
            setRequestProperty("User-Agent", "dragonspeedXL/1.0")
        }
        try {
            if (conn.responseCode !in 200..299) throw IllegalStateException("dskjal ${conn.responseCode}")
            return conn.inputStream.use { it.readBytes() }
        } finally {
            conn.disconnect()
        }
    }

    companion object {
        const val BASE = "https://dskjal.com/spa/danbooru-tag-search/dbs/"
        private val NAME = Regex("tags-\\d{4}-\\d{2}-\\d{2}\\.csv")
        private const val MIN_TAGS = 1000
    }
}
