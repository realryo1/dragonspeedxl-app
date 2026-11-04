package com.dragonspeedxl.app.prompt

import java.io.Reader

/**
 * 端末内のタグ辞書。Danbooru に繋がらないときの補完に使う（引くときは通信しない）。
 * 中身は dskjal の「Danbooru タグ検索」の CSV（[TagDictionary] が端末に落としたもの）。
 *   列: タグ名, 投稿数, タググループ（使わない）, 別名・訳（空白区切り。日本語・中国語・韓国語・pixiv のタグなど）
 *
 * 当たり方（Danbooru の autocomplete に寄せる）:
 *   1. タグ名と完全一致
 *   2. タグ名の前方一致、語の頭（`_` `(` `-` の後ろ）の前方一致、別名の前方一致（別名で当たったら「別名 → タグ名」）
 *   3. タグ名・別名の途中に含む（2 文字以上のとき）
 * 同じ段の中は投稿数の多い順。
 */
class LocalTagIndex private constructor(
    private val names: Array<String>,
    private val namesLower: Array<String>,
    private val counts: LongArray,
    /** 別名（表示用の元の形と、比べる用の小文字） */
    private val aliases: Array<Array<String>>,
    private val aliasesLower: Array<Array<String>>,
) {
    val size: Int get() = names.size

    fun search(query: String, limit: Int = 20): List<DanbooruClient.Tag> {
        val raw = query.trim().lowercase()
        if (raw.isEmpty()) return emptyList()
        val q = raw.replace(WS, "_")         // タグ名は空白を _ にする
        val qAlias = raw.replace(WS, " ")    // 別名は空白区切りの 1 語として比べる
        val allowContains = q.length >= 2

        // 段ごとに、投稿数の多い順（= 並べ替え済みの順）で集める
        val tiers = Array(3) { ArrayList<DanbooruClient.Tag>() }
        for (i in names.indices) {
            val n = namesLower[i]
            if (n == q) { tiers[0].add(tag(i, "")); continue }
            if (tiers[0].size + tiers[1].size >= limit) continue   // 上の段が埋まった。残りは完全一致だけ探す
            if (n.startsWith(q) || wordPrefix(n, q)) { tiers[1].add(tag(i, "")); continue }
            val al = aliasesLower[i]
            var hit = -1
            for (k in al.indices) if (al[k].startsWith(qAlias)) { hit = k; break }
            if (hit >= 0) { tiers[1].add(tag(i, aliases[i][hit])); continue }
            if (!allowContains || tiers[2].size >= limit) continue
            if (n.contains(q)) { tiers[2].add(tag(i, "")); continue }
            for (k in al.indices) if (al[k].contains(qAlias)) { hit = k; break }
            if (hit >= 0) tiers[2].add(tag(i, aliases[i][hit]))
        }
        val out = ArrayList<DanbooruClient.Tag>(limit)
        for (t in tiers) for (x in t) { if (out.size >= limit) return out; out.add(x) }
        return out
    }

    private fun tag(i: Int, antecedent: String): DanbooruClient.Tag {
        val name = names[i]
        return DanbooruClient.Tag(name, name.replace('_', ' '), antecedent, counts[i])
    }

    /** 語の頭からの一致（"miku" が "hatsune_miku" に、"medium" が "breasts_(medium)" に当たる）。 */
    private fun wordPrefix(name: String, q: String): Boolean {
        var from = 0
        while (true) {
            val at = name.indexOf(q, from)
            if (at < 0) return false
            if (at > 0 && name[at - 1].let { it == '_' || it == '(' || it == '-' }) return true
            from = at + 1
        }
    }

    companion object {
        private val WS = Regex("\\s+")

        /** dskjal の CSV を読む。列が 4 つでない行・投稿数が数でない行は読み飛ばす。同じタグは投稿数の多いほう。 */
        fun loadCsv(reader: Reader): LocalTagIndex {
            data class Row(val name: String, val count: Long, val aliases: Array<String>)
            val byName = HashMap<String, Row>(64_000)
            reader.use { r ->
                parseCsv(r) { f ->
                    if (f.size != 4) return@parseCsv
                    val name = f[0].trim().replace(WS, "_")
                    val count = f[1].trim().toLongOrNull() ?: return@parseCsv
                    if (name.isEmpty()) return@parseCsv
                    val prev = byName[name]
                    if (prev != null && prev.count >= count) return@parseCsv
                    val lower = name.lowercase()
                    val al = f[3].split(WS).filter { it.isNotEmpty() && it.lowercase() != lower }.distinctBy { it.lowercase() }
                    byName[name] = Row(name, count, al.toTypedArray())
                }
            }
            val rows = byName.values.sortedWith(compareByDescending<Row> { it.count }.thenBy { it.name })
            return LocalTagIndex(
                Array(rows.size) { rows[it].name },
                Array(rows.size) { rows[it].name.lowercase() },
                LongArray(rows.size) { rows[it].count },
                Array(rows.size) { rows[it].aliases },
                Array(rows.size) { i -> Array(rows[i].aliases.size) { rows[i].aliases[it].lowercase() } },
            )
        }

        /** RFC 4180 の CSV（`"` で囲む、`""` は `"`、囲みの中の改行・カンマはそのまま）。1 行ごとに [onRow] を呼ぶ。 */
        internal fun parseCsv(reader: Reader, onRow: (List<String>) -> Unit) {
            val fields = ArrayList<String>(4)
            val sb = StringBuilder()
            var quoted = false
            var any = false   // この行に何か読んだ（空行は飛ばす）
            val buf = CharArray(1 shl 16)
            var pendingQuote = false   // 囲みの中で " を読んだ直後（"" か、囲みの終わりか）
            fun endRow() {
                if (any) { fields.add(sb.toString()); onRow(fields.toList()) }
                fields.clear(); sb.setLength(0); any = false
            }
            while (true) {
                val n = reader.read(buf)
                if (n < 0) break
                for (k in 0 until n) {
                    val c = buf[k]
                    if (pendingQuote) {
                        pendingQuote = false
                        if (c == '"') { sb.append('"'); continue }
                        quoted = false   // 囲みの終わり。この文字は囲みの外として下で扱う
                    }
                    if (quoted) {
                        if (c == '"') pendingQuote = true else sb.append(c)
                        continue
                    }
                    when (c) {
                        '"' -> { quoted = true; any = true }
                        ',' -> { fields.add(sb.toString()); sb.setLength(0); any = true }
                        '\n' -> endRow()
                        '\r' -> {}
                        else -> { sb.append(c); any = true }
                    }
                }
            }
            endRow()
        }
    }
}
