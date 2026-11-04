package com.dragonspeedxl.app.prompt

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class LocalTagIndexTest {
    // dskjal の CSV と同じ形（タグ名, 投稿数, タググループ, 別名）。囲み・"" ・囲みの中の改行も入れる
    private val csv = listOf(
        "1girl,7604847,character_count,女の子 女性 girl",
        "sitting,1194933,verbs_and_gerunds posture,座る 座り",
        "hatsune_miku,132011,,初音ミク miku ミク",
        "kagamine_rin,24170,,鏡音リン rinkagamine",
        "miku_(another),500,,",
        "\"don't_say_\"\"lazy\"\"\",1089,,\"Don't_say_\"\"lazy\"\"\"",
        "breasts_(medium),900,,\"中\r\nくらい\"",
        "shirayuki_towa,305,artist,白雪とわ",
        "broken_row,abc,,",
        "",
        "sitting,10,,重複は多いほう",
    ).joinToString("\r\n")

    private val index = LocalTagIndex.loadCsv(csv.reader())

    @Test fun parsesQuotesAndDuplicates() {
        assertEquals(8, index.size)
        assertEquals("don't_say_\"lazy\"", index.search("don't").first().name)
        assertEquals(1194933L, index.search("sitting").first().postCount)
    }

    @Test fun exactThenPrefixByCount() {
        val r = index.search("miku")
        // 語の頭の一致（hatsune_miku）は前方一致（miku_(another)）と同じ段で、投稿数の多い順
        assertEquals(listOf("hatsune_miku", "miku_(another)"), r.map { it.name })
        assertEquals("", r[0].antecedent)
    }

    @Test fun spacesBecomeUnderscores() {
        assertEquals("kagamine_rin", index.search("kagamine r").first().name)
        assertEquals("hatsune miku", index.search("hatsune_miku").first().label)
    }

    @Test fun japaneseAliasShowsAntecedent() {
        val r = index.search("初音")
        assertEquals("hatsune_miku", r.first().name)
        assertEquals("初音ミク", r.first().antecedent)
        // 途中に含む（2 文字以上）
        val c = index.search("ガミネ")
        assertTrue(c.isEmpty())
        assertEquals("kagamine_rin", index.search("音リン").first().name)
        assertEquals("shirayuki_towa", index.search("とわ").first().name)
    }

    @Test fun wordPrefixAfterParen() {
        assertEquals("breasts_(medium)", index.search("medium").first().name)
    }

    @Test fun limit() {
        // sitting, don't_say_"lazy"（語の頭）, shirayuki_towa
        assertEquals(listOf("sitting", "don't_say_\"lazy\"", "shirayuki_towa"), index.search("s").map { it.name })
        assertEquals(listOf("sitting"), index.search("s", limit = 1).map { it.name })
    }

    @Test fun realDskjalCsvIfGiven() {
        // 手元で実データを試すとき: DSXL_TAG_CSV=<tags-YYYY-MM-DD.csv> を渡す（無ければ飛ばす）
        val path = System.getenv("DSXL_TAG_CSV") ?: return
        val t0 = System.nanoTime()
        val real = LocalTagIndex.loadCsv(File(path).reader(Charsets.UTF_8))
        val t1 = System.nanoTime()
        for (q in listOf("hatsune miku", "初音ミク", "kagamine rin", "sitting", "shirayuki towa", "rin")) {
            val s0 = System.nanoTime()
            val r = real.search(q)
            println("$q -> ${r.take(5).map { if (it.antecedent.isEmpty()) it.name else "${it.antecedent}→${it.name}" }} (${(System.nanoTime() - s0) / 1000}us)")
        }
        println("tags=${real.size} load=${(t1 - t0) / 1_000_000}ms")
        assertTrue(real.size > 40_000)
        assertEquals("hatsune_miku", real.search("初音ミク").first().name)
    }
}
