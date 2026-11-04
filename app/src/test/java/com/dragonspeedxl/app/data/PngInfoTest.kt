package com.dragonspeedxl.app.data

import com.dragonspeedxl.app.engine.Orientation
import com.dragonspeedxl.app.lora.LoraUse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.util.zip.CRC32
import java.util.zip.Deflater
import java.util.zip.Inflater

/**
 * 生成情報（A1111 互換の parameters）の文面と、PNG へのチャンクの埋め込みを確かめる。
 * 画像は 4×3 の小さなものを組み立てるだけで、生成はしない（プロンプトは 2 つの文だけ使う）。
 * Android の unit test では ImageIO（java.awt）が使えないので、PNG は手で組む。
 */
class PngInfoTest {
    private fun item(
        prompt: String = "hatsune miku, shirayuki towa",
        hires: Boolean = true,
        loras: List<LoraUse> = emptyList(),
        variant: String = "",
        orientation: Orientation = Orientation.PORTRAIT,
    ) = HistoryItem(
        id = "1", prompt = prompt, seed = 11L, orientation = orientation, hires = hires, totalMillis = 0L, createdAt = 0L,
        baseSteps = 2, hiresSteps = if (hires) 4 else 0, denoise = 0.39, variant = variant, loras = loras,
    )

    @Test
    fun portraitHiresWithoutLora() {
        val expected = "hatsune miku, shirayuki towa <lora:NoobV065sHyperDmd:1>\n" +
            "Negative prompt: \n" +
            "Steps: 2, Sampler: LCM, Schedule type: SGM Uniform, CFG scale: 1, Seed: 11, Size: 720x1280, " +
            "Model: waiIllustriousSDXL_v160, Denoising strength: 0.39, Hires CFG Scale: 1, Hires upscale: 1.5, " +
            "Hires steps: 4, Hires upscaler: Latent (nearest-exact), Version: dragonspeedXL"
        assertEquals(expected, PngInfo.infotext(item()))
    }

    @Test
    fun landscapeBaseOnlyWithLoraAndDraft() {
        val expected = "hatsune miku <lora:NoobV065sHyperDmd:1> <lora:mmm:1.2>\n" +
            "Negative prompt: \n" +
            "Steps: 2, Sampler: LCM, Schedule type: SGM Uniform, CFG scale: 1, Seed: 11, Size: 1280x720, " +
            "Model: waiIllustriousSDXL_v160, Decoder: TAESD, Version: dragonspeedXL"
        assertEquals(expected, PngInfo.infotext(item(
            prompt = "hatsune miku", hires = false, loras = listOf(LoraUse("mmm", 1.2f)),
            variant = "_draft", orientation = Orientation.LANDSCAPE,
        )))
    }

    @Test
    fun multipleLorasInOrder() {
        val text = PngInfo.infotext(item(prompt = "kagamine rin, sitting", loras = listOf(LoraUse("mw25", 0.8f), LoraUse("mmm", 1f))))
        assertTrue(text, text.startsWith("kagamine rin, sitting <lora:NoobV065sHyperDmd:1> <lora:mw25:0.8> <lora:mmm:1>\n"))
    }

    @Test
    fun embedsParametersChunkAfterIhdr() {
        val text = PngInfo.infotext(item())
        val png = PngInfo.embed(solidPng(4, 3), text)

        val chunks = chunks(png)
        // IHDR が先頭、そのすぐ後に tEXt（PNG の仕様）。IEND で終わる
        assertEquals(listOf("IHDR", "tEXt"), chunks.take(2).map { it.type })
        assertEquals("IEND", chunks.last().type)
        val t = chunks[1]
        val nul = t.data.indexOf(0)
        assertEquals("parameters", String(t.data, 0, nul, Charsets.ISO_8859_1))
        assertEquals(text, String(t.data, nul + 1, t.data.size - nul - 1, Charsets.ISO_8859_1))

        // 埋め込んでも、画像データ（IDAT）は元のまま。展開すると、4×3 の RGB（各行の先頭にフィルタ 0 の 1 バイト）になる
        val idat = chunks.first { it.type == "IDAT" }.data
        val inflater = Inflater().apply { setInput(idat) }
        val pixels = ByteArray(3 * (1 + 4 * 3)).also { inflater.inflate(it) }
        inflater.end()
        assertEquals(0, pixels[0].toInt())
    }

    @Test
    fun japanesePromptUsesUtf8InternationalText() {
        val text = PngInfo.infotext(item(prompt = "初音ミク"))
        val png = PngInfo.embed(solidPng(4, 3), text)

        val t = chunks(png).first { it.type == "iTXt" }
        val kw = t.data.indexOf(0)
        assertEquals("parameters", String(t.data, 0, kw, Charsets.ISO_8859_1))
        // 圧縮フラグ 0、圧縮方式 0、言語（空）、翻訳キーワード（空）の後に本文（UTF-8）
        assertEquals(0, t.data[kw + 1].toInt())
        assertEquals(0, t.data[kw + 2].toInt())
        val body = t.data.copyOfRange(kw + 5, t.data.size)
        assertEquals(text, String(body, Charsets.UTF_8))
    }

    /** 黒一色の RGB の PNG（IHDR、IDAT、IEND。IHDR の直後に IDAT が来る、普通の形）。 */
    private fun solidPng(w: Int, h: Int): ByteArray {
        val raw = ByteArrayOutputStream()
        repeat(h) { raw.write(0); repeat(w * 3) { raw.write(0) } }   // 各行は、フィルタ 0（None）の 1 バイトと RGB の画素
        val deflater = Deflater().apply { setInput(raw.toByteArray()); finish() }
        val z = ByteArrayOutputStream()
        val buf = ByteArray(1024)
        while (!deflater.finished()) z.write(buf, 0, deflater.deflate(buf))
        deflater.end()
        val ihdr = ByteArrayOutputStream().also { DataOutputStream(it).apply { writeInt(w); writeInt(h); write(byteArrayOf(8, 2, 0, 0, 0)) } }
        val out = ByteArrayOutputStream()
        out.write(byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))
        out.write(rawChunk("IHDR", ihdr.toByteArray()))
        out.write(rawChunk("IDAT", z.toByteArray()))
        out.write(rawChunk("IEND", ByteArray(0)))
        return out.toByteArray()
    }

    private fun rawChunk(type: String, data: ByteArray): ByteArray {
        val typeBytes = type.toByteArray(Charsets.US_ASCII)
        val out = ByteArrayOutputStream()
        DataOutputStream(out).use { d ->
            d.writeInt(data.size)
            d.write(typeBytes)
            d.write(data)
            d.writeInt(CRC32().apply { update(typeBytes); update(data) }.value.toInt())
        }
        return out.toByteArray()
    }

    private data class Chunk(val type: String, val data: ByteArray)

    /** チャンクを読み、各 CRC が合うことを確かめる。 */
    private fun chunks(png: ByteArray): List<Chunk> {
        val out = mutableListOf<Chunk>()
        val din = DataInputStream(ByteArrayInputStream(png, 8, png.size - 8))
        while (din.available() > 0) {
            val len = din.readInt()
            val type = ByteArray(4).also { din.readFully(it) }
            val data = ByteArray(len).also { din.readFully(it) }
            val crc = din.readInt()
            assertEquals(CRC32().apply { update(type); update(data) }.value.toInt(), crc)
            out += Chunk(String(type, Charsets.US_ASCII), data)
        }
        assertTrue(out.isNotEmpty())
        return out
    }
}
