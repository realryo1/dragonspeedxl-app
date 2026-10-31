package com.dragonspeedxl.app.lora

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile

/**
 * Kotlin 版の変換（4 グラフの super adapter）が、PC 版（loraconverter/lora_to_npu.py の <名前>.4g.bin）と全バイト一致することを確かめる。
 * mw25 の PC 版の出力は、Colab（cbg）で作った 4 グラフの super adapter とも全バイト一致する（2026-10-18）。
 * 入力（Model フォルダの safetensors、sdxl-npu/out/lora_super4 の default の super adapter、sdxl-npu/out/lora_conv4 の PC 版の出力）は
 * リポジトリに入っていないので、無ければ飛ばす。default の super adapter は loraconverter/super_adapter.py build で作る（端末の models/ に置くものと同じ）。
 */
class LoraConverterTest {
    private val root: File = generateSequence(File(System.getProperty("user.dir")).absoluteFile) { it.parentFile }.first { File(it, "loraconverter").isDirectory }
    private val data = File(root, "loraconverter/data")
    private val converter by lazy {
        LoraConverter(
            File(data, "layout.json").readText(), File(data, "attach_map.json").readText(), File(data, "super_4g_windows.json").readText(),
            File(data, "clip_layout.json").readText(),
        )
    }

    private fun check(name: String, safetensors: String, ref: File) {
        val src = File(root, "Model/$safetensors")
        val modelDir = File(root, "sdxl-npu/out/lora_super4")
        assumeTrue(src.isFile && ref.isFile && File(modelDir, LoraConverter.DEFAULT_ADAPTER).isFile)
        val out = File(root, "sdxl-npu/out/lora_local/kotlin_$name").also { it.deleteRecursively() }
        val r = converter.convert(src, name, "", modelDir, out)
        assertEquals(0, r.missing)
        assertEquals("$name${LoraConverter.SUFFIX} と ${ref.name} の最初の差の位置（-1 なら一致）", -1L, firstDiff(File(out, name + LoraConverter.SUFFIX), ref))
        assertTrue(File(out, "$name.json").isFile)
    }

    /** テキストエンコーダ（CLIP）の adapter が、PC 版（実機で動作確認済みの cure の adapter と全バイト一致）と全バイト一致する。 */
    @Test fun clipAdaptersMatchPc() {
        val src = File(root, "Model/curearcanashadow.safetensors")
        val ref = File(root, "sdxl-npu/out/clip_conv/curetest")
        val modelDir = File(root, "sdxl-npu/out/lora_super4")
        assumeTrue(src.isFile && File(ref, "curetest.clip_l.bin").isFile && File(ref, "curetest.clip_g.bin").isFile && File(modelDir, LoraConverter.DEFAULT_ADAPTER).isFile)
        val out = File(root, "sdxl-npu/out/lora_local/kotlin_clip").also { it.deleteRecursively() }
        val r = converter.convert(src, "curetest", "", modelDir, out)
        assertEquals(listOf("curetest.clip_l.bin", "curetest.clip_g.bin"), r.clipFiles)
        for (n in r.clipFiles) assertEquals("$n の最初の差の位置（-1 なら一致）", -1L, firstDiff(File(out, n), File(ref, n)))
    }

    @Test fun halfRoundingMatchesNumpy() {
        // numpy の astype(float16) と同じ丸め（最近接偶数、非正規数、桁上がり、大きすぎる値は inf）
        fun h(f: Float) = ClipLoraConverter.floatToHalf(f)
        assertEquals(0x3c00, h(1.0f)); assertEquals(0xc000, h(-2.0f)); assertEquals(0, h(0f)); assertEquals(0x8000, h(-0f))
        assertEquals(0x0400, h(6.1035156e-5f)); assertEquals(0x0001, h(5.9604645e-8f)); assertEquals(0, h(2.0e-8f))
        assertEquals(0x7bff, h(65504f)); assertEquals(0x7c00, h(65520f)); assertEquals(0x7c00, h(1e9f)); assertEquals(0xfc00, h(-1e9f))
        assertEquals(0x3c00, h(1.00048828125f)); assertEquals(0x3c02, h(1.00146484375f))   // ちょうど中間は偶数へ（1 + 2^-11 → 1、1 + 3 * 2^-11 → 1 + 2^-9）
    }

    @Test fun mmmMatchesPc() = check("mmm", "MMM2013XLNAI.safetensors", File(root, "sdxl-npu/out/lora_conv4/mmmc/mmmc.4g.bin"))
    @Test fun mw25MatchesColab() = check("mw25", "mw25.safetensors", File(root, "sdxl-npu/out/lora_4g/unet_720x1280-unet_576x1024-unet_1280x720-unet_1024x576_mw25.bin"))

    @Test fun parsesCivitaiUrls() {
        assertEquals(1493187L to 1691140L, CivitaiClient.parseUrl("https://civitai.com/models/1493187/x?modelVersionId=1691140"))
        assertEquals(1493187L to null, CivitaiClient.parseUrl("https://civitai.com/models/1493187/x"))
        assertEquals(null to 1690879L, CivitaiClient.parseUrl("https://civitai.com/api/download/models/1690879"))
    }

    @Test fun halfFloatIsExact() {
        assertEquals(1.0f, Safetensors.halfToFloat(0x3c00), 0f)
        assertEquals(-2.0f, Safetensors.halfToFloat(0xc000), 0f)
        assertEquals(6.1035156e-5f, Safetensors.halfToFloat(0x0400), 0f)    // 最小の正規数
        assertEquals(5.9604645e-8f, Safetensors.halfToFloat(0x0001), 0f)    // 最小の非正規数
    }

    private fun firstDiff(a: File, b: File): Long {
        if (a.length() != b.length()) return -2
        RandomAccessFile(a, "r").use { fa -> RandomAccessFile(b, "r").use { fb ->
            val x = ByteArray(1 shl 20); val y = ByteArray(1 shl 20)
            var pos = 0L
            while (pos < fa.length()) {
                val n = minOf(x.size.toLong(), fa.length() - pos).toInt()
                fa.readFully(x, 0, n); fb.readFully(y, 0, n)
                for (i in 0 until n) if (x[i] != y[i]) return pos + i
                pos += n
            }
        } }
        return -1
    }
}
