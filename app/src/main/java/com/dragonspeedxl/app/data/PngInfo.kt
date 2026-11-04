package com.dragonspeedxl.app.data

import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.util.Locale
import java.util.zip.CRC32

/**
 * A1111（Stable Diffusion WebUI / reforge）と互換の生成情報（infotext）を、PNG に埋め込む。
 * 形式は、A1111 の出力と同じ「parameters」という名前のテキストチャンク。A1111 の PNG Info や「send to txt2img」で読める。
 * 本文（prompt）の後に `Negative prompt:` の行、その下に `Steps: 2, Sampler: LCM, …` の行が続く。
 * 見本は sample/a1111sample.png（reforge で作った画像）。項目の並びも同じにしている。
 */
object PngInfo {
    const val KEY = "parameters"

    private const val CHECKPOINT = "waiIllustriousSDXL_v160"
    /** UNet に融合済みの固定 LoRA（ComfyUI の仕様）。A1111 で同じ絵を再現するため、本文の <lora:…> に入れる。 */
    private const val FIXED_LORA = "NoobV065sHyperDmd"
    private const val HIRES_UPSCALE = 1.5
    private const val DRAFT_VARIANT = "_draft"

    /** A1111 の infotext を作る。履歴の項目（HistoryItem）だけから作るので、あとから書き出すときも同じ内容になる。 */
    fun infotext(item: HistoryItem): String {
        // LoRA の強さは A1111 の weight（w）。係数 c は、端末の内部の alpha にだけ掛ける値なので入れない。
        val loras = buildString {
            append(" <lora:$FIXED_LORA:1>")
            for (u in item.loras) append(" <lora:${u.name}:${num(u.strength.toDouble())}>")   // 多重適用は、使った順に並べる（A1111 も足し算）
        }
        val params = buildList {
            add("Steps: ${item.baseSteps}")
            add("Sampler: LCM")
            add("Schedule type: SGM Uniform")
            add("CFG scale: 1")
            add("Seed: ${item.seed}")
            add("Size: ${sizeOf(item)}")
            add("Model: $CHECKPOINT")
            if (item.hires) {
                add("Denoising strength: ${num(item.denoise)}")
                add("Hires CFG Scale: 1")
                add("Hires upscale: ${num(HIRES_UPSCALE)}")
                add("Hires steps: ${item.hiresSteps}")
                add("Hires upscaler: Latent (nearest-exact)")
            }
            if (item.variant == DRAFT_VARIANT) add("Decoder: TAESD")   // 速度優先の最終画像は、VAE ではなく TAESD でデコードした
            add("Version: dragonspeedXL")
        }
        return "${item.prompt}$loras\nNegative prompt: \n${params.joinToString(", ")}"
    }

    /**
     * PNG（バイト列）に「parameters」のテキストチャンクを入れて返す。IHDR の直後に入れる（PNG の仕様で、IHDR が先頭に来る）。
     * 全部 Latin-1 に収まるなら tEXt、収まらない（日本語の prompt など）なら iTXt（UTF-8）にする。
     */
    fun embed(png: ByteArray, text: String): ByteArray {
        val ihdrEnd = 8 + 12 + 13   // シグネチャ 8 バイト ＋ IHDR のチャンク（長さ・型・本体 13・CRC）
        require(png.size > ihdrEnd && String(png, 12, 4, Charsets.US_ASCII) == "IHDR") { "PNG ではありません" }
        val chunk = if (text.all { it.code < 256 }) textChunk(text) else internationalTextChunk(text)
        return png.copyOfRange(0, ihdrEnd) + chunk + png.copyOfRange(ihdrEnd, png.size)
    }

    /** 1 行の数値（A1111 と同じ表記。小数点は "." で、末尾の 0 を落とす）。 */
    private fun num(v: Double): String = String.format(Locale.US, "%.2f", v).trimEnd('0').trimEnd('.')

    /** A1111 の Size は、Hires の前の大きさ（Base の寸法）。 */
    private fun sizeOf(item: HistoryItem): String = "${item.orientation.baseW}x${item.orientation.baseH}"

    private fun textChunk(text: String): ByteArray {
        val data = KEY.toByteArray(Charsets.ISO_8859_1) + byteArrayOf(0) + text.toByteArray(Charsets.ISO_8859_1)
        return chunk("tEXt", data)
    }

    private fun internationalTextChunk(text: String): ByteArray {
        // キーワード、0、圧縮フラグ（0 = なし）、圧縮方式（0）、言語（空）、0、翻訳キーワード（空）、0、本文（UTF-8）
        val data = KEY.toByteArray(Charsets.ISO_8859_1) + byteArrayOf(0, 0, 0, 0, 0) + text.toByteArray(Charsets.UTF_8)
        return chunk("iTXt", data)
    }

    /** チャンク: 長さ（4 バイト）、型（4 バイト）、本体、CRC（型と本体の CRC-32）。 */
    private fun chunk(type: String, data: ByteArray): ByteArray {
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
}
