package com.dragonspeedxl.app.lora

import java.io.File

/** Civitai の URL から LoRA を取って adapter に変換し、loras/ に置くまでをまとめる。 */
object LoraImporter {
    class Outcome(val name: String, val activation: String, val result: LoraConverter.Result)

    /**
     * progress(メッセージ, 0〜1)。ダウンロードは 0〜0.5、変換は 0.5〜1。
     * 取った safetensors は work に置いて、変換できたら消す（端末の空きを使い続けない）。
     */
    fun fromCivitai(
        url: String, token: String, converter: LoraConverter, modelDir: File, lorasDir: File, work: File,
        progress: (String, Float) -> Unit,
    ): Outcome {
        progress("Civitai に問い合わせ中", 0f)
        val client = CivitaiClient(token.trim())
        val v = client.fetchVersion(url)
        val name = LoraConverter.safeName(File(v.fileName).nameWithoutExtension)
        LoraConverter.checkName(name)
        LoraConverter.requireNew(lorasDir, name)   // ダウンロードの前に、同じ名前が既にあるか確かめる
        val activation = LoraConverter.cleanActivation(v.trainedWords.joinToString(", "))
        val src = File(work, "$name.safetensors")
        work.mkdirs()
        try {
            client.download(v, src) { done, total ->
                val f = if (total > 0) done.toFloat() / total else 0f
                progress("ダウンロード中 ${done shr 20} / ${total shr 20} MB", 0.5f * f)
            }
            val r = converter.convert(src, name, activation, modelDir, lorasDir) { msg, f -> progress(msg, 0.5f + 0.5f * f) }
            return Outcome(name, activation, r)
        } finally {
            src.delete()
        }
    }

    /**
     * 端末から選んだ safetensors（src）を変換する。名前（英数字）と activation text は利用者が入力したもの。
     * activation text は空でもよい。`\(` は `(` に直す（Civitai と同じ）。src の後始末は呼び出し側。
     */
    fun fromFile(
        src: File, name: String, activation: String, converter: LoraConverter, modelDir: File, lorasDir: File,
        progress: (String, Float) -> Unit,
    ): Outcome {
        LoraConverter.checkName(name)
        val act = LoraConverter.cleanActivation(activation.trim())
        val r = converter.convert(src, name, act, modelDir, lorasDir, progress)
        return Outcome(name, act, r)
    }
}
