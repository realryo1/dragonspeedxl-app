package com.dragonspeedxl.app.lora

import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** safetensors の読み取り（必要なテンソルだけ float32 にする。F32 / F16 / BF16）。 */
class Safetensors(file: File) : Closeable {
    class Info(val dtype: String, val shape: IntArray, val start: Long, val end: Long)

    private val raf = RandomAccessFile(file, "r")
    private val dataStart: Long
    val tensors: Map<String, Info>

    init {
        val n = ByteArray(8).also { raf.readFully(it) }.let { b -> (0 until 8).fold(0L) { acc, i -> acc or ((b[i].toLong() and 0xff) shl (8 * i)) } }
        require(n in 2..(100L shl 20)) { "safetensors のヘッダーが読めない" }
        val header = JSONObject(String(ByteArray(n.toInt()).also { raf.readFully(it) }, Charsets.UTF_8))
        dataStart = 8 + n
        val dataLength = raf.length() - dataStart
        val m = LinkedHashMap<String, Info>()
        for (key in header.keys()) {
            if (key == "__metadata__") continue
            val spec = header.getJSONObject(key)
            val sh = spec.getJSONArray("shape")
            val off = spec.getJSONArray("data_offsets")
            val info = Info(spec.getString("dtype"), IntArray(sh.length()) { sh.getInt(it) }, off.getLong(0), off.getLong(1))
            checkInfo(key, info, dataLength)
            m[key] = info
        }
        tensors = m
    }

    /** 位置と大きさが、ファイルの中に収まり、形と型の大きさが合うかを確かめる（壊れた・悪意のあるファイルで、範囲外を読まないように）。 */
    private fun checkInfo(key: String, info: Info, dataLength: Long) {
        require(info.start in 0L..info.end && info.end <= dataLength) { "safetensors の $key の位置が不正" }
        val size = info.end - info.start
        require(size <= Int.MAX_VALUE) { "safetensors の $key が大きすぎる" }
        require(info.shape.all { it >= 0 }) { "safetensors の $key の形が不正" }
        val elem = ELEM_SIZE[info.dtype] ?: return   // 未知の型は floats で弾く
        val count = info.shape.fold(1L) { acc, d -> acc * d }
        require(count * elem == size) { "safetensors の $key の大きさが形と合わない" }
    }

    fun floats(key: String): FloatArray {
        val info = tensors[key] ?: error("safetensors に $key が無い")
        val size = (info.end - info.start).toInt()
        val raw = ByteArray(size)
        raf.seek(dataStart + info.start)
        raf.readFully(raw)
        // リトルエンディアンの並びとしてまとめて読む（1 要素ずつバイトを組み立てない）
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        return when (info.dtype) {
            "F32" -> FloatArray(size / 4).also { bb.asFloatBuffer().get(it) }
            "F16" -> bb.asShortBuffer().let { s -> FloatArray(size / 2) { HALF_TO_FLOAT[s.get(it).toInt() and 0xffff] } }
            "BF16" -> bb.asShortBuffer().let { s -> FloatArray(size / 2) { java.lang.Float.intBitsToFloat((s.get(it).toInt() and 0xffff) shl 16) } }
            else -> error("$key の型 ${info.dtype} は読めない")
        }
    }

    override fun close() = raf.close()

    companion object {
        /** 読める型の 1 要素の大きさ（バイト） */
        private val ELEM_SIZE = mapOf("F32" to 4L, "F16" to 2L, "BF16" to 2L)

        /** 半精度の全 65,536 通りの値（halfToFloat の表。LoRA の FP16 の重みを読むたびに計算し直さない） */
        private val HALF_TO_FLOAT by lazy { FloatArray(1 shl 16) { computeHalf(it) } }

        /** IEEE 半精度（下位 16 ビット）→ 単精度（厳密。numpy の float16 → float32 と同じ）。 */
        fun halfToFloat(h: Int): Float = HALF_TO_FLOAT[h and 0xffff]

        private fun computeHalf(h: Int): Float {
            val sign = (h shr 15) and 1
            val exp = (h shr 10) and 0x1f
            val man = h and 0x3ff
            val bits = when {
                exp == 0 && man == 0 -> sign shl 31
                exp == 0 -> {   // 非正規数: 正規化する
                    var e = 127 - 15 + 1
                    var m = man
                    while (m and 0x400 == 0) { m = m shl 1; e-- }
                    (sign shl 31) or (e shl 23) or ((m and 0x3ff) shl 13)
                }
                exp == 0x1f -> (sign shl 31) or (0xff shl 23) or (man shl 13)
                else -> (sign shl 31) or ((exp - 15 + 127) shl 23) or (man shl 13)
            }
            return java.lang.Float.intBitsToFloat(bits)
        }
    }
}
