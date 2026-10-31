package com.dragonspeedxl.app.lora

import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile

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
        val m = LinkedHashMap<String, Info>()
        for (key in header.keys()) {
            if (key == "__metadata__") continue
            val spec = header.getJSONObject(key)
            val sh = spec.getJSONArray("shape")
            val off = spec.getJSONArray("data_offsets")
            m[key] = Info(spec.getString("dtype"), IntArray(sh.length()) { sh.getInt(it) }, off.getLong(0), off.getLong(1))
        }
        tensors = m
    }

    fun floats(key: String): FloatArray {
        val info = tensors[key] ?: error("safetensors に $key が無い")
        val size = (info.end - info.start).toInt()
        val raw = ByteArray(size)
        raf.seek(dataStart + info.start)
        raf.readFully(raw)
        return when (info.dtype) {
            "F32" -> FloatArray(size / 4) { java.lang.Float.intBitsToFloat(le32(raw, it * 4)) }
            "F16" -> FloatArray(size / 2) { halfToFloat(le16(raw, it * 2)) }
            "BF16" -> FloatArray(size / 2) { java.lang.Float.intBitsToFloat(le16(raw, it * 2) shl 16) }
            else -> error("$key の型 ${info.dtype} は読めない")
        }
    }

    override fun close() = raf.close()

    private fun le16(b: ByteArray, i: Int) = (b[i].toInt() and 0xff) or ((b[i + 1].toInt() and 0xff) shl 8)
    private fun le32(b: ByteArray, i: Int) = le16(b, i) or (le16(b, i + 2) shl 16)

    companion object {
        /** IEEE 半精度 → 単精度（厳密。numpy の float16 → float32 と同じ）。 */
        fun halfToFloat(h: Int): Float {
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
