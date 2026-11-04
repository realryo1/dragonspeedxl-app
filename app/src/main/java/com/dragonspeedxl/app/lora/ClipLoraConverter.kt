package com.dragonspeedxl.app.lora

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * LoRA の safetensors から、テキストエンコーダ（CLIP-L と bigG）の adapter（<名前>.clip_l.bin、<名前>.clip_g.bin）を作る。
 * PC 版（loraconverter/clip_lora.py）の移植。出力は PC 版とバイト一致する（JVM のテストで確かめる）。
 *
 * CLIP の LoRA は float_only（FP16）で、adapter は、attach 点ごとの A と B の FP16 の重みを並べたものに、全 LoRA で同じ定数の部分（見出しなど）を足したもの:
 *   A = lora_down を転置した [in, 64]、B = lora_up を転置して alpha / rank を掛けた [64, out]（rank は 64 まで 0 で埋める。UNet と違い、係数 c は使わない）。
 *   どちらも [K, N] の行列として、N 256 本ごとの塊に分け、塊の中は [N の 32 本のタイル][K / 2][32][2]。塊の並びは blockOrder。
 * 位置・大きさ・定数の部分は assets の clip_layout.json（loraconverter/data と同じもの）。0 で埋めたファイルに、定数と重みを書いて作る（ひな型は要らない）。
 */
class ClipLoraConverter(layoutJson: String) {
    internal class Tensor(val node: String, val inC: Int, val outC: Int, val aOff: Long, val bOff: Long, val stem: String)
    internal class Spec(val size: Long, val tensors: List<Tensor>, val patch: List<Pair<Long, ByteArray>>)

    private val specs = LinkedHashMap<String, Spec>()

    init {
        val root = JSONObject(layoutJson)
        for (clip in CLIPS.keys) {
            val s = root.getJSONObject(clip)
            val ts = s.getJSONArray("tensors")
            val pa = s.getJSONArray("patch")
            specs[clip] = Spec(
                s.getLong("size"),
                List(ts.length()) {
                    val t = ts.getJSONObject(it)
                    Tensor(t.getString("node"), t.getInt("in"), t.getInt("out"), t.getLong("a_off"), t.getLong("b_off"), stemOf(t.getString("node")))
                },
                List(pa.length()) { val p = pa.getJSONObject(it); p.getLong("off") to hex(p.getString("hex")) },
            )
        }
    }

    /**
     * src のテキストエンコーダの重みを変換して、outDir に <name>.clip_l.bin と <name>.clip_g.bin を作る。
     * テキストエンコーダの重み（lora_te1_*、lora_te2_*）が無ければ何も作らず、空を返す。戻り値は作ったファイルの名前。
     * 途中の出力は .part に書き、全部できてから名前を変える（読み込み中のファイルを壊さない）。
     */
    fun convert(sf: Safetensors, name: String, outDir: File, progress: (String, Float) -> Unit = { _, _ -> }): List<String> {
        val made = ArrayList<String>()
        for ((ci, kv) in CLIPS.entries.withIndex()) {
            val (clip, which) = kv
            val spec = specs.getValue(clip)
            val matched = match(sf, spec, which)
            if (matched.isEmpty()) continue
            outDir.mkdirs()
            val dest = File(outDir, "$name.$clip.bin")
            val part = File(outDir, dest.name + ".part")
            RandomAccessFile(part, "rw").use { raf ->
                raf.setLength(spec.size)
                val buf = raf.channel.map(FileChannel.MapMode.READ_WRITE, 0, spec.size).also { it.order(ByteOrder.LITTLE_ENDIAN) }
                for ((off, bytes) in spec.patch) put(buf, off, bytes)
                for ((i, t) in spec.tensors.withIndex()) {
                    val base = matched[t.node]
                    val a = FloatArray(t.inC * MAX_RANK)    // [in, 64]
                    val b = FloatArray(MAX_RANK * t.outC)   // [64, out]
                    if (base != null) {
                        val down = sf.floats("$base.lora_down.weight")   // [rank, in]
                        val up = sf.floats("$base.lora_up.weight")       // [out, rank]
                        val rank = sf.tensors.getValue("$base.lora_down.weight").shape[0]
                        require(rank <= MAX_RANK) { "rank が $MAX_RANK を超える層がある: $base" }
                        require(down.size == rank * t.inC && up.size == t.outC * rank) { "$base の形が layout と合わない" }
                        val bs = (sf.floats("$base.alpha")[0].toDouble() / rank).toFloat()
                        for (r in 0 until rank) for (i2 in 0 until t.inC) a[i2 * MAX_RANK + r] = down[r * t.inC + i2]
                        for (o in 0 until t.outC) for (r in 0 until rank) b[r * t.outC + o] = up[o * rank + r] * bs
                    }
                    put(buf, t.aOff, packFp16(a, t.inC, MAX_RANK))
                    put(buf, t.bOff, packFp16(b, MAX_RANK, t.outC))
                    if (i % 16 == 0) progress("CLIP 変換中 $clip ${i + 1} / ${spec.tensors.size}", (ci + i.toFloat() / spec.tensors.size) / CLIPS.size)
                }
                buf.force()
            }
            LoraConverter.moveInto(part, dest)
            made += dest.name
        }
        return made
    }

    /** clip（"clip_l" か "clip_g"）の adapter の形（LoraMixer が使う）。 */
    internal fun spec(clip: String): Spec = specs.getValue(clip)

    /** adapter（buf）から、attach 点 t の A [in, 64] と B [64, out]（行優先の float）を読み戻す。buf の位置を動かす。 */
    internal fun readAB(buf: java.nio.ByteBuffer, t: Tensor): Pair<FloatArray, FloatArray> {
        fun read(off: Long, k: Int, n: Int): FloatArray {
            val b = ByteArray(k * n * 2)
            buf.position(off.toInt())
            buf.get(b)
            return unpackFp16(b, k, n)
        }
        return read(t.aOff, t.inC, MAX_RANK) to read(t.bOff, MAX_RANK, t.outC)
    }

    /**
     * clip の adapter を dest に作る（定数の部分と、fill が返す A [in, 64]・B [64, out]。null の attach 点は 0 のまま）。
     * 途中の出力は .part に書き、全部できてから名前を変える。
     */
    internal fun writeAdapter(clip: String, dest: File, fill: (Int, Tensor) -> Pair<FloatArray, FloatArray>?) {
        val spec = specs.getValue(clip)
        val part = File(dest.parentFile, dest.name + ".part")
        RandomAccessFile(part, "rw").use { raf ->
            raf.setLength(0)
            raf.setLength(spec.size)
            val buf = raf.channel.map(FileChannel.MapMode.READ_WRITE, 0, spec.size).also { it.order(ByteOrder.LITTLE_ENDIAN) }
            for ((off, bytes) in spec.patch) put(buf, off, bytes)
            for ((i, t) in spec.tensors.withIndex()) {
                val (a, b) = fill(i, t) ?: continue
                put(buf, t.aOff, packFp16(a, t.inC, MAX_RANK))
                put(buf, t.bOff, packFp16(b, MAX_RANK, t.outC))
            }
            buf.force()
        }
        LoraConverter.moveInto(part, dest)
    }

    /** attach 点 → LoRA のキーの基底（lora_te1_… の .alpha を除いたもの）。無い層は入らない。グラフに出ない層（CLIP-L の最終層）は読み飛ばす。 */
    private fun match(sf: Safetensors, spec: Spec, which: String): Map<String, String> {
        val byStem = spec.tensors.associate { it.stem to it.node }
        val last = byStem.keys.mapNotNull { LAYER.matchEntire(it)?.groupValues?.get(1)?.toInt() }.max()
        val prefix = "lora_${which}_"
        val out = HashMap<String, String>()
        val unmatched = ArrayList<String>()
        for (base in sf.tensors.keys.filter { it.endsWith(".alpha") && it.startsWith(prefix) }.map { it.removeSuffix(".alpha") }.sorted()) {
            val stem = base.removePrefix(prefix).removePrefix("text_model_")
            val node = byStem[stem]
            if (node == null) {
                val m = LAYER.matchEntire(stem)
                if (m != null && m.groupValues[1].toInt() > last) continue
                unmatched += base
            } else out[node] = base
        }
        require(unmatched.isEmpty()) { "対応しない $which の LoRA のキーが ${unmatched.size} 個ある。例: ${unmatched.take(3)}" }
        return out
    }

    companion object {
        const val MAX_RANK = 64
        /** adapter の名前 → LoRA のキーの接頭辞（lora_te1_ / lora_te2_） */
        val CLIPS = linkedMapOf("clip_l" to "te1", "clip_g" to "te2")
        private val LAYER = Regex("encoder_layers_(\\d+)_.*")

        /** attach 点（ONNX のノード名）→ LoRA のキーの名前（text_model_ を除いたもの）。/m/text_model/encoder/layers.0/mlp/fc1/conv/Conv → encoder_layers_0_mlp_fc1 */
        fun stemOf(node: String): String =
            node.removePrefix("/m/").removeSuffix("/conv/Conv").replace('/', '_').replace('.', '_').removePrefix("text_model_")

        /** 出力 256 本の塊が、ファイルに並んでいる順（論理番号）。塊が 10 個以下でも入れ替わる（fp16。UNet の int8 とは違う）。 */
        fun blockOrder(n: Int): List<Int> {
            val seq = ArrayList<Int>()
            seq += 0
            for (i in 4 until n) seq += i
            for ((v, at) in listOf(1 to 7, 2 to 18, 3 to 29)) {
                if (v >= n) break
                seq.add(minOf(at, seq.size), v)
            }
            return seq
        }

        /** [K, N]（行優先の float）を adapter の並びの FP16（リトルエンディアン）にする。K は 2 の倍数、N は 32 の倍数。 */
        fun packFp16(m: FloatArray, k: Int, n: Int): ByteArray {
            require(k % 2 == 0 && n % 32 == 0) { "形が合わない: $k x $n" }
            val nChunk = (n + 255) / 256
            val width = { l: Int -> minOf(256, n - l * 256) }
            val dest = IntArray(nChunk)
            var acc = 0
            for (l in blockOrder(nChunk)) { dest[l] = acc; acc += k * width(l) * 2 }
            val out = ByteArray(k * n * 2)
            for (l in 0 until nChunk) {
                val c0 = l * 256
                var p = dest[l]
                for (t in 0 until width(l) / 32) for (kp in 0 until k / 2) for (n32 in 0 until 32) for (k2 in 0 until 2) {
                    val h = floatToHalf(m[(2 * kp + k2) * n + c0 + t * 32 + n32])
                    out[p++] = h.toByte(); out[p++] = (h shr 8).toByte()
                }
            }
            return out
        }

        /** packFp16 の逆（adapter の並びの FP16 から [K, N] の float に戻す）。 */
        fun unpackFp16(p: ByteArray, k: Int, n: Int): FloatArray {
            require(k % 2 == 0 && n % 32 == 0 && p.size == k * n * 2) { "形が合わない: $k x $n" }
            val nChunk = (n + 255) / 256
            val width = { l: Int -> minOf(256, n - l * 256) }
            val src = IntArray(nChunk)
            var acc = 0
            for (l in blockOrder(nChunk)) { src[l] = acc; acc += k * width(l) * 2 }
            val m = FloatArray(k * n)
            for (l in 0 until nChunk) {
                val c0 = l * 256
                var q = src[l]
                for (t in 0 until width(l) / 32) for (kp in 0 until k / 2) for (n32 in 0 until 32) for (k2 in 0 until 2) {
                    val h = (p[q].toInt() and 0xFF) or ((p[q + 1].toInt() and 0xFF) shl 8)
                    q += 2
                    m[(2 * kp + k2) * n + c0 + t * 32 + n32] = Safetensors.halfToFloat(h)
                }
            }
            return m
        }

        /** float32 → float16 のビット列（最近接偶数への丸め。numpy の astype(float16) と同じ。大きすぎる値は inf）。 */
        fun floatToHalf(f: Float): Int {
            val bits = java.lang.Float.floatToRawIntBits(f)
            val sign = (bits ushr 16) and 0x8000
            val exp = (bits ushr 23) and 0xFF
            var mant = bits and 0x7FFFFF
            if (exp == 0xFF) return sign or 0x7C00 or (if (mant != 0) 0x200 else 0)   // inf / nan
            val e = exp - 127 + 15
            if (e >= 0x1F) return sign or 0x7C00
            if (e <= 0) {   // half の非正規数（または 0）
                if (e < -10) return sign
                mant = mant or 0x800000
                val shift = 14 - e
                var r = mant ushr shift
                val rem = mant and ((1 shl shift) - 1)
                val half = 1 shl (shift - 1)
                if (rem > half || (rem == half && (r and 1) == 1)) r++
                return sign or r
            }
            var h = (e shl 10) or (mant ushr 13)
            val rem = mant and 0x1FFF
            if (rem > 0x1000 || (rem == 0x1000 && (h and 1) == 1)) h++   // 桁上がりで指数が上がっても正しい（最大なら inf）
            return sign or h
        }

        private fun put(buf: MappedByteBuffer, off: Long, bytes: ByteArray) {
            buf.position(off.toInt())
            buf.put(bytes)
        }

        private fun hex(s: String): ByteArray = ByteArray(s.length / 2) { ((Character.digit(s[2 * it], 16) shl 4) or Character.digit(s[2 * it + 1], 16)).toByte() }
    }
}
