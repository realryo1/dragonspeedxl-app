package com.dragonspeedxl.app.lora

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * LoRA の多重適用（LoraMixer）の確かめ。
 * - 数値の部分（rank の切り詰め、固有値分解、並びの逆変換）は、乱数の行列で確かめる（入力のファイルは要らない）。
 * - 実物の LoRA（Model フォルダの safetensors と、sdxl-npu/out/lora_super4 の default の super adapter）があれば、
 *   変換してから合成し、合成した adapter の ΔW が、各 LoRA の ΔW の和（強さを掛けたもの）に近いことを確かめる。無ければ飛ばす。
 */
class LoraMixerTest {
    private val root: File = generateSequence(File(System.getProperty("user.dir")).absoluteFile) { it.parentFile }.first { File(it, "loraconverter").isDirectory }
    private val data = File(root, "loraconverter/data")
    private val converter by lazy {
        LoraConverter(
            File(data, "layout.json").readText(), File(data, "attach_map.json").readText(), File(data, "super_4g_windows.json").readText(),
            File(data, "clip_layout.json").readText(),
        )
    }
    private val modelDir = File(root, "sdxl-npu/out/lora_super4")
    private val work = File(root, "sdxl-npu/out/lora_local/mix_test")

    @Test fun packRoundTrips() {
        val rnd = Random(1)
        for ((o, i) in listOf(64 to 640, 1280 to 64, 10240 to 64, 64 to 2048)) {
            val q = ByteArray(o * i) { rnd.nextInt(-127, 128).toByte() }
            assertArrayEquals(q, LoraConverter.unpackConv(LoraConverter.packConv(q, o, i), o, i))
        }
        for ((k, n) in listOf(768 to 64, 64 to 3072, 1280 to 64, 64 to 5120)) {
            val m = FloatArray(k * n) { Safetensors.halfToFloat(ClipLoraConverter.floatToHalf(rnd.nextFloat() - 0.5f)) }
            assertArrayEquals(m, ClipLoraConverter.unpackFp16(ClipLoraConverter.packFp16(m, k, n), k, n), 0f)
        }
    }

    @Test fun eigenDecomposes() {
        val rnd = Random(2)
        for (n in listOf(1, 2, 5, 40, 130)) {
            val x = Array(n) { DoubleArray(n) { rnd.nextDouble() - 0.5 } }
            val s = Array(n) { p -> DoubleArray(n) { q -> var v = 0.0; for (l in 0 until n) v += x[p][l] * x[q][l]; v } }
            val (vals, vecs) = SymEigen.decompose(s)
            for (i in 0 until n) {
                for (p in 0 until n) {   // S v = λ v
                    var v = 0.0
                    for (q in 0 until n) v += s[p][q] * vecs[i][q]
                    assertEquals(vals[i] * vecs[i][p], v, 1e-9 * maxOf(1.0, vals[0]))
                }
                if (i > 0) assertTrue(vals[i] <= vals[i - 1] + 1e-12)
            }
        }
    }

    /** 切り詰めは、rank が k 以下ならそのまま（誤差なし）、超えたら最良の近似（誤差の二乗 = 捨てた固有値の和）。 */
    @Test fun truncationIsOptimal() {
        val rnd = Random(3)
        val inC = 48
        val outC = 72
        for ((r, k) in listOf(20 to 32, 70 to 32, 128 to 40)) {
            val a = List(r) { FloatArray(inC) { (rnd.nextDouble() - 0.5).toFloat() } }
            val b = List(r) { l -> FloatArray(outC) { ((rnd.nextDouble() - 0.5) / (1 + l)).toFloat() } }   // 成分ごとに大きさを変える
            val w = dense(a, b, inC, outC)
            val (na, nb) = LowRank.truncate(a, b, k)
            val approx = Array(outC) { o -> DoubleArray(inC) { i -> var s = 0.0; for (l in na.indices) s += nb[l][o] * na[l][i]; s } }
            var err = 0.0
            var norm = 0.0
            for (o in 0 until outC) for (i in 0 until inC) { val d = w[o][i] - approx[o][i]; err += d * d; norm += w[o][i] * w[o][i] }
            // 最良の誤差: WᵀW の固有値のうち、上位 k 以外の和
            val wtw = Array(inC) { p -> DoubleArray(inC) { q -> var s = 0.0; for (o in 0 until outC) s += w[o][p] * w[o][q]; s } }
            val best = SymEigen.decompose(wtw).first.drop(k).sumOf { maxOf(it, 0.0) }
            assertEquals("r=$r k=$k", best, err, 1e-8 * norm)
            for (p in na.indices) for (q in na.indices) {   // a の行は正規直交
                var s = 0.0
                for (i in 0 until inC) s += na[p][i] * na[q][i]
                assertEquals(if (p == q) 1.0 else 0.0, s, 1e-9)
            }
        }
    }

    /** 実物: rank の合計が 64 以下（mmm 8 + cure 32）は並べるだけ。ΔW は各 LoRA の ΔW の和と、B の量子化の差だけ違う。 */
    @Test fun mixConcatMatchesSum() = mixCheck(listOf("mmm" to "MMM2013XLNAI.safetensors", "cure" to "curearcanashadow.safetensors"), listOf(1.0f, 0.7f), expectTruncated = false)

    /** 実物: rank の合計が 64 を超える（mw25 64 + cure 32）は切り詰める。ΔW の相対誤差を見る。 */
    @Test fun mixTruncatedApproximatesSum() = mixCheck(listOf("mw25" to "mw25.safetensors", "cure" to "curearcanashadow.safetensors"), listOf(1.0f, 0.8f), expectTruncated = true)

    private fun mixCheck(loras: List<Pair<String, String>>, ws: List<Float>, expectTruncated: Boolean) {
        assumeTrue(File(modelDir, LoraConverter.DEFAULT_ADAPTER).isFile && loras.all { File(root, "Model/${it.second}").isFile })
        val dir = File(work, "loras").also { it.mkdirs() }
        for ((n, src) in loras) if (!File(dir, n + LoraConverter.SUFFIX).isFile || !File(dir, "$n.json").isFile) {
            File(dir, n + LoraConverter.SUFFIX).delete()
            converter.convert(File(root, "Model/$src"), n, "", modelDir, dir)
        }
        val parts = loras.mapIndexed { j, (n, _) -> LoraMixer.Part(n, ws[j], JSONObject(File(dir, "$n.json").readText()).getDouble("c")) }
        val out = File(dir, "mix")
        val key = "test_" + loras.joinToString("_") { it.first }
        val t0 = System.nanoTime()
        val r = LoraMixer(converter).mix(parts, dir, modelDir, out, key)
        println("mix $key: ${(System.nanoTime() - t0) / 1_000_000} ms, c=${r.c}, truncated=${r.truncated}, tensors=${r.tensors}, over=${r.overRatio}, clip=${r.clipFiles}")
        assertEquals(expectTruncated, r.truncated > 0)
        assertTrue(File(out, "$key.json").isFile)
        assertEquals(converter.size, File(out, key + LoraConverter.SUFFIX).length())

        // UNet: いくつかの attach 点で ΔW を比べる（大きい層は重いので、in・out が 1280 以下のもの）
        val mixMap = map(File(out, key + LoraConverter.SUFFIX))
        val srcMaps = parts.map { map(File(dir, it.name + LoraConverter.SUFFIX)) }
        val picks = converter.tensors.filter { it.inC <= 1280 && it.outC <= 1280 }.let { l -> listOf(0, l.size / 3, l.size / 2, l.size - 1).map { l[it] } }
        for (t in picks) {
            val ref = Array(t.outC) { DoubleArray(t.inC) }
            for ((j, p) in parts.withIndex()) addDelta(ref, converter.readQ(srcMaps[j].duplicate(), t), t, p.strength * p.c)
            val got = Array(t.outC) { DoubleArray(t.inC) }
            addDelta(got, converter.readQ(mixMap.duplicate(), t), t, r.c)
            val rel = relErr(ref, got)
            println("  unet ${t.node} ${t.outC}x${t.inC}: rel=$rel")
            assertTrue("${t.node} の相対誤差 $rel", rel < if (expectTruncated) 0.35 else 0.02)
        }
        // CLIP: 最初と真ん中の attach 点で比べる（FP16 の丸めと、切り詰めの差だけ）
        for (clip in r.clipFiles.map { it.removePrefix("$key.").removeSuffix(".bin") }) {
            val spec = converter.clip.spec(clip)
            val mm = map(File(out, "$key.$clip.bin"))
            for (t in listOf(spec.tensors[0], spec.tensors[spec.tensors.size / 2])) {
                val ref = Array(t.outC) { DoubleArray(t.inC) }
                for (p in parts) {
                    val f = File(dir, "${p.name}.$clip.bin")
                    if (f.isFile) addClipDelta(ref, converter.clip.readAB(map(f), t), t, p.strength)
                }
                val got = Array(t.outC) { DoubleArray(t.inC) }
                addClipDelta(got, converter.clip.readAB(mm, t), t, 1f)
                val rel = relErr(ref, got)
                println("  $clip ${t.node}: rel=$rel")
                assertTrue("$clip ${t.node} の相対誤差 $rel", rel < if (expectTruncated) 0.35 else 0.01)
            }
        }
    }

    private fun map(f: File): ByteBuffer = RandomAccessFile(f, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }.order(ByteOrder.LITTLE_ENDIAN)

    private fun addDelta(acc: Array<DoubleArray>, q: Pair<ByteArray, ByteArray>, t: LoraConverter.Tensor, alpha: Double) {
        val (qa, qb) = q
        for (k in 0 until LoraMixer.MAX_RANK) for (o in 0 until t.outC) {
            val bv = qb[o * LoraMixer.MAX_RANK + k] * t.bScale * alpha
            if (bv == 0.0) continue
            for (i in 0 until t.inC) acc[o][i] += bv * qa[k * t.inC + i] * t.aScale
        }
    }

    private fun addClipDelta(acc: Array<DoubleArray>, ab: Pair<FloatArray, FloatArray>, t: ClipLoraConverter.Tensor, w: Float) {
        val (a, b) = ab   // a [in, 64]、b [64, out]
        for (k in 0 until LoraMixer.MAX_RANK) for (o in 0 until t.outC) {
            val bv = b[k * t.outC + o].toDouble() * w
            if (bv == 0.0) continue
            for (i in 0 until t.inC) acc[o][i] += bv * a[i * LoraMixer.MAX_RANK + k]
        }
    }

    private fun relErr(ref: Array<DoubleArray>, got: Array<DoubleArray>): Double {
        var e = 0.0
        var n = 0.0
        for (o in ref.indices) for (i in ref[o].indices) { val d = ref[o][i] - got[o][i]; e += d * d; n += ref[o][i] * ref[o][i] }
        return if (n == 0.0) sqrt(e) else sqrt(e / n)
    }

    private fun dense(a: List<FloatArray>, b: List<FloatArray>, inC: Int, outC: Int) =
        Array(outC) { o -> DoubleArray(inC) { i -> var s = 0.0; for (l in a.indices) s += b[l][o].toDouble() * a[l][i]; s } }
}
