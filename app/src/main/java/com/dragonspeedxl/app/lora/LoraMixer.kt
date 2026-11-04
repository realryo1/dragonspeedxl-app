package com.dragonspeedxl.app.lora

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.sqrt

/** 生成に使う LoRA 1 つ分（名前と強さ w。A1111 の <lora:名前:w> の w）。 */
data class LoraUse(val name: String, val strength: Float)

/**
 * 複数の LoRA（取り込み済みの adapter。loras/<名前>.4g.bin と .clip_l.bin / .clip_g.bin）を、1 つの adapter に合成する（LoRA の多重適用）。
 * ComfyUI の LoraLoader を重ねたときと同じく、重みの差分を足し合わせる: ΔW = Σ w_j · B_j A_j（UNet も CLIP も、強さは同じ w_j）。
 *
 * NPU の UNet の LoRA の枝は rank 64 で、alpha の入力は全体で 1 つ（attach 点ごとにも LoRA ごとにも分けられない）。だから:
 * - 強さ w_j と係数 c_j は B に焼き込み、実行時の alpha は合成した adapter の係数 C（強さ 1）にする。
 * - attach 点ごとに、各 LoRA の使っている lane（A の行と B の列がどちらも 0 でない rank の成分）を集める。
 *   合計が 64 以下なら、そのまま並べる（A の int8 はそのまま写すので、元の adapter と同じ。B だけ C で量子化し直す）。
 *   64 を超えたら、ΔW の rank 64 の最良近似（特異値の大きい順の 64 成分）に切り詰める。A の行の大きさは、元の LoRA の A の行の大きさにそろえる
 *   （A の出力の 16bit の活性の範囲を、元の LoRA の範囲から外さないため）。
 * - C は、1 つの LoRA の変換と同じ決め方（B の最大の中央値 / b_ref）で、合成した B から決める。
 * CLIP（FP16）も同じ考え方で合成する（実行時の強さは 1。B に w_j を焼き込む）。CLIP の adapter を持つ LoRA が 1 つも無ければ作らない。
 * 量子化した adapter から合成するので、元の safetensors は要らない（端末には残っていない）。
 */
class LoraMixer(private val conv: LoraConverter) {
    /** 合成する LoRA 1 つ分。c は loras/<名前>.json の係数。 */
    class Part(val name: String, val strength: Float, val c: Double)

    /** truncated は rank を切り詰めた attach 点の数（UNet と CLIP の合計）、tensors は合成した attach 点の数。overRatio は量子化で切った値の割合。 */
    class Result(val c: Double, val clipFiles: List<String>, val truncated: Int, val tensors: Int, val overRatio: Double)

    /**
     * 並列の数。1 スレッドが持つ作業の配列（いちばん大きい attach 点で、lane の B の列 + 切り詰めの結果 + 読んだ int8）の見積もりで、
     * Java のヒープ（アプリは 256MB）の空きの 4 割に収まる数にする。
     */
    private fun threadsFor(n: Int): Int {
        val rt = Runtime.getRuntime()
        val free = rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())
        val maxOut = conv.tensors.maxOf { it.outC }.toLong()
        val perThread = maxOut * MAX_RANK * (5L * n + 8)
        return (free * 4 / 10 / perThread).toInt().coerceIn(1, minOf(8, rt.availableProcessors()))
    }

    /**
     * parts を合成して、outDir に <name>.4g.bin（と、あれば <name>.clip_l.bin / .clip_g.bin）と <name>.json を作る。<name>.json は最後に書く（できた印）。
     * modelDir に、ひな型の default の super adapter が要る。progress(0〜1)。cancelled が true を返したら止めて CancellationException。
     */
    fun mix(
        parts: List<Part>, lorasDir: File, modelDir: File, outDir: File, name: String,
        progress: (Float) -> Unit = {}, cancelled: () -> Boolean = { false },
    ): Result {
        require(parts.size >= 2) { "合成する LoRA は 2 つ以上" }
        val template = File(modelDir, LoraConverter.DEFAULT_ADAPTER)
        require(template.isFile && template.length() == conv.size) { "ひな型 ${LoraConverter.DEFAULT_ADAPTER} が無いか、大きさが合わない" }
        val srcs = parts.map { p ->
            File(lorasDir, p.name + LoraConverter.SUFFIX).also {
                require(it.isFile && it.length() == conv.size) { "LoRA「${p.name}」の adapter が無いか、大きさが合わない" }
            }
        }
        outDir.mkdirs()
        File(outDir, "$name.json").delete()
        val threads = threadsFor(parts.size)
        val pool = Executors.newFixedThreadPool(threads)
        val made = ArrayList<File>()
        var ok = false
        try {
            val unet = mixUnet(parts, srcs, template, outDir, name, pool, { progress(0.85f * it) }, cancelled)
            made += File(outDir, name + LoraConverter.SUFFIX)
            val clipFiles = ArrayList<String>()
            var truncated = unet.truncated
            var tensors = unet.tensors
            for ((ci, clip) in ClipLoraConverter.CLIPS.keys.withIndex()) {
                val r = mixClip(clip, parts, lorasDir, File(outDir, "$name.$clip.bin"), pool, threads, cancelled) ?: continue
                made += File(outDir, "$name.$clip.bin")
                clipFiles += "$name.$clip.bin"
                truncated += r.first
                tensors += r.second
                progress(0.85f + 0.15f * (ci + 1) / ClipLoraConverter.CLIPS.size)
            }
            val meta = JSONObject().put("c", unet.c).put("truncated", truncated).put("tensors", tensors)
                .put("mix", JSONArray().also { a -> parts.forEach { a.put(JSONObject().put("name", it.name).put("strength", it.strength.toDouble())) } })
            File(outDir, "$name.json").writeText(meta.toString(), Charsets.UTF_8)
            progress(1f)
            ok = true
            return Result(unet.c, clipFiles, truncated, tensors, unet.overRatio)
        } finally {
            pool.shutdownNow()
            if (!ok) {   // 途中までのもの（.part、一時ファイル、できた分）を消す
                for (f in made) f.delete()
                outDir.listFiles { f -> f.name.startsWith("$name.") }?.forEach { it.delete() }
            }
        }
    }

    private class UnetResult(val c: Double, val truncated: Int, val tensors: Int, val overRatio: Double)

    private fun mixUnet(
        parts: List<Part>, srcs: List<File>, template: File, outDir: File, name: String,
        pool: java.util.concurrent.ExecutorService, progress: (Float) -> Unit, cancelled: () -> Boolean,
    ): UnetResult {
        val ts = conv.tensors
        val part = File(outDir, name + LoraConverter.SUFFIX + ".part")
        val tmp = File(outDir, "$name.btmp")   // 1 回目で作った B（float）。C が決まってから量子化する
        val bPos = LongArray(ts.size + 1).also { for (i in ts.indices) it[i + 1] = it[i] + ts[i].outC.toLong() * MAX_RANK * 4 }
        template.copyTo(part, overwrite = true)
        val inputs = srcs.map { RandomAccessFile(it, "r") }
        try {
            val maps = inputs.map { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }
            RandomAccessFile(part, "rw").use { praf -> RandomAccessFile(tmp, "rw").use { traf ->
                traf.setLength(0)
                traf.setLength(bPos.last())
                val out = conv.writer(praf.channel.map(FileChannel.MapMode.READ_WRITE, 0, praf.length()).also { it.order(ByteOrder.LITTLE_ENDIAN) })
                val bTmp = traf.channel.map(FileChannel.MapMode.READ_WRITE, 0, bPos.last()).also { it.order(ByteOrder.LITTLE_ENDIAN) }
                val maxB = DoubleArray(ts.size) { -1.0 }   // -1 はこの attach 点に lane が無い（ひな型の 0 のまま）
                val truncated = AtomicInteger()
                val over = AtomicLong()
                val total = AtomicLong()
                val done = AtomicInteger()
                // 1 回目: attach 点ごとに lane を集め、A を書き、B（float）を一時ファイルへ
                runAll(pool, ts.indices.toList(), cancelled) { i ->
                    val t = ts[i]
                    val qa = ArrayList<ByteArray>()     // lane ごとの A の行（int8）
                    val bCol = ArrayList<FloatArray>()  // lane ごとの B の列（w・c を掛けた、alpha 込みの float）
                    for ((j, p) in parts.withIndex()) {
                        val (a, b) = conv.readQ(maps[j].duplicate(), t)
                        val f = p.strength.toDouble() * p.c * t.bScale
                        for (k in 0 until MAX_RANK) {
                            if (!rowNonZero(a, k * t.inC, t.inC) || !colNonZero(b, k, t.outC)) continue
                            qa += a.copyOfRange(k * t.inC, (k + 1) * t.inC)
                            bCol += FloatArray(t.outC) { (b[it * MAX_RANK + k] * f).toFloat() }
                        }
                    }
                    if (qa.isNotEmpty()) {
                        val qaOut = ByteArray(MAX_RANK * t.inC)
                        val bOut = FloatArray(t.outC * MAX_RANK)
                        if (qa.size <= MAX_RANK) {
                            for (k in qa.indices) {
                                qa[k].copyInto(qaOut, k * t.inC)
                                val col = bCol[k]
                                for (o in 0 until t.outC) bOut[o * MAX_RANK + k] = col[o]
                            }
                        } else {
                            truncated.incrementAndGet()
                            val aRows = qa.map { r -> FloatArray(t.inC) { r[it] * t.aScale } }
                            val tau = rmsNorm(aRows)
                            val (aN, bN) = LowRank.truncate(aRows, bCol, MAX_RANK)
                            val lim = t.aScale * 127.0 * 0.98
                            var o2 = 0L
                            for (k in aN.indices) {
                                var mx = 0.0
                                for (v in aN[k]) mx = maxOf(mx, abs(v))
                                val s = if (mx * tau > lim) lim / mx else tau   // A の行の大きさは元の LoRA の A にそろえる（int8 に収まる範囲で）
                                for (x in 0 until t.inC) {
                                    val v = (aN[k][x] * s).toFloat()
                                    qaOut[k * t.inC + x] = LoraConverter.quant(v, t.aScale)
                                    if (abs(v) > t.aScale * 127f) o2++
                                }
                                for (o in 0 until t.outC) bOut[o * MAX_RANK + k] = (bN[k][o] / s).toFloat()
                            }
                            over.addAndGet(o2)
                        }
                        out.writeA(t, qaOut)
                        var mx = 0.0
                        for (v in bOut) mx = maxOf(mx, abs(v).toDouble())
                        maxB[i] = mx
                        val bb = bTmp.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                        bb.position(bPos[i].toInt())
                        bb.asFloatBuffer().put(bOut)
                        total.addAndGet(qaOut.size.toLong())
                    }
                    progress(0.85f * done.incrementAndGet() / ts.size)
                }
                // C（B の最大の中央値 / b_ref。1 つの LoRA の変換と同じ決め方）
                val used = ts.indices.filter { maxB[it] >= 0 }
                require(used.isNotEmpty()) { "合成する LoRA に、UNet の重みがない" }
                val sorted = used.map { maxB[it].toFloat() }.sorted()
                val n = sorted.size
                val med = if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2f
                // alpha の入力の encodings は 0〜32（sdxl-npu/convert/lora_prep.py の ALPHA_MAX）。超える分は B の量子化で切れる（overRatio に出る）
                val c = ((med.toDouble() / conv.bRef).takeIf { it > 0 } ?: 1.0).coerceAtMost(ALPHA_CAP)
                val cf = c.toFloat()
                // 2 回目: B を C で割って量子化して書く
                runAll(pool, used, cancelled) { i ->
                    val t = ts[i]
                    val bOut = FloatArray(t.outC * MAX_RANK)
                    val bb = bTmp.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                    bb.position(bPos[i].toInt())
                    bb.asFloatBuffer().get(bOut)
                    val qb = ByteArray(bOut.size)
                    var o2 = 0L
                    for (k in bOut.indices) {
                        val v = bOut[k] / cf
                        qb[k] = LoraConverter.quant(v, t.bScale)
                        if (abs(v) > t.bScale * 127f) o2++
                    }
                    over.addAndGet(o2)
                    total.addAndGet(qb.size.toLong())
                    out.writeB(t, qb)
                }
                out.force()
                progress(0.85f)
                praf.close()
                LoraConverter.moveInto(part, File(outDir, name + LoraConverter.SUFFIX))
                return UnetResult(c, truncated.get(), used.size, if (total.get() == 0L) 0.0 else over.get().toDouble() / total.get())
            } }
        } finally {
            inputs.forEach { it.close() }
            tmp.delete()
            part.delete()
        }
    }

    /** CLIP（clip_l / clip_g）の adapter を合成して dest に書く。この CLIP の adapter を持つ LoRA が無ければ null。戻り値は（切り詰めた数、合成した数）。 */
    private fun mixClip(
        clip: String, parts: List<Part>, lorasDir: File, dest: File,
        pool: java.util.concurrent.ExecutorService, threads: Int, cancelled: () -> Boolean,
    ): Pair<Int, Int>? {
        val spec = conv.clip.spec(clip)
        val have = parts.mapNotNull { p -> File(lorasDir, "${p.name}.$clip.bin").takeIf { it.isFile && it.length() == spec.size }?.let { p to it } }
        if (have.isEmpty()) return null
        val inputs = have.map { RandomAccessFile(it.second, "r") }
        try {
            val maps = inputs.map { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }
            val truncated = AtomicInteger()
            val tensors = AtomicInteger()
            // attach 点ごとの計算は並列に先回りし（メモリを抑えるため、先回りは数個まで）、書き込みは順に行う
            val window = threads * 2
            val futures = arrayOfNulls<Future<Pair<FloatArray, FloatArray>?>>(spec.tensors.size)
            fun submit(i: Int) {
                if (i >= spec.tensors.size) return
                val t = spec.tensors[i]
                futures[i] = pool.submit<Pair<FloatArray, FloatArray>?> {
                    if (cancelled()) throw CancellationException()
                    val aCol = ArrayList<FloatArray>()   // lane ごとの A の列（[in]）
                    val bRow = ArrayList<FloatArray>()   // lane ごとの B の行（[out]。w を掛けたもの）
                    for ((j, pf) in have.withIndex()) {
                        val (a, b) = conv.clip.readAB(maps[j].duplicate().order(ByteOrder.LITTLE_ENDIAN), t)
                        val w = pf.first.strength
                        for (k in 0 until MAX_RANK) {
                            val ac = FloatArray(t.inC) { a[it * MAX_RANK + k] }
                            val br = FloatArray(t.outC) { b[k * t.outC + it] * w }
                            if (ac.any { it != 0f } && br.any { it != 0f }) { aCol += ac; bRow += br }
                        }
                    }
                    if (aCol.isEmpty()) return@submit null
                    tensors.incrementAndGet()
                    val a2 = FloatArray(t.inC * MAX_RANK)
                    val b2 = FloatArray(MAX_RANK * t.outC)
                    if (aCol.size <= MAX_RANK) {
                        for (k in aCol.indices) {
                            for (x in 0 until t.inC) a2[x * MAX_RANK + k] = aCol[k][x]
                            bRow[k].copyInto(b2, k * t.outC)
                        }
                    } else {
                        truncated.incrementAndGet()
                        val tau = rmsNorm(aCol)
                        val (aN, bN) = LowRank.truncate(aCol, bRow, MAX_RANK)
                        for (k in aN.indices) {
                            for (x in 0 until t.inC) a2[x * MAX_RANK + k] = (aN[k][x] * tau).toFloat()
                            for (o in 0 until t.outC) b2[k * t.outC + o] = (bN[k][o] / tau).toFloat()
                        }
                    }
                    a2 to b2
                }
            }
            for (i in 0 until minOf(window, spec.tensors.size)) submit(i)
            conv.clip.writeAdapter(clip, dest) { i, _ ->
                val r = try { futures[i]!!.get() } catch (e: java.util.concurrent.ExecutionException) { throw e.cause ?: e }
                futures[i] = null
                submit(i + window)
                r
            }
            if (cancelled()) { dest.delete(); throw CancellationException() }
            return truncated.get() to tensors.get()
        } finally {
            inputs.forEach { it.close() }
        }
    }

    /** items を pool で並列に処理する。例外は呼び出し側へ投げ直す。cancelled なら CancellationException。 */
    private fun runAll(pool: java.util.concurrent.ExecutorService, items: List<Int>, cancelled: () -> Boolean, body: (Int) -> Unit) {
        val fs = items.map { i -> pool.submit(java.util.concurrent.Callable { if (!cancelled()) body(i) }) }
        try {
            for (f in fs) f.get()
        } catch (e: java.util.concurrent.ExecutionException) {
            fs.forEach { it.cancel(true) }
            throw e.cause ?: e
        }
        if (cancelled()) throw CancellationException("LoRA の合成を中断しました")
    }

    companion object {
        const val MAX_RANK = LoraConverter.MAX_RANK
        /** 合成の係数 C（実行時の alpha）の上限。UNet の alpha の入力の範囲（0〜32）に余裕を残す */
        private const val ALPHA_CAP = 30.0

        private fun rowNonZero(q: ByteArray, off: Int, n: Int): Boolean { for (x in off until off + n) if (q[x] != 0.toByte()) return true; return false }
        private fun colNonZero(q: ByteArray, k: Int, outC: Int): Boolean { for (o in 0 until outC) if (q[o * MAX_RANK + k] != 0.toByte()) return true; return false }

        /** 行（または列）の L2 ノルムの二乗平均の平方根。 */
        private fun rmsNorm(v: List<FloatArray>): Double {
            var s = 0.0
            for (r in v) for (x in r) s += x.toDouble() * x
            return sqrt(s / v.size)
        }
    }
}

/** 低 rank の和の切り詰め（LoraMixer が使う）。 */
internal object LowRank {
    /**
     * ΔW = Σ_l b[l] ⊗ a[l]（b[l] は出力側の列 [out]、a[l] は入力側の行 [in]）を、rank k の最良近似（フロベニウスノルムで）に切り詰める。
     * 戻り値は（a の行、b の列）で、a の各行は単位ノルムで互いに直交する。特異値の大きい順。k 個より少ないこともある（ΔW の rank が小さいとき）。
     *
     * 方法: Ga = A Aᵀ を Cholesky 分解（Ga = Rᵀ R。ほかの行の組み合わせで表せる行は落とす）すると、A = Rᵀ Qᵀ（Q は正規直交）。
     * ΔW = (B Rᵀ) Qᵀ なので、N = B Rᵀ の右特異ベクトル（S = Nᵀ N = R Gb Rᵀ の固有ベクトル Y）の上位 k を取り、
     * a の新しい行 = Yᵀ Qᵀ = Yᵀ R⁻ᵀ A、b の新しい列 = N Y = B Rᵀ Y とする。計算は R の大きさ（lane の数）の行列と、Gram の内積だけ。
     */
    fun truncate(a: List<FloatArray>, b: List<FloatArray>, k: Int): Pair<Array<DoubleArray>, Array<DoubleArray>> {
        val r = a.size
        require(b.size == r && r > 0)
        val inC = a[0].size
        val outC = b[0].size
        val ga = gram(a)
        val gb = gram(b)
        // Cholesky（上三角 R。Ga = Rᵀ R）。残りが小さい行は落とす（ほかの行の組み合わせ）
        var maxDiag = 0.0
        for (i in 0 until r) maxDiag = maxOf(maxDiag, ga[i][i])
        val tol = 1e-10 * maxDiag
        val rr = Array(r) { DoubleArray(r) }
        val kept = ArrayList<Int>()
        for (j in 0 until r) {
            var s = ga[j][j]
            for (i in kept) s -= rr[i][j] * rr[i][j]
            if (s <= tol) continue
            val d = sqrt(s)
            rr[j][j] = d
            for (l in j + 1 until r) {
                var v = ga[j][l]
                for (i in kept) v -= rr[i][j] * rr[i][l]
                rr[j][l] = v / d
            }
            kept += j
        }
        val m = kept.size
        if (m == 0) return emptyArray<DoubleArray>() to emptyArray()
        val rk = Array(m) { rr[kept[it]] }   // m × r
        // S = Rk Gb Rkᵀ（m × m）
        val tm = Array(m) { p -> DoubleArray(r) { q -> var s = 0.0; for (l in 0 until r) s += rk[p][l] * gb[l][q]; s } }
        val sm = Array(m) { p -> DoubleArray(m) { q -> var s = 0.0; for (l in 0 until r) s += tm[p][l] * rk[q][l]; s } }
        for (p in 0 until m) for (q in 0 until p) { val v = 0.5 * (sm[p][q] + sm[q][p]); sm[p][q] = v; sm[q][p] = v }
        val (vals, vecs) = SymEigen.decompose(sm)   // vecs[i] は i 番目の固有ベクトル（大きい順）
        var kk = 0
        while (kk < minOf(k, m) && vals[kk] > 1e-14 * maxOf(vals[0], 1e-300)) kk++
        // a の新しい行: Yᵀ Rk_KK⁻ᵀ A_K。X = Rk_KK⁻¹ Y（上三角の後退代入）として、行 l = Σ_j X[j][l] · a[kept[j]]
        val newA = Array(kk) { l ->
            val y = vecs[l]
            val x = DoubleArray(m)
            for (p in m - 1 downTo 0) {
                var s = y[p]
                for (q in p + 1 until m) s -= rk[p][kept[q]] * x[q]
                x[p] = s / rk[p][kept[p]]
            }
            val row = DoubleArray(inC)
            for (j in 0 until m) { val c = x[j]; if (c != 0.0) { val src = a[kept[j]]; for (i in 0 until inC) row[i] += c * src[i] } }
            row
        }
        // b の新しい列: B Rkᵀ y_l。係数 z = Rkᵀ y_l（長さ r）
        val newB = Array(kk) { l ->
            val y = vecs[l]
            val col = DoubleArray(outC)
            for (q in 0 until r) {
                var z = 0.0
                for (p in 0 until m) z += rk[p][q] * y[p]
                if (z != 0.0) { val src = b[q]; for (o in 0 until outC) col[o] += z * src[o] }
            }
            col
        }
        // 数値の誤差で単位ノルムからずれた分は b へ移す（積は変わらない）
        for (l in 0 until kk) {
            var s = 0.0
            for (v in newA[l]) s += v * v
            val n = sqrt(s)
            if (n > 0) { for (i in 0 until inC) newA[l][i] /= n; for (o in 0 until outC) newB[l][o] *= n }
        }
        return newA to newB
    }

    private fun gram(v: List<FloatArray>): Array<DoubleArray> {
        val r = v.size
        val g = Array(r) { DoubleArray(r) }
        for (p in 0 until r) {
            val x = v[p]
            for (q in 0..p) {
                val y = v[q]
                var s = 0.0
                for (i in x.indices) s += x[i].toDouble() * y[i]
                g[p][q] = s; g[q][p] = s
            }
        }
        return g
    }
}

/** 実対称行列の固有値分解（Householder の三重対角化 + 陰的 QL）。 */
internal object SymEigen {
    /** 戻り値は（固有値の大きい順、それぞれの固有ベクトル）。a は壊さない。 */
    fun decompose(a0: Array<DoubleArray>): Pair<DoubleArray, Array<DoubleArray>> {
        val n = a0.size
        val a = Array(n) { a0[it].copyOf() }
        val d = DoubleArray(n)
        val e = DoubleArray(n)
        tred2(a, d, e)
        tqli(d, e, a)
        // a の列 i が固有値 d[i] の固有ベクトル
        val order = (0 until n).sortedByDescending { d[it] }
        return DoubleArray(n) { d[order[it]] } to Array(n) { i -> DoubleArray(n) { k -> a[k][order[i]] } }
    }

    private fun tred2(a: Array<DoubleArray>, d: DoubleArray, e: DoubleArray) {
        val n = a.size
        for (i in n - 1 downTo 1) {
            val l = i - 1
            var h = 0.0
            if (l > 0) {
                var scale = 0.0
                for (k in 0..l) scale += abs(a[i][k])
                if (scale == 0.0) {
                    e[i] = a[i][l]
                } else {
                    for (k in 0..l) { a[i][k] /= scale; h += a[i][k] * a[i][k] }
                    var f = a[i][l]
                    var g = if (f >= 0.0) -sqrt(h) else sqrt(h)
                    e[i] = scale * g
                    h -= f * g
                    a[i][l] = f - g
                    f = 0.0
                    for (j in 0..l) {
                        a[j][i] = a[i][j] / h
                        g = 0.0
                        for (k in 0..j) g += a[j][k] * a[i][k]
                        for (k in j + 1..l) g += a[k][j] * a[i][k]
                        e[j] = g / h
                        f += e[j] * a[i][j]
                    }
                    val hh = f / (h + h)
                    for (j in 0..l) {
                        f = a[i][j]
                        g = e[j] - hh * f
                        e[j] = g
                        for (k in 0..j) a[j][k] -= (f * e[k] + g * a[i][k])
                    }
                }
            } else {
                e[i] = a[i][l]
            }
            d[i] = h
        }
        d[0] = 0.0
        e[0] = 0.0
        for (i in 0 until n) {
            val l = i - 1
            if (d[i] != 0.0) {
                for (j in 0..l) {
                    var g = 0.0
                    for (k in 0..l) g += a[i][k] * a[k][j]
                    for (k in 0..l) a[k][j] -= g * a[k][i]
                }
            }
            d[i] = a[i][i]
            a[i][i] = 1.0
            for (j in 0..l) { a[j][i] = 0.0; a[i][j] = 0.0 }
        }
    }

    private fun tqli(d: DoubleArray, e: DoubleArray, z: Array<DoubleArray>) {
        val n = d.size
        for (i in 1 until n) e[i - 1] = e[i]
        if (n > 0) e[n - 1] = 0.0
        for (l in 0 until n) {
            var iter = 0
            var m: Int
            do {
                m = l
                while (m < n - 1) {
                    val dd = abs(d[m]) + abs(d[m + 1])
                    if (abs(e[m]) <= 1e-15 * dd) break
                    m++
                }
                if (m != l) {
                    check(iter++ < 60) { "固有値の計算が収束しない" }
                    var g = (d[l + 1] - d[l]) / (2.0 * e[l])
                    var r = hypot(g, 1.0)
                    g = d[m] - d[l] + e[l] / (g + if (g >= 0) abs(r) else -abs(r))
                    var s = 1.0
                    var c = 1.0
                    var p = 0.0
                    var i = m - 1
                    var early = false
                    while (i >= l) {
                        var f = s * e[i]
                        val b = c * e[i]
                        r = hypot(f, g)
                        e[i + 1] = r
                        if (r == 0.0) { d[i + 1] -= p; e[m] = 0.0; early = true; break }
                        s = f / r
                        c = g / r
                        g = d[i + 1] - p
                        r = (d[i] - g) * s + 2.0 * c * b
                        p = s * r
                        d[i + 1] = g + p
                        g = c * r - b
                        for (k in 0 until n) {
                            f = z[k][i + 1]
                            z[k][i + 1] = s * z[k][i] + c * f
                            z[k][i] = c * z[k][i] - s * f
                        }
                        i--
                    }
                    if (early) continue
                    d[l] -= p
                    e[l] = g
                    e[m] = 0.0
                }
            } while (m != l)
        }
    }

    private fun hypot(a: Double, b: Double): Double = kotlin.math.hypot(a, b)
}
