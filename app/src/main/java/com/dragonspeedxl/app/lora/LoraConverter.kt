package com.dragonspeedxl.app.lora

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.floor

/**
 * LoRA の safetensors を、縦横 4 グラフの共有 bin（unet_lora_4g.bin）用の super adapter（<名前>.4g.bin）にする。
 * PC 版（loraconverter/ の lora_to_npu.py と super_adapter.py）の移植。UNet は再コンパイルしない。出力は PC 版とバイト一致する（JVM のテストで確かめる）。
 *
 * super adapter = [ヘッダー] + [縦 Base の adapter の本体] + [末尾 × 3（縦タイル、横 Base、横タイル）]。
 * - ひな型は、端末の models/unet_lora_4g_default.bin（LoRA なしのときにエンジンが使う default の super adapter）。mw25 の super adapter の入れ物に
 *   default の値を入れたもの（loraconverter/super_adapter.py の template + fill + build。Colab の本物の default とは入れ物の形が違う）。これを丸ごと写す。
 * - 固定の尺度で int8 にした重みと、cbg が付ける統計を、layout.json のグラフごとの位置に書く。縦 Base は本体へ直接、
 *   残りの 3 グラフは、窓の対応表（super_4g_windows.json。そのグラフの adapter の位置 → 末尾の位置）を通して末尾へ書く。窓の外の位置は書かない
 *   （その位置は縦 Base と同じ値で、本体がそれを持っている）。
 * UNet の LoRA だけを対象にする（テキストエンコーダのキーは読み飛ばす）。rank が 64 を超える LoRA は受け付けない。
 *
 * 数値は PC 版（numpy の float32）と同じ演算の順序で、Float で計算する（丸めの境界で 1 バイトずれないように）。
 */
class LoraConverter(layoutJson: String, attachJson: String, windowsJson: String, clipLayoutJson: String) {
    /** clipFiles: 作ったテキストエンコーダの adapter（<名前>.clip_l.bin、.clip_g.bin。LoRA にテキストエンコーダの重みが無ければ空）。 */
    class Result(val c: Double, val missing: Int, val total: Int, val overRatio: Double, val clipFiles: List<String> = emptyList())

    private val clip = ClipLoraConverter(clipLayoutJson)

    private class Tensor(
        val node: String, val inC: Int, val outC: Int, val aScale: Float, val bScale: Float,
        val aOff: LongArray, val bOff: LongArray, val stat: Array<Stat>,
    )

    private class Stat(val aStat: Long, val aZp: Int, val bStat: Long, val bZp: Int)
    private class Entry(val base: String)
    /** 末尾の窓。グラフの adapter の [so, so + n) が、末尾の s から始まる。 */
    private class Win(val s: Long, val so: Long, val n: Long)
    private class Tail(val offset: Long, val wins: List<Win>)   // wins は so の昇順

    private val bRef: Double
    private val tensors = ArrayList<Tensor>()
    private val byNorm = HashMap<String, String>()   // diffusers の層の名前（'.' を '_' にしたもの） → attach 点（ONNX のノード名）
    private val header: Int        // super adapter のヘッダーの長さ（4 グラフ: 21,024）
    private val body: Long         // 本体の長さ（縦 Base の adapter の大きさ − 18,048）
    private val tails: List<Tail>  // 縦タイル、横 Base、横タイル
    private val size: Long         // super adapter の大きさ

    init {
        val layout = JSONObject(layoutJson)
        bRef = layout.getDouble("b_ref")
        val names = layout.getJSONObject("graphs").keys().asSequence().toSet()   // キーの順は org.json の実装によって違う（JVM では保たれない）
        require(names == GRAPHS.toSet()) { "layout.json のグラフが $GRAPHS ではない: $names" }
        val ts = layout.getJSONArray("tensors")
        for (i in 0 until ts.length()) {
            val t = ts.getJSONObject(i)
            fun perGraph(key: String) = LongArray(GRAPHS.size) { t.getJSONObject(key).getLong(GRAPHS[it]) }
            val st = t.getJSONObject("stat")
            tensors += Tensor(
                t.getString("node"), t.getInt("in"), t.getInt("out"), t.getDouble("a_scale").toFloat(), t.getDouble("b_scale").toFloat(),
                perGraph("a_off"), perGraph("b_off"),
                Array(GRAPHS.size) {
                    val s = st.getJSONObject(GRAPHS[it])
                    Stat(s.getLong("a_stat"), s.getInt("a_zp"), s.getLong("b_stat"), s.getInt("b_zp"))
                },
            )
        }
        val attach = JSONObject(attachJson)
        for (m in attach.keys()) byNorm[m.replace(".", "_")] = attach.getString(m)

        val w = JSONObject(windowsJson)
        header = w.getInt("header")
        body = w.getLong("body")
        val tj = w.getJSONArray("tails")
        require(tj.length() == GRAPHS.size - 1) { "super_4g_windows.json の末尾の数が ${GRAPHS.size - 1} ではない" }
        tails = List(tj.length()) { k ->
            val t = tj.getJSONObject(k)
            val ws = t.getJSONArray("windows")
            Tail(t.getLong("offset"), List(ws.length()) { val a = ws.getJSONArray(it); Win(a.getLong(0), a.getLong(1), a.getLong(2)) }.sortedBy { it.so })
        }
        size = header + body + tj.getJSONObject(tj.length() - 1).let { it.getLong("offset") + it.getLong("length") }
    }

    /**
     * src を変換して outDir/<name>.4g.bin と <name>.json を作る。modelDir にひな型の default の super adapter（DEFAULT_ADAPTER）が要る。
     * 途中の出力は .part に書き、全部できてから名前を変える（読み込み中のファイルを壊さない）。
     */
    fun convert(
        src: File, name: String, activation: String, modelDir: File, outDir: File,
        progress: (String, Float) -> Unit = { _, _ -> },
    ): Result {
        checkName(name)
        requireNew(outDir, name)
        val template = File(modelDir, DEFAULT_ADAPTER)
        require(template.isFile) { "ひな型 $DEFAULT_ADAPTER が ${modelDir.path} に無い" }
        require(template.length() == size) { "$DEFAULT_ADAPTER の大きさが super_4g_windows.json と合わない（Colab の本物の default ではなく、loraconverter で作ったものを置く）" }
        Safetensors(src).use { sf ->
            progress("読み込み中", 0f)
            val entries = matchKeys(sf)
            // 1 回目: c（B の最大の中央値 / b_ref）。B は alpha/rank を掛けたあとの値
            val maxB = ArrayList<Float>(entries.size)
            for (node in tensors.map { it.node }) {
                val e = entries[node] ?: continue
                val r = rankOf(sf, e)
                val scale = (sf.floats(e.base + ".alpha")[0].toDouble() / r).toFloat()
                var mx = 0f
                for (v in sf.floats(e.base + ".lora_up.weight")) mx = maxOf(mx, abs(v * scale))
                maxB += mx
            }
            maxB.sort()
            val n = maxB.size
            require(n > 0) { "UNet の LoRA のキー（lora_unet_*）が無い" }
            val med = if (n % 2 == 1) maxB[n / 2] else (maxB[n / 2 - 1] + maxB[n / 2]) / 2f
            val c = med.toDouble() / bRef
            val cf = c.toFloat()

            outDir.mkdirs()
            val part = File(outDir, name + SUFFIX + ".part")
            template.copyTo(part, overwrite = true)   // [ヘッダー] + [縦 Base の本体] + [末尾]。どれも default の値
            var over = 0L
            var total = 0L
            RandomAccessFile(part, "rw").use { raf ->
                val buf = raf.channel.map(FileChannel.MapMode.READ_WRITE, 0, raf.length()).also { it.order(java.nio.ByteOrder.LITTLE_ENDIAN) }
                val out = Writer(buf)
                for ((i, t) in tensors.withIndex()) {
                    val e = entries[t.node] ?: continue   // LoRA の無い層は、ひな型の 0 のまま（統計も 0）
                    val rank = rankOf(sf, e)
                    require(rank <= MAX_RANK) { "rank が $MAX_RANK を超える層がある" }
                    val a = sf.floats(e.base + ".lora_down.weight")
                    val bs = (sf.floats(e.base + ".alpha")[0].toDouble() / rank).toFloat()
                    val b = sf.floats(e.base + ".lora_up.weight")
                    require(a.size == rank * t.inC && b.size == t.outC * rank) { "${t.node} の形が layout と合わない" }
                    val qa = ByteArray(MAX_RANK * t.inC)
                    val qb = ByteArray(t.outC * MAX_RANK)
                    for (k in 0 until rank * t.inC) { qa[k] = quant(a[k], t.aScale); if (abs(a[k]) > t.aScale * QMAX_F) over++ }
                    for (o in 0 until t.outC) for (k in 0 until rank) {
                        val v = (b[o * rank + k] * bs) / cf
                        qb[o * MAX_RANK + k] = quant(v, t.bScale)
                        if (abs(v) > t.bScale * QMAX_F) over++
                    }
                    total += qa.size.toLong() + qb.size
                    val pa = packConv(qa, MAX_RANK, t.inC)
                    val pb = packConv(qb, t.outC, MAX_RANK)
                    val sa = rowSums(qa, MAX_RANK, t.inC)
                    val sb = rowSums(qb, t.outC, MAX_RANK)
                    for (g in GRAPHS.indices) {
                        out.write(g, t.aOff[g], pa)
                        out.write(g, t.bOff[g], pb)
                        val st = t.stat[g]
                        out.stats(g, st.aStat, sa, st.aZp)
                        out.stats(g, st.bStat, sb, st.bZp)
                    }
                    if (i % 20 == 0) progress("変換中 ${i + 1} / ${tensors.size}", 0.1f + 0.7f * i / tensors.size)
                }
                buf.force()
            }
            // テキストエンコーダ（CLIP）の adapter。以前の取り込みの残りが、重みの無い LoRA の取り込みで残らないよう、先に消す
            for (c in ClipLoraConverter.CLIPS.keys) File(outDir, "$name.$c.bin").delete()
            val clipFiles = clip.convert(sf, name, outDir) { msg, f -> progress(msg, 0.8f + 0.17f * f) }
            progress("書き出し中", 0.97f)
            val dest = File(outDir, name + SUFFIX)
            dest.delete()
            if (!part.renameTo(dest)) {   // Windows の JVM（テスト）では、マップが残っていて名前を変えられないことがある。端末（Linux）では起きない
                part.copyTo(dest, overwrite = true)
                part.delete()
            }
            val meta = JSONObject().put("c", c).put("activation", activation)
            File(outDir, "$name.json").writeText(meta.toString(), Charsets.UTF_8)
            progress("完了", 1f)
            return Result(c, tensors.count { entries[it.node] == null }, tensors.size, if (total == 0L) 0.0 else over.toDouble() / total, clipFiles)
        }
    }

    /** グラフ g の adapter の位置 off（別 context 用の adapter の座標）への書き込みを、super adapter の位置に置き換えて書く。 */
    private inner class Writer(private val buf: MappedByteBuffer) {
        private val four = ByteArray(4)
        private val tailBase = header + body

        fun write(g: Int, off: Long, bytes: ByteArray) {
            if (g == 0) {   // 縦 Base: 本体へ直接（ヘッダーより前の位置は、super adapter ではひな型のヘッダーが持つ）
                if (off < OLD_HEADER) return
                buf.position((off - OLD_HEADER + header).toInt())
                buf.put(bytes)
                return
            }
            val t = tails[g - 1]
            val end = off + bytes.size
            // so + n > off となる最初の窓から、so < end の間（窓は so の昇順で重ならない）
            var lo = 0
            var hi = t.wins.size
            while (lo < hi) { val m = (lo + hi) ushr 1; if (t.wins[m].so + t.wins[m].n <= off) lo = m + 1 else hi = m }
            var k = lo
            while (k < t.wins.size && t.wins[k].so < end) {
                val w = t.wins[k]
                val a = maxOf(off, w.so)
                val b = minOf(end, w.so + w.n)
                if (a < b) {
                    buf.position((tailBase + t.offset + w.s + (a - w.so)).toInt())
                    buf.put(bytes, (a - off).toInt(), (b - a).toInt())
                }
                k++
            }
        }

        /** 統計[j] = -((S[j] * zp + 128) >> 8)。S は出力 j の int8 の重みの合計、zp は入力活性のゼロ点。int32 で 8 バイト刻みに書く。 */
        fun stats(g: Int, base: Long, sums: LongArray, zp: Int) {
            val seq = storedBlockOrder((sums.size + 255) / 256)
            val pos = IntArray(seq.size).also { for ((p, l) in seq.withIndex()) it[l] = p }
            for (j in sums.indices) {
                val jj = j % 256
                val off = base + ((16 * pos[j / 256] + (jj / 32) * 2 + (jj % 2)) * 256 + ((jj % 32) / 2) * 8)
                val v = (-((sums[j] * zp + 128) shr 8)).toInt()
                four[0] = v.toByte(); four[1] = (v shr 8).toByte(); four[2] = (v shr 16).toByte(); four[3] = (v shr 24).toByte()
                write(g, off, four)
            }
        }
    }

    private fun rankOf(sf: Safetensors, e: Entry): Int = sf.tensors.getValue(e.base + ".lora_down.weight").shape[0]

    /** UNet の対象層を attach 点に対応づける。対応しない UNet のキーがあれば例外。lora_te* は読み飛ばす。 */
    private fun matchKeys(sf: Safetensors): Map<String, Entry> {
        val out = HashMap<String, Entry>()
        val unmatched = ArrayList<String>()
        for (base in sf.tensors.keys.filter { it.endsWith(".alpha") }.map { it.removeSuffix(".alpha") }.sorted()) {
            if (!base.startsWith("lora_unet_")) continue
            val node = byNorm[sgmToDiffusersStem(base.removePrefix("lora_unet_"))]
            if (node == null) unmatched += base else out[node] = Entry(base)
        }
        require(unmatched.isEmpty()) { "対応しない UNet の LoRA のキーが ${unmatched.size} 個ある。例: ${unmatched.take(3)}" }
        return out
    }

    companion object {
        const val MAX_RANK = 64
        /** LoRA の super adapter のファイル名の末尾（models/loras/<名前>.4g.bin） */
        const val SUFFIX = ".4g.bin"
        /** ひな型（models/ の default の super adapter。エンジンの Pipeline::DefaultAdapter と同じ） */
        const val DEFAULT_ADAPTER = "unet_lora_4g_default.bin"
        /** layout.json のグラフの順（縦 Base、縦タイル、横 Base、横タイル）。super adapter の末尾は 2 番目から順に並ぶ */
        private val GRAPHS = listOf("base", "tile", "land", "landtile")
        private const val OLD_HEADER = 18048   // グラフごとの（別 context 用の）adapter のヘッダー
        private const val QMAX = 127
        private const val QMAX_F = 127f

        private val SGM_IN = mapOf(4 to (1 to 0), 5 to (1 to 1), 7 to (2 to 0), 8 to (2 to 1))
        private val SGM_OUT = mapOf(0 to (0 to 0), 1 to (0 to 1), 2 to (0 to 2), 3 to (1 to 0), 4 to (1 to 1), 5 to (1 to 2))

        /** kohya（sgm）の UNet の名前を、diffusers の層の名前（'.' を '_' にしたもの）に直す。diffusers 形式は、そのまま返す。 */
        fun sgmToDiffusersStem(stem: String): String {
            Regex("input_blocks_(\\d+)_1_(.*)").matchEntire(stem)?.let { m ->
                SGM_IN[m.groupValues[1].toInt()]?.let { (b, i) -> return "down_blocks_${b}_attentions_${i}_${m.groupValues[2]}" }
            }
            Regex("middle_block_1_(.*)").matchEntire(stem)?.let { return "mid_block_attentions_0_${it.groupValues[1]}" }
            Regex("output_blocks_(\\d+)_1_(.*)").matchEntire(stem)?.let { m ->
                SGM_OUT[m.groupValues[1].toInt()]?.let { (b, i) -> return "up_blocks_${b}_attentions_${i}_${m.groupValues[2]}" }
            }
            return stem
        }

        /** 対称 8bit。0.5 は 0 から遠い側へ丸め、±127 を超えた分は切る。float32 のまま計算する。 */
        fun quant(x: Float, scale: Float): Byte {
            if (x == 0f) return 0
            val y = floor(abs(x) / scale + 0.5f)
            val q = if (y > QMAX) QMAX else y.toInt()
            return (if (x > 0f) q else -q).toByte()
        }

        /** 出力 256 本の塊を、ファイルに並んでいる順（論理番号）で返す。塊が 10 個以下は論理順。 */
        fun storedBlockOrder(n: Int): List<Int> {
            if (n <= 10) return (0 until n).toList()
            val seq = ArrayList<Int>()
            seq += 0
            for (i in 4 until n) seq += i
            for ((v, at) in listOf(1 to 7, 2 to 18, 3 to 29)) {
                if (v >= n) break
                seq.add(minOf(at, seq.size), v)
            }
            return seq
        }

        /** Conv の重み [out, in]（どちらも 32 の倍数）を、adapter に入っている並びに置く（loraconverter の pack_conv と同じ）。 */
        fun packConv(q: ByteArray, outC: Int, inC: Int): ByteArray {
            require(outC % 32 == 0 && inC % 32 == 0) { "チャネル数が 32 の倍数ではない: $outC x $inC" }
            val nChunk = (outC + 255) / 256
            val seq = storedBlockOrder(nChunk)
            val rowsOf = { l: Int -> minOf(256, outC - l * 256) }
            val dest = IntArray(nChunk)
            var acc = 0
            for (l in seq) { dest[l] = acc; acc += rowsOf(l) * inC }
            val out = ByteArray(outC * inC)
            for (l in 0 until nChunk) {
                val oc0 = l * 256
                val rows = rowsOf(l)
                var p = dest[l]
                for (ic in 0 until inC step 32) for (oc in 0 until rows step 32) for (cg in 0 until 32 step 4) for (r in 0 until 32) {
                    val base = (oc0 + oc + r) * inC + ic + cg
                    out[p++] = q[base]; out[p++] = q[base + 1]; out[p++] = q[base + 2]; out[p++] = q[base + 3]
                }
            }
            return out
        }

        fun rowSums(q: ByteArray, outC: Int, inC: Int): LongArray =
            LongArray(outC) { o -> var s = 0L; val b = o * inC; for (k in 0 until inC) s += q[b + k]; s }

        /** 名前は英数字で始める 64 文字以内（adapter のファイル名になる）。 */
        fun checkName(name: String) {
            val ok = name.isNotEmpty() && name[0].isLetterOrDigit() && name.all { it.isLetterOrDigit() || it in "_-." }
            require(ok && name.length <= 64) { "名前は英数字で始め、英数字と _-. だけの 64 文字以内にしてください: $name" }
        }

        /** 同じ名前の LoRA が既に outDir にあれば止める（上書きしない。二重に追加しない）。 */
        fun requireNew(outDir: File, name: String) {
            require(!File(outDir, name + SUFFIX).exists()) { "「$name」は既に追加されています（同じ名前の LoRA があります）。" }
        }

        /** Civitai のファイル名などから、adapter の名前にできる文字列を作る。 */
        fun safeName(s: String): String {
            val t = s.replace(Regex("[^A-Za-z0-9_.-]+"), "_").trim('_', '.', '-')
            return t.take(64).ifEmpty { "lora" }
        }

        /** LoRA の JSON の activation text。`\(` は `(` に戻す。 */
        fun cleanActivation(s: String): String = s.replace("\\(", "(").replace("\\)", ")")
    }
}
