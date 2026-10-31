package com.dragonspeedxl.app.engine

import android.content.Context
import java.io.File
import kotlin.concurrent.thread

/** root（su）で読む端末の生の値。GPU 使用率・クロック・温度、CPU 各コアの使用率と周波数、NPU（HVX/HMX）温度。 */
data class RawMetrics(
    val gpuBusy: Float,          // 0..1（端末全体の GPU 使用率）
    val gpuMhz: Float,
    val gpuTempC: Float,
    val hvxTempC: Float,
    val hmxTempC: Float,
    val cpuTempC: Float,
    val cpuUtil: List<Float>,    // コアごと 0..1
    val cpuMhz: List<Float>,
    val cpuBig: List<Boolean>,   // 各コアが最上位クラスタ（Prime = B）か。それ以外は L
)

/**
 * `su -c sh metrics_loop.sh` を常駐させ、出力を読み取って最新値を保持する（assets/metrics_loop.sh を filesDir に展開して使う）。
 * su が拒否された・出力が来ないときは available=false のままで、呼び出し側は通常の（root なしの）計測にフォールバックする。
 */
class RootMetrics(private val context: Context) {
    @Volatile var latest: RawMetrics? = null
        private set
    @Volatile var state: String = "未開始"
        private set
    private var proc: Process? = null

    /** 最大周波数が最も高いコア＝B、それ以外＝L。sysfs が読めなければ末尾 2 コアを B とみなす。 */
    private val bigCores: List<Boolean> by lazy {
        val n = Runtime.getRuntime().availableProcessors()
        val max = List(n) { i ->
            try { File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq").readText().trim().toLong() } catch (_: Exception) { -1L }
        }
        val top = max.maxOrNull() ?: -1L
        if (top <= 0L) List(n) { it >= n - 2 } else max.map { it == top }
    }
    @Volatile private var lastRxMs = 0L

    /** 最後の受信が 3 秒以内なら有効。 */
    val available: Boolean get() = latest != null && System.currentTimeMillis() - lastRxMs < 3000

    @Synchronized
    fun start(intervalSec: Double = 0.25) {
        if (proc != null) return
        try {
            val f = File(context.filesDir, "metrics_loop.sh")
            context.assets.open("metrics_loop.sh").use { i -> f.outputStream().use { o -> i.copyTo(o) } }
            val p = ProcessBuilder("su", "-c", "sh ${f.absolutePath} $intervalSec").redirectErrorStream(true).start()
            proc = p
            state = "su 起動中"
            thread(name = "root-metrics", isDaemon = true) { readLoop(p) }
        } catch (e: Exception) {
            state = "su を起動できません: ${e.message}"
        }
    }

    @Synchronized
    fun stop() {
        proc?.destroy()
        proc = null
        latest = null
        state = "停止"
    }

    private fun readLoop(p: Process) {
        val block = ArrayList<String>()
        var prev: List<Pair<Long, Long>>? = null
        try {
            p.inputStream.bufferedReader().forEachLine { line ->
                val l = line.trim()
                if (l != "E") { block.add(l); return@forEachLine }
                try {
                    val cpuLines = block.filter { it.startsWith("cpu") }
                    val d = block.filter { !it.startsWith("cpu") && it.isNotEmpty() }.associateBy { it.substring(0, 1) }
                    val cur = cpuLines.map { s ->
                        val f = s.split(Regex("\\s+")).drop(1).map { it.toLong() }
                        f.sum() to (f[3] + f.getOrElse(4) { 0L })     // 合計, idle+iowait
                    }
                    val util = prev?.let { pv -> cur.zip(pv) { (t1, i1), (t0, i0) -> (1f - (i1 - i0).toFloat() / (t1 - t0).coerceAtLeast(1)).coerceIn(0f, 1f) } }
                        ?: List(cur.size) { 0f }
                    prev = cur
                    val fr = d["F"]!!.split(' ').drop(1).filter { it.isNotEmpty() }.map { it.toFloat() / 1000f }
                    val g = d["G"]!!.split(' ')
                    val t = d["T"]!!.split(' ')
                    latest = RawMetrics(
                        gpuBusy = g[1].toFloat() / 100f, gpuMhz = g[2].toFloat() / 1e6f, gpuTempC = t[4].toFloat() / 1000f,
                        hvxTempC = t[1].toFloat() / 1000f, hmxTempC = t[2].toFloat() / 1000f, cpuTempC = t[3].toFloat() / 1000f,
                        cpuUtil = util, cpuMhz = fr, cpuBig = bigCores,
                    )
                    lastRxMs = System.currentTimeMillis()
                    state = "root 計測中"
                } catch (_: Exception) {
                }
                block.clear()
            }
        } catch (_: Exception) {
        }
        state = "su が終了しました（root の許可を確認してください）"
    }
}
