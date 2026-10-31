package com.dragonspeedxl.app.engine

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import android.system.Os
import android.system.OsConstants
import java.io.File
import kotlin.math.abs

/** モニターのグラフが表す時間の幅（ミリ秒）。ポーリングの間隔（生成中 0.25 秒、待機中 1 秒）によらず、グラフの横軸はこの時間に比例する。 */
const val MetricsWindowMs = 20_000L

/** ある瞬間の端末の負荷。NPU・GPU の使用率は「このアプリのエンジンが実行していた時間の割合」（端末全体ではない）。 */
data class Metrics(
    val npu: Float,          // 0..1
    val gpu: Float,          // 0..1
    val cpuCores: Float,     // このアプリが使っているコア数（8 コア中）
    val cpuTotalCores: Int,
    val memUsedGb: Float,    // 端末全体の使用量（合計 - 利用可能）
    val memTotalGb: Float,
    val battTempC: Float,
    val watts: Float,        // バッテリーの電流×電圧（充電中は入力が混ざるので目安）
    val charging: Boolean,
    val thermalStatus: Int,  // PowerManager.THERMAL_STATUS_*（0=なし … 3=重度 以上）
    val raw: RawMetrics? = null,   // root で読めたときの端末の生の値
    val rootState: String = "",
    val timeMs: Long = 0L,   // このサンプルを取った時刻（SystemClock.elapsedRealtime）。グラフの横軸を時間に比例させるため
)

/**
 * root なしで読める情報だけでタスクマネージャー風の値を作る。
 * NPU / GPU は NativeEngine が数える「graphExecute の累積時間」の差分から使用率を出す。
 * CPU はこのプロセスの /proc/self/stat、メモリは ActivityManager、電池・温度・電力は BatteryManager。
 */
class DeviceMetrics(private val context: Context, private val busy: () -> LongArray?, private val root: RootMetrics? = null) {
    private val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    private val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val tick = Os.sysconf(OsConstants._SC_CLK_TCK).toDouble().coerceAtLeast(1.0)
    private val cores = Runtime.getRuntime().availableProcessors()

    private var prevWall = 0L
    private var prevCpuTicks = -1L
    private var prevBusy: LongArray? = null

    private fun processCpuTicks(): Long = try {
        // utime(14) + stime(15)。コマンド名に空白が入りうるので最後の ')' 以降を数える
        val s = File("/proc/self/stat").readText()
        val f = s.substring(s.lastIndexOf(')') + 2).split(' ')
        f[11].toLong() + f[12].toLong()
    } catch (_: Exception) { -1L }

    fun sample(): Metrics {
        val wall = System.nanoTime()
        val dt = (wall - prevWall).coerceAtLeast(1L).toDouble() / 1e9
        val first = prevWall == 0L

        val ticks = processCpuTicks()
        val cpu = if (first || prevCpuTicks < 0 || ticks < 0) 0f else ((ticks - prevCpuTicks) / tick / dt).toFloat()
        prevCpuTicks = ticks

        var npu = 0f
        var gpu = 0f
        val b = busy()
        val pb = prevBusy
        if (b != null && pb != null && !first) {
            val span = (b[2] - pb[2]).coerceAtLeast(1L).toDouble()
            npu = ((b[0] - pb[0]) / span).toFloat().coerceIn(0f, 1f)
            gpu = ((b[1] - pb[1]) / span).toFloat().coerceIn(0f, 1f)
        }
        prevBusy = b
        prevWall = wall

        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val gb = 1024.0 * 1024.0 * 1024.0
        val batt: Intent? = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tempC = (batt?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10f
        val voltMv = batt?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
        val plugged = (batt?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val curUa = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val watts = (abs(curUa) / 1e6 * voltMv / 1e3).toFloat()
        return Metrics(
            npu, gpu, cpu.coerceAtLeast(0f), cores, ((mi.totalMem - mi.availMem) / gb).toFloat(), (mi.totalMem / gb).toFloat(),
            tempC, watts, plugged, pm.currentThermalStatus,
            raw = if (root?.available == true) root.latest else null, rootState = root?.state ?: "",
            timeMs = android.os.SystemClock.elapsedRealtime(),
        )
    }
}
