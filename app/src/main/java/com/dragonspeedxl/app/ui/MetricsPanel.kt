package com.dragonspeedxl.app.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandHorizontally
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkHorizontally
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.dragonspeedxl.app.MainViewModel
import com.dragonspeedxl.app.engine.Metrics
import com.dragonspeedxl.app.engine.MetricsWindowMs
import com.dragonspeedxl.app.engine.RawMetrics

private val LabelWidth = 52.dp

// 開閉アニメーションは既定のばね（揺れ・終端のカクつきが出やすい）ではなく、一定時間のなめらかな動きにそろえる
private const val AnimMs = 260
// 要約は横幅を変えず（毎フレームの再レイアウトを避け）フェードだけで、パネルが縮み始めてから現れる。
// パネルは縮みながら先にフェードアウトさせる。そうしないと、縮み終わり際の背景色の薄い切れ端が
// 画像エリアの上枠のすぐ上に残り、枠が二重に見える。
private val PanelEnter = fadeIn(tween(AnimMs)) + expandVertically(tween(AnimMs, easing = FastOutSlowInEasing), expandFrom = Alignment.Top)
private val PanelExit = shrinkVertically(tween(AnimMs, easing = FastOutSlowInEasing), shrinkTowards = Alignment.Top) + fadeOut(tween(AnimMs / 2))

/**
 * ヘッダのロゴの右に出す要約（CPU / MEM / NPU / GPU を、ラベルの下に値の 2 段で横に並べる）。
 * タップで詳細パネル（MetricsPanel）を開閉する。
 */
@Composable
fun MetricsSummary(vm: MainViewModel) {
    val m = vm.metrics ?: return
    val r = m.raw
    val cpu = r?.cpuUtil?.takeIf { it.isNotEmpty() }?.average()?.toFloat() ?: (m.cpuCores / m.cpuTotalCores).coerceIn(0f, 1f)
    val gpu = r?.gpuBusy ?: m.gpu
    Row(
        Modifier.clip(AppShape).clickable { vm.monitorExpanded = !vm.monitorExpanded }.padding(horizontal = 8.dp, vertical = 2.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        SummaryItem("CPU", "${pct(cpu)}%", sc.cpu)
        SummaryItem("MEM", "${"%.1f".format(m.memUsedGb)}G", memColor(m, sc))
        SummaryItem("NPU", "${pct(m.npu)}%", sc.npu)
        SummaryItem("GPU", "${pct(gpu)}%", sc.gpu)
        Icon(
            if (vm.monitorExpanded) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
            contentDescription = if (vm.monitorExpanded) "モニターを畳む" else "モニターを展開",
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun SummaryItem(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = color)
        Text(value, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
    }
}

/** 詳細モニター。ロゴ横の要約をタップすると開く。見出しをタップすると畳まれる。 */
@Composable
fun MetricsPanel(vm: MainViewModel) {
    val m = vm.metrics
    val h = vm.metricsHistory
    val r = m?.raw
    AnimatedVisibility(visible = vm.monitorExpanded, enter = PanelEnter, exit = PanelExit) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(top = 12.dp)   // タイトルとの間隔。パネルの内側に持たせ、開閉のアニメーションと一緒に伸び縮みさせる
                .clip(AppShape)
                .background(MaterialTheme.colorScheme.surface)
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Row(Modifier.fillMaxWidth().clickable { vm.monitorExpanded = false }, verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (m == null) "モニター" else if (r != null) "モニター（root・生の値）" else "モニター（root なし: ${m.rootState}）",
                    style = MaterialTheme.typography.titleSmall,
                    modifier = Modifier.weight(1f),
                )
                Icon(Icons.Filled.KeyboardArrowUp, contentDescription = "モニターを畳む", tint = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (m != null) MetricsBody(m, r, h)
            MonitorStatus(vm)   // GPU の行の下に、エンジンの状態・モデルの状態とボタン・Progress をまとめる
        }
    }
}

/**
 * モニターの GPU 行の下にまとめる状態表示（以前は画面のあちこちにあった）:
 *  1. エンジンの状態（"native v0.6 …": QNN ライブラリ・モデル配置・UNet の組み合わせ）
 *  2. モデルの常駐状態と、[ロード]（今の向きのモデルを全部読み込む）・[解放]（常駐モデルを全部手放す）
 *  3. Progress（生成・VAE・ロードの進み具合）
 * エラーは見落とさないよう、これまでどおり生成ボタンの下に出す。
 */
@Composable
private fun MonitorStatus(vm: MainViewModel) {
    Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(vm.nativeStatus, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(vm.modelStateText, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
            OutlinedButton(shape = AppShape, onClick = { vm.loadModels() }, enabled = vm.modelIdle, contentPadding = PaddingValues(horizontal = 14.dp), modifier = Modifier.height(40.dp)) { Text("ロード") }
            OutlinedButton(shape = AppShape, onClick = { vm.unloadModels() }, enabled = vm.modelIdle && vm.anyModelLoaded, contentPadding = PaddingValues(horizontal = 14.dp), modifier = Modifier.height(40.dp)) { Text("解放") }
        }
        Text(vm.statusline, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun MetricsBody(m: Metrics, r: RawMetrics?, h: List<Metrics>) {
    val t = h.map { it.timeMs }   // 各サンプルの時刻（グラフの横軸を時間に比例させる）
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        // CPU（全体の推移 + コア別。コアの裏に B=Prime / L=Performance）
        if (r != null) {
            val avg = if (r.cpuUtil.isEmpty()) 0f else r.cpuUtil.average().toFloat()
            Graph("CPU", avg, h.map { x -> x.raw?.cpuUtil?.takeIf { it.isNotEmpty() }?.average()?.toFloat() ?: 0f }, t, sc.cpu)
            CoreBars(r)
            Info("${pct(avg)}%", "CPU 最高 %.1f℃   このアプリ %.1f / %d コア".format(r.cpuTempC, m.cpuCores, m.cpuTotalCores))
        } else {
            val u = (m.cpuCores / m.cpuTotalCores).coerceIn(0f, 1f)
            Graph("CPU", u, h.map { (it.cpuCores / it.cpuTotalCores).coerceIn(0f, 1f) }, t, sc.cpu)
            Info("${pct(u)}%", "このアプリ %.1f / %d コア".format(m.cpuCores, m.cpuTotalCores))
        }

        // メモリ（+ 電池・電力・熱状態）
        Graph("MEM", (m.memUsedGb / m.memTotalGb).coerceIn(0f, 1f), h.map { (it.memUsedGb / it.memTotalGb).coerceIn(0f, 1f) }, t, memColor(m, sc))
        Info(
            "%.1f / %.1f GB".format(m.memUsedGb, m.memTotalGb),
            "電池 %.1f℃   電力 %.1f W%s%s".format(
                m.battTempC, m.watts, if (m.charging) "（充電中は参考）" else "",
                if (m.thermalStatus > 0) "   熱状態 ${thermalLabel(m.thermalStatus)}" else "",
            ),
        )

        // NPU（アプリ計測の稼働率 + HVX/HMX 温度）
        Graph("NPU", m.npu, h.map { it.npu }, t, sc.npu)
        Info("${pct(m.npu)}%", "アプリ計測の稼働率" + (r?.let { "   HVX %.1f℃   HMX %.1f℃".format(it.hvxTempC, it.hmxTempC) } ?: ""))

        // GPU
        if (r != null) {
            Graph("GPU", r.gpuBusy, h.map { it.raw?.gpuBusy ?: 0f }, t, sc.gpu)
            Info("${pct(r.gpuBusy)}%", "%.0f MHz   %.1f℃".format(r.gpuMhz, r.gpuTempC))
        } else {
            Graph("GPU", m.gpu, h.map { it.gpu }, t, sc.gpu)
            Info("${pct(m.gpu)}%", "アプリ計測の稼働率")
        }
    }
}

/** モニターの系列色。パレット（壁紙由来）から取る。GPU だけは区別しやすいよう固定の琥珀色。 */
private class Series(val cpu: Color, val mem: Color, val npu: Color, val gpu: Color, val warn: Color)

private val sc = Series(Color(0xFF7C9CF5), Color(0xFF9B7FD1), Color(0xFF35C4B8), Color(0xFFE8A33D), Color(0xFFE0566A))   // テーマに依らない固定のカラフルな配色

private fun pct(v: Float) = (v * 100).toInt()

private fun memColor(m: Metrics, sc: Series) = if (m.memUsedGb / m.memTotalGb > 0.92f) sc.warn else sc.mem

private fun thermalLabel(s: Int) = when {
    s <= 0 -> "正常"
    s == 1 -> "軽度"
    s == 2 -> "中度"
    s == 3 -> "重度"
    else -> "危険"
}

/**
 * 左にラベル、右は端までのグラフ（現在値の塗りと直近の推移の折れ線）。
 * 折れ線の横軸は時間に比例する（右端が最新のサンプルの時刻で、左端が MetricsWindowMs 前）。ポーリングの間隔が変わっても、1 回の更新での動きは経過時間に比例する。
 */
@Composable
private fun Graph(label: String, value: Float, history: List<Float>, times: List<Long>, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.labelMedium, textAlign = TextAlign.Center, modifier = Modifier.width(LabelWidth))
        Canvas(Modifier.weight(1f).height(32.dp).clip(AppShape).background(color.copy(alpha = 0.10f))) {
            val v = value.coerceIn(0f, 1f)
            drawRect(color.copy(alpha = 0.28f), topLeft = Offset(0f, size.height * (1f - v)), size = Size(size.width, size.height * v))
            if (history.size > 1) {
                val p = Path()
                val tLast = times.last()
                history.forEachIndexed { i, x ->
                    val px = size.width * (1f - (tLast - times[i]).toFloat() / MetricsWindowMs)
                    val py = size.height * (1f - x.coerceIn(0f, 1f))
                    if (i == 0) p.moveTo(px, py) else p.lineTo(px, py)
                }
                drawPath(p, color, style = Stroke(width = 2.dp.toPx()))
            }
        }
    }
}

/** グラフの下の数値行。先頭の値（％など）を強調する。 */
@Composable
private fun Info(main: String, rest: String) {
    Row(Modifier.fillMaxWidth().padding(start = LabelWidth, top = 6.dp, bottom = 10.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(main, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
        Text(rest, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** CPU の各コアの使用率（縦棒）と周波数。四角の裏にうっすら B（Prime）/ L（Performance）。 */
@Composable
private fun CoreBars(r: RawMetrics) {
    Row(Modifier.fillMaxWidth().padding(start = LabelWidth, top = 4.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        r.cpuUtil.forEachIndexed { i, u ->
            Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                val big = r.cpuBig.getOrElse(i) { false }
                Box(
                    Modifier.fillMaxWidth().height(36.dp).clip(AppShape).background(sc.cpu.copy(alpha = 0.10f)),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(if (big) "B" else "L", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.16f))
                    Canvas(Modifier.fillMaxSize()) {
                        val hh = size.height * u.coerceIn(0f, 1f)
                        drawRect(sc.cpu.copy(alpha = 0.85f), topLeft = Offset(0f, size.height - hh), size = Size(size.width, hh))
                    }
                }
                Text("${r.cpuMhz.getOrElse(i) { 0f }.toInt()}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}
