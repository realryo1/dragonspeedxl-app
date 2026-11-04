package com.dragonspeedxl.app.ui

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.layout.layout
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.material3.LocalMinimumInteractiveComponentSize
import androidx.compose.ui.unit.Dp
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.dragonspeedxl.app.MainViewModel
import com.dragonspeedxl.app.data.HistoryItem
import com.dragonspeedxl.app.engine.Orientation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

/**
 * メイン画面。配置と名前は姉妹プロジェクト SSXL の webui（index.html）に揃えつつ、部品は Material 3（Material You）。
 * 上から: ヘッダ（ブランド・CPU/MEM/NPU/GPU の要約）→ history（検索・ページ・グリッド）→ viewer（Job #…、プレビュー）
 *        → プロンプト編集 → gen-bar（解像度・seed/denoise/steps・コピー/流用・生成）→ statusline → 設定。
 */
@Composable
fun MainScreen(vm: MainViewModel) {
    // ロック中は、ロック画面だけを出す（中身は構成に入れない）
    if (vm.locked) {
        LockScreen(vm)
        return
    }
    Surface(color = MaterialTheme.colorScheme.background, contentColor = MaterialTheme.colorScheme.onBackground) {
        Column(
            Modifier
                .statusBarsPadding()
                .navigationBarsPadding()
                .imePadding()   // キーボードを出したとき、その分だけ表示領域を縮める（edge-to-edge では adjustResize だけでは持ち上がらない）。フォーカスした入力欄は、スクロールで見える位置に来る
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 12.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // ヘッダとモニターは 1 つのまとまりにする（畳んだときに外側の列の要素数＝余白の数が変わらないように）
            Column {
                Header(vm)
                MetricsPanel(vm)
            }
            History(vm)
            // プレビュー（Job 行まで）とプロンプト編集の間は、他の行間と同じ 8dp。プロンプト編集・gen-bar・生成ボタンも 8dp 間隔
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Viewer(vm)
                PromptEditor(vm)
                LoraPicker(vm)   // プロンプトと解像度（gen-bar）の間
                GenBar(vm)
                Status(vm)
            }
            Spacer(Modifier.height(20.dp))   // 設定は、生成まわりから少し離す
            SettingsButton(vm)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// header: ブランド + 要約（CPU / MEM / NPU / GPU。タップで詳細モニター）
@Composable
private fun Header(vm: MainViewModel) {
    // 要約の出入りで高さが変わるとガクつくので、高さを固定する
    Row(Modifier.fillMaxWidth().height(48.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("dragonspeedXL", style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.primary)
        // ロゴの右の空きの、右端に要約を寄せる（中央だと左に寄って見えるため）
        Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
            MetricsSummary(vm)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// LoRA の選択（SSXL の loraPicker に揃える: 折りたたみの見出し「▸ LoRA」・選んだ LoRA・固定の DMD、開くと検索つきの一覧）。
// いくつでも選べる（SSXL のチェックボックスと同じ。ComfyUI の LoraLoader を重ねるのと同じく足し算）。縦横どちらでも使える。
// 選んだ LoRA ごとの強さ w（既定 1.0）はスライダー。2 つ以上は、生成の前に端末で 1 つの adapter に合成する（MainViewModel.resolveLora）。
@Composable
private fun LoraPicker(vm: MainViewModel) {
    var open by remember { mutableStateOf(false) }
    var query by remember { mutableStateOf("") }
    var adding by remember { mutableStateOf(false) }   // 「LoRA を追加」のダイアログ
    val usable = vm.loraReady
    val clipboard = LocalClipboardManager.current
    val selected = vm.loraSelection
    // 選んだ LoRA の activation text（トリガーの文）を、選んだ順に「, 」でつないだもの
    val activation = selected.map { vm.loraActivation(it.name) }.filter { it.isNotEmpty() }.joinToString(", ")
    if (adding) LoraAddDialog(vm) { adding = false; vm.refreshLoras() }
    CompositionLocalProvider(LocalMinimumInteractiveComponentSize provides Dp.Unspecified) {
        // 背景は塗らない（プロンプト欄などと同じ。輪郭線だけ）
        Surface(shape = AppShape, color = Color.Transparent, border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline), modifier = Modifier.fillMaxWidth()) {
            Column {
                // 見出し行。タップで開閉する。開くたびに一覧を読み直す
                Row(
                    Modifier.fillMaxWidth().height(44.dp).clickable { open = !open; if (open) vm.refreshLoras() }.padding(horizontal = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Icon(if (open) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown, contentDescription = if (open) "LoRA の一覧を閉じる" else "LoRA の一覧を開く", modifier = Modifier.size(20.dp))
                    Text("LoRA", style = MaterialTheme.typography.titleSmall)
                    if (selected.isNotEmpty()) {
                        // 選んだ LoRA（UNet が無いときは薄く出す）
                        Text(
                            selected.joinToString(" + ") { "${it.name} ×${"%.1f".format(it.strength)}" },
                            style = MaterialTheme.typography.labelLarge,
                            color = if (usable) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f, fill = false),
                        )
                        // 選んだ LoRA の activation text をコピー（プロンプトに貼れる）
                        if (activation.isNotEmpty()) {
                            IconButton(onClick = { clipboard.setText(AnnotatedString(activation)) }, modifier = Modifier.size(32.dp)) {
                                Icon(AppIcons.ContentCopy, contentDescription = "選んだ LoRA のプロンプト（activation text）をコピー", modifier = Modifier.size(18.dp))
                            }
                        }
                    }
                    Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
                        Text("固定: DMD (weight 1.0)", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1)
                    }
                }
                AnimatedVisibility(
                    visible = open,
                    enter = expandVertically(tween(180, easing = FastOutSlowInEasing)) + fadeIn(tween(180)),
                    exit = shrinkVertically(tween(180, easing = FastOutSlowInEasing)) + fadeOut(tween(180)),
                ) {
                    Column(Modifier.padding(start = 8.dp, end = 8.dp, bottom = 8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        if (!vm.loraReady) {
                            Text("UNet（unet_lora_4g.bin と unet_lora_4g_default.bin）が models/ にありません。", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            // 追加（変換）は生成中に走らせない（LoRA の取り込みは busy のときに拒否する）
                            TextButton(onClick = { vm.clearLoraImportMessage(); adding = true }, enabled = !vm.busy) { Text("＋ LoRA を追加") }
                            Spacer(Modifier.weight(1f))
                            if (selected.isNotEmpty()) TextButton(onClick = { vm.clearLoras() }) { Text("すべて外す") }
                        }
                        if (vm.loraNames.isEmpty()) {
                            Text("models/loras/ に LoRA（<名前>.4g.bin）を置くか、上のボタンで追加すると選べます。", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        } else {
                            CompactTextField(query, { query = it }, "LoRAを検索…", true, Modifier.fillMaxWidth(), onClear = { query = "" })
                            val shown = vm.loraNames.filter { query.isBlank() || it.contains(query.trim(), ignoreCase = true) }
                            // 一覧は 4 行ぶんの高さで、超えた分は中でスクロール（タグ補完と同じ）。LoraRow は 40dp
                            Column(Modifier.fillMaxWidth().heightIn(max = 40.dp * 4).verticalScroll(rememberScrollState())) {
                                shown.forEach { n ->
                                    val act = vm.loraActivation(n)
                                    LoraRow(n, selected.any { it.name == n }, usable, if (act.isEmpty()) null else ({ clipboard.setText(AnnotatedString(act)) })) { vm.toggleLora(n) }
                                }
                            }
                            if (shown.isEmpty()) Text("該当する LoRA がありません。", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            // 選んだ LoRA ごとの強さ（選んだ順）
                            selected.forEach { u ->
                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Text(u.name, style = MaterialTheme.typography.bodyMedium, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.width(112.dp))
                                    Text("%.1f".format(u.strength), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.width(28.dp))
                                    Slider(
                                        value = u.strength,
                                        onValueChange = { vm.updateLoraStrength(u.name, (it * 10).roundToInt() / 10f) },
                                        valueRange = 0f..2f,
                                        steps = 19,
                                        modifier = Modifier.weight(1f),
                                    )
                                }
                            }
                            if (selected.count { it.strength != 0f } >= 2) {
                                Text(
                                    "2 つ以上は、生成の前に 1 つに合成します（組み合わせや強さを変えた最初の 1 回は、数十秒かかります）。rank の合計が 64 を超えると、上位 64 成分に近似します。",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/** LoRA の一覧の 1 行（チェックボックス。いくつでも選べる）。onCopy があれば、末尾にその LoRA の activation text をコピーするボタンを出す（選べない状態でも押せる）。 */
@Composable
private fun LoraRow(label: String, selected: Boolean, enabled: Boolean, onCopy: (() -> Unit)?, onToggle: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().height(40.dp).clip(AppShape).clickable(enabled = enabled, onClick = onToggle),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Checkbox(checked = selected, onCheckedChange = null, enabled = enabled, modifier = Modifier.padding(start = 4.dp))
        Text(label, style = MaterialTheme.typography.bodyMedium, maxLines = 1, color = if (enabled) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f), modifier = Modifier.weight(1f))
        if (onCopy != null) {
            IconButton(onClick = onCopy, modifier = Modifier.size(36.dp)) {
                Icon(AppIcons.ContentCopy, contentDescription = "$label のプロンプト（activation text）をコピー", modifier = Modifier.size(18.dp))
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// gen-bar: 解像度 / seed・denoise・steps / コピー・流用 / 生成
@Composable
private fun GenBar(vm: MainViewModel) {
    // ボタン・チップの押せる範囲の下限（48dp）で、上下に余白が足されないようにする（行間を 8dp に揃える）
    CompositionLocalProvider(LocalMinimumInteractiveComponentSize provides Dp.Unspecified) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // 解像度は 2 択（横 1280×720 / 縦 720×1280）。Hires は ×1.5 固定
        Column(verticalArrangement = Arrangement.spacedBy(0.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Orientation.entries.forEach { o ->
                    FilterChip(shape = AppShape, 
                        selected = vm.orientation == o,
                        onClick = { vm.orientation = o },
                        label = { Text(o.label, maxLines = 1, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth()) },
                        modifier = Modifier.weight(1f).height(40.dp),
                    )
                }
            }
            val o = vm.orientation
            Text(
                "HiresFixは${vm.hiresScale}倍固定。${(o.baseW * vm.hiresScale).toInt()}x${(o.baseH * vm.hiresScale).toInt()}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 4.dp, top = 4.dp),
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            // 生成中も変えられる（走っているジョブは開始時の値で動くので、次の生成に向けて書き換えるだけ）
            ClearableField(vm.seed, { vm.seed = it }, "seed（ランダム）", KeyboardType.Number, true, Modifier.weight(1f))
            ClearableField(vm.hiresDenoise, { vm.hiresDenoise = it }, "denoise（0.39）", KeyboardType.Decimal, true, Modifier.weight(1f))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ClearableField(vm.baseSteps, { vm.baseSteps = it }, "base steps（2）", KeyboardType.Number, true, Modifier.weight(1f))
            ClearableField(vm.hiresSteps, { vm.hiresSteps = it }, "hires steps（4）", KeyboardType.Number, vm.hiresAvailable, Modifier.weight(1f))
        }
        if (!vm.hiresAvailable) {
            Text("UNet（unet_lora_4g.bin）が未配置のため生成できません", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        if (vm.running) {
            Button(shape = AppShape, onClick = { vm.cancel() }, modifier = Modifier.fillMaxWidth().height(52.dp)) { Text("キャンセル") }
        } else {
            Button(shape = AppShape,
                onClick = { vm.generate() },
                enabled = vm.prompt.isNotBlank() && !vm.busy,   // VAE 中も押せない（最終画像が上がってから次の生成）
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary, contentColor = MaterialTheme.colorScheme.onSecondary),
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) { Text("生成", fontWeight = FontWeight.Bold) }
        }
    }
    }
}

/** SSXL の clearable-field。空欄なら既定値（placeholder に表示）。末尾の × で消す。 */
@Composable
private fun ClearableField(value: String, onChange: (String) -> Unit, placeholder: String, keyboard: KeyboardType, enabled: Boolean, modifier: Modifier = Modifier) {
    CompactTextField(
        value = value,
        onValueChange = { v -> onChange(v.filter { it.isDigit() || it == '.' }.take(12)) },
        placeholder = placeholder,
        enabled = enabled,
        keyboardOptions = KeyboardOptions(keyboardType = keyboard),
        onClear = { onChange("") },
        modifier = modifier,
    )
}

/**
 * 高さ 40dp（コピー・流用ボタンと同じ）の 1 行入力欄。標準の OutlinedTextField は 56dp が下限なので、BasicTextField で組む。
 * onClear を渡すと、入力があるとき末尾に × を出す。
 */
@OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
internal fun CompactTextField(
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    keyboardOptions: KeyboardOptions = KeyboardOptions.Default,
    keyboardActions: KeyboardActions = KeyboardActions.Default,
    onClear: (() -> Unit)? = null,
    leadingIcon: (@Composable () -> Unit)? = null,
) {
    val source = remember { MutableInteractionSource() }
    val colors = OutlinedTextFieldDefaults.colors()
    val textColor = if (enabled) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f)
    BasicTextField(
        value = value,
        onValueChange = onValueChange,
        enabled = enabled,
        singleLine = true,
        textStyle = MaterialTheme.typography.bodyMedium.copy(color = textColor),
        cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
        keyboardOptions = keyboardOptions,
        keyboardActions = keyboardActions,
        interactionSource = source,
        modifier = modifier.height(40.dp),
        decorationBox = { inner ->
            OutlinedTextFieldDefaults.DecorationBox(
                value = value,
                innerTextField = inner,
                enabled = enabled,
                singleLine = true,
                visualTransformation = VisualTransformation.None,
                interactionSource = source,
                placeholder = { Text(placeholder, style = MaterialTheme.typography.bodyMedium) },
                leadingIcon = leadingIcon,
                trailingIcon = if (onClear != null && value.isNotEmpty() && enabled) {
                    {
                        Icon(
                            Icons.Filled.Close, contentDescription = "クリア",
                            modifier = Modifier.size(24.dp).clip(CircleShape).clickable { onClear() }.padding(3.dp),
                        )
                    }
                } else null,
                colors = colors,
                contentPadding = OutlinedTextFieldDefaults.contentPadding(top = 0.dp, bottom = 0.dp),
                container = { OutlinedTextFieldDefaults.Container(enabled = enabled, isError = false, interactionSource = source, colors = colors, shape = AppShape) },
            )
        },
    )
}

// ---------------------------------------------------------------------------------------------
// error（statusline = Progress は、モニターの GPU 行の下に移した）
@Composable
private fun Status(vm: MainViewModel) {
    vm.error?.let { Text(it, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.error) }
}

// ---------------------------------------------------------------------------------------------
// viewer: プレビュー（全幅）→ Job #… [BASE|HIRES] ・ コピー ・ 流用 ・ ダウンロード
@Composable
private fun Viewer(vm: MainViewModel) {
    val item = vm.activeItem
    val clipboard = LocalClipboardManager.current
    var fullscreen by remember { mutableStateOf(false) }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // 生成中の途中経過（TAESD）は、それを作っているジョブの向きの絵（次の生成の向きを変えても枠は動かさない）。無ければ表示中のジョブの向き
        val o = if (vm.livePreview != null) vm.previewOrientation else item?.orientation ?: vm.orientation
        // 画像は画面の左右いっぱい（外側の余白を打ち消す）。縦向きだけは細くする
        Box(
            Modifier
                // 縦向きは、以前の幅（左右 48dp の余白）の 2/3 にして、中央に置く（縦横比は固定なので、高さも 2/3 になる）
                .let { if (o == Orientation.PORTRAIT) it.align(Alignment.CenterHorizontally).padding(horizontal = 48.dp).fillMaxWidth(2f / 3f) else it.bleedHorizontally(12.dp) }
                .aspectRatio(o.baseW.toFloat() / o.baseH)
                .background(MaterialTheme.colorScheme.surface),
            contentAlignment = Alignment.Center,
        ) {
            val bmp = vm.displayBitmap
            if (bmp != null) {
                Image(bmp.asImageBitmap(), contentDescription = "生成画像", contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize().clickable { fullscreen = true })
            } else {
                Text("まだ生成がありません", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            val p = vm.activeProgress
            if (vm.running || vm.decodingJobs > 0) {
                LinearProgressIndicator(
                    progress = { if (p == null || p.totalSteps == 0) 0f else p.step.toFloat() / p.totalSteps },
                    modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth(),
                )
            }
        }
        // TAESD が無いと、生成中のプレビューと高速デコードは使えない（生成は VAE で動く）。黙って出ないので、理由を出す
        if (vm.previewBlocked) {
            Text(
                "${vm.taesdMissingFile} が無いため、プレビューと高速デコードは使えません（VAE で仕上げます）",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
            )
        }
        CompositionLocalProvider(LocalMinimumInteractiveComponentSize provides Dp.Unspecified) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (item == null) {
                    Text("Job", style = MaterialTheme.typography.titleMedium)
                } else {
                    Text("Job #${item.jobId}", style = MaterialTheme.typography.titleMedium, maxLines = 1)
                    Badge(item.kind)
                    if (item.variant.isNotEmpty()) Badge(item.variant.trimStart('_'))
                }
                Spacer(Modifier.weight(1f))
                OutlinedButton(shape = AppShape, onClick = { clipboard.setText(AnnotatedString(vm.prompt)) }, contentPadding = PaddingValues(horizontal = 12.dp)) { Text("コピー") }
                OutlinedButton(shape = AppShape, onClick = { vm.useAsBase() }, enabled = item != null && !vm.busy, contentPadding = PaddingValues(horizontal = 12.dp)) { Text("↙流用") }
                IconButton(onClick = { vm.saveToGallery() }, enabled = vm.previewBitmap != null && !vm.busy) {
                    Icon(AppIcons.Download, contentDescription = "ギャラリーに保存")
                }
            }
        }
    }
    if (fullscreen) vm.displayBitmap?.let { FullscreenPreview(it) { fullscreen = false } }
}

/** 親の左右の余白（padding）を打ち消して、画面の端まで広げる。 */
private fun Modifier.bleedHorizontally(padding: Dp): Modifier = layout { m, c ->
    val extra = padding.roundToPx() * 2
    val p = m.measure(c.copy(minWidth = c.maxWidth + extra, maxWidth = c.maxWidth + extra))
    layout(c.maxWidth, p.height) { p.place(-extra / 2, 0) }
}

@Composable
private fun Badge(kind: String) {
    val hires = kind == "HIRES"
    Text(
        kind,
        style = MaterialTheme.typography.labelSmall,
        fontWeight = FontWeight.Bold,
        color = if (hires) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface,
        modifier = Modifier
            .clip(AppShape)
            .background(if (hires) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant)
            .padding(horizontal = 6.dp, vertical = 2.dp),
    )
}

/**
 * 全画面表示。ピンチで拡大・縮小（1〜8 倍）、拡大中はドラッグで移動、ダブルタップで 1 倍 ⇄ 3 倍。等倍のときにタップすると閉じる（拡大中のタップでは閉じない）。
 * 画像は等倍で画面に収まる大きさ（Fit）に置き、その上に拡大・移動をかける。移動は、拡大した画像が画面の外へ出きらない範囲に収める。
 */
@Composable
private fun FullscreenPreview(bitmap: Bitmap, onClose: () -> Unit) {
    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        var scale by remember { mutableFloatStateOf(1f) }
        var offset by remember { mutableStateOf(Offset.Zero) }
        var box by remember { mutableStateOf(IntSize.Zero) }
        // 等倍（Fit）で画像が画面に占める大きさ
        fun fitted(): Size {
            if (box.width == 0 || box.height == 0) return Size.Zero
            val bw = bitmap.width.toFloat(); val bh = bitmap.height.toFloat()
            val cw = box.width.toFloat(); val ch = box.height.toFloat()
            return if (bw / bh > cw / ch) Size(cw, cw * bh / bw) else Size(ch * bw / bh, ch)
        }
        // 移動の範囲: 拡大した画像が画面より大きい分だけ（小さい軸は動かさない）
        fun clamp(o: Offset, s: Float): Offset {
            val f = fitted()
            val mx = maxOf(0f, (f.width * s - box.width) / 2f); val my = maxOf(0f, (f.height * s - box.height) / 2f)
            return Offset(o.x.coerceIn(-mx, mx), o.y.coerceIn(-my, my))
        }
        Box(
            Modifier
                .fillMaxSize()
                .background(Color.Black)
                .onSizeChanged { box = it }
                .pointerInput(Unit) {
                    detectTapGestures(
                        onTap = { if (scale <= 1.01f) onClose() },
                        onDoubleTap = { p ->
                            if (scale > 1.01f) { scale = 1f; offset = Offset.Zero }
                            else {   // タップした点を中心に 3 倍
                                val c = Offset(box.width / 2f, box.height / 2f)
                                scale = 3f
                                offset = clamp((c - p) * (3f - 1f), 3f)
                            }
                        },
                    )
                }
                .pointerInput(Unit) {
                    detectTransformGestures { centroid, pan, zoom, _ ->
                        val c = Offset(box.width / 2f, box.height / 2f)
                        val s = (scale * zoom).coerceIn(1f, 8f)
                        val z = s / scale
                        // 指の下の点を動かさないように拡大し、そのうえで指の移動分だけ動かす
                        val rel = centroid - c
                        scale = s
                        offset = clamp(rel + pan - (rel - offset) * z, s)
                    }
                },
            contentAlignment = Alignment.Center,
        ) {
            Image(
                bitmap.asImageBitmap(), contentDescription = "生成画像（全画面。ピンチで拡大）", contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize().graphicsLayer(scaleX = scale, scaleY = scale, translationX = offset.x, translationY = offset.y),
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------
// history: 検索 ・ 🎲 🗑️ ⬅ ➡ ・ 設定の列数のグリッド
@Composable
private fun History(vm: MainViewModel) {
    Column {
        // 見出し（タップで開閉）。畳んでも見出しだけは残る
        Row(
            Modifier.fillMaxWidth().clip(AppShape).clickable { vm.toggleHistoryExpanded() }.padding(horizontal = 4.dp).height(40.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("履歴", style = MaterialTheme.typography.titleMedium)
            Text("  ${vm.history.size}件", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
            Icon(
                if (vm.historyExpanded) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                contentDescription = if (vm.historyExpanded) "履歴を畳む" else "履歴を開く",
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        AnimatedVisibility(
            visible = vm.historyExpanded,
            enter = expandVertically(tween(250, easing = FastOutSlowInEasing)) + fadeIn(tween(250)),
            exit = shrinkVertically(tween(250, easing = FastOutSlowInEasing)) + fadeOut(tween(150)),
        ) {
            HistoryBody(vm)
        }
    }
}

@Composable
private fun HistoryBody(vm: MainViewModel) {
    Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // seed・タグの入力欄と同じ 40dp の入力欄（標準の OutlinedTextField は 56dp が下限）
        CompactTextField(
            value = vm.historySearch,
            onValueChange = { vm.updateHistorySearch(it) },
            placeholder = "履歴検索（-語句で除外）",
            enabled = true,
            modifier = Modifier.fillMaxWidth(),
            onClear = { vm.updateHistorySearch("") },
            leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null, modifier = Modifier.size(20.dp)) },
        )
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
            PagerButton("🎲", enabled = vm.history.isNotEmpty()) { vm.historyRandom() }
            PagerButton(if (vm.historyDeleteMode) "🔙" else "🗑️") { vm.toggleHistoryDeleteMode() }
            if (vm.historyDeleteMode && vm.historySelectedIds.isNotEmpty()) {
                Button(shape = AppShape, onClick = { vm.deleteSelected() }, colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error, contentColor = MaterialTheme.colorScheme.onError)) {
                    Text("${vm.historySelectedIds.size}件削除")
                }
            }
            if (vm.historyPageCount > 1 || vm.historyPage > 1) {
                // ランダム表示中も有効。押すとランダムを解除し、表示中のページの前後へ移る（historyPrev / historyNext）
                PagerButton("⬅", enabled = vm.historyPage > 1) { vm.historyPrev() }
                PagerButton("➡", enabled = vm.historyPage < vm.historyPageCount) { vm.historyNext() }
                Text("${vm.historyPage} / ${vm.historyPageCount}", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        val items = vm.historyPageItems
        if (items.isEmpty()) {
            Text(if (vm.history.isEmpty()) "履歴はまだありません" else "該当する履歴がありません", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        // SSXL と同じ、隙間なしの正方形タイル。列数は設定の「横」。画面の左右いっぱいまで敷き詰める（外側の余白を打ち消す）
        val cols = vm.historyCols
        Column(Modifier.bleedHorizontally(12.dp)) {
            items.chunked(cols).forEach { row ->
                Row(Modifier.fillMaxWidth()) {
                    row.forEach { HistoryCell(vm, it, Modifier.weight(1f)) }
                    repeat(cols - row.size) { Spacer(Modifier.weight(1f)) }
                }
            }
        }
    }
}

@Composable
private fun PagerButton(label: String, enabled: Boolean = true, onClick: () -> Unit) {
    OutlinedButton(shape = AppShape, onClick = onClick, enabled = enabled, contentPadding = PaddingValues(horizontal = 14.dp), modifier = Modifier.height(40.dp)) {
        Text(label, fontSize = 16.sp)
    }
}

/** 履歴のカード。左上にバッジ（BASE / HIRES）、左下に #ジョブ ID。 */
@Composable
private fun HistoryCell(vm: MainViewModel, item: HistoryItem, modifier: Modifier = Modifier) {
    val selected = item.id == vm.activeId
    val deleting = vm.historyDeleteMode
    val checked = item.id in vm.historySelectedIds
    val thumb by produceState<Bitmap?>(null, item.id) { value = withContext(Dispatchers.IO) { vm.thumbnail(item) } }
    Box(
        modifier
            .aspectRatio(1f)
            .background(MaterialTheme.colorScheme.surface)
            .clickable { if (deleting) vm.toggleHistorySelected(item.id) else vm.selectHistory(item) },
    ) {
        thumb?.let { Image(it.asImageBitmap(), contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
        CellLabel(item.tileLabel, Modifier.align(Alignment.TopStart).padding(2.dp), strong = item.hires)
        CellLabel("#${item.jobId}", Modifier.align(Alignment.BottomStart).padding(2.dp))
        if (selected && !deleting) Box(Modifier.fillMaxSize().border(3.dp, MaterialTheme.colorScheme.primary))
        if (deleting) {
            Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = if (checked) 0.5f else 0.15f)))
            Checkbox(checked = checked, onCheckedChange = { vm.toggleHistorySelected(item.id) }, modifier = Modifier.align(Alignment.TopEnd).size(32.dp))
        }
    }
}

@Composable
private fun CellLabel(text: String, modifier: Modifier = Modifier, strong: Boolean = false) {
    Text(
        text,
        fontSize = 9.sp,
        fontWeight = if (strong) FontWeight.Bold else FontWeight.Normal,
        color = Color.White,
        maxLines = 1,
        modifier = modifier.clip(AppShape).background(Color.Black.copy(alpha = 0.55f)).padding(horizontal = 3.dp, vertical = 1.dp),
    )
}

// ---------------------------------------------------------------------------------------------
/** 履歴の下の設定ボタン。押すと設定（セキュリティ、起動時のプロンプト、履歴、性能）を開く。 */
@Composable
private fun SettingsButton(vm: MainViewModel) {
    FilledTonalButton(shape = AppShape, onClick = { vm.showSettings = true }, modifier = Modifier.fillMaxWidth()) {
        Icon(Icons.Filled.Settings, contentDescription = null, modifier = Modifier.size(18.dp))
        Text("  設定")
    }
    if (vm.showSettings) SettingsDialog(vm, onDismiss = { vm.showSettings = false })
}
