package com.dragonspeedxl.app.ui

import android.os.SystemClock
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.VectorConverter
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.HorizontalDivider
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.layout.layoutId
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInParent
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.dragonspeedxl.app.MainViewModel
import com.dragonspeedxl.app.prompt.DanbooruClient
import com.dragonspeedxl.app.prompt.PromptChip
import com.dragonspeedxl.app.prompt.PromptChips
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * プロンプト編集。SSXL の prompt-editor（チップ編集 / テキスト編集 / 戻る）と同じ構成。
 * チップ編集では、タグ入力欄に打つと Danbooru のタグ補完（投稿数つき）が出て、選ぶとチップになる。
 * チップは、タップで詳細ウィンドウ（名前・重み・コピー・Danbooru Wiki・削除）、掴んで動かすと並べ替え（SSXL と同じく FLIP で滑らかに入れ替わる）。
 */
@Composable
fun PromptEditor(vm: MainViewModel) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            ModeTab("チップ編集", vm.promptMode == "chip") { vm.changePromptMode("chip") }
            Spacer(Modifier.width(16.dp))
            ModeTab("テキスト編集", vm.promptMode == "text") { vm.changePromptMode("text") }
            Spacer(Modifier.weight(1f))
            TextButton(shape = AppShape, onClick = { vm.undoPrompt() }, enabled = vm.canUndoPrompt) { Text("戻る") }
        }
        if (vm.promptMode == "chip") ChipMode(vm) else TextMode(vm)
    }
}

@Composable
private fun ModeTab(label: String, selected: Boolean, onClick: () -> Unit) {
    val color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
    Column(Modifier.clickable(enabled = !selected, onClick = onClick), horizontalAlignment = Alignment.CenterHorizontally) {
        Text(label, style = MaterialTheme.typography.labelLarge, color = color, modifier = Modifier.padding(vertical = 6.dp))
        Box(Modifier.width(72.dp).height(2.dp).background(if (selected) color else Color.Transparent))
    }
}

@Composable
private fun TextMode(vm: MainViewModel) {
    OutlinedTextField(
        value = vm.prompt,
        onValueChange = { vm.prompt = it },
        placeholder = { Text("Positive Prompt") },
        minLines = 3,
        maxLines = 8,
        modifier = Modifier.fillMaxWidth(),
    )
}

/** ドラッグ中の状態。draft は掴んでいる間だけの仮の並び（離したときに確定する）。 */
private class DragState {
    var dragId by mutableStateOf<String?>(null)
    var offset by mutableStateOf(Offset.Zero)     // 掴んだチップの、レイアウト上の位置からのずれ
    var draft by mutableStateOf<List<PromptChip>?>(null)
    /** 並べ替えた直後は、レイアウトが更新されるまで次の判定をしない（古い位置で判定して元に戻す「行ったり来たり」を防ぐ）。 */
    var awaitLayout = false
    var awaitSince = 0L
}

@OptIn(ExperimentalLayoutApi::class, ExperimentalFoundationApi::class)
@Composable
private fun ChipMode(vm: MainViewModel) {
    var openId by remember { mutableStateOf<String?>(null) }
    val drag = remember { DragState() }
    val rects = remember { HashMap<String, Rect>() }        // 各チップの直近のレイアウト位置（FlowRow 内）
    Column(
        Modifier
            .fillMaxWidth()
            .clip(AppShape)
            .border(1.dp, MaterialTheme.colorScheme.outline, AppShape)
            .padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        val chips = drag.draft ?: vm.positiveChips
        if (chips.isNotEmpty()) {
            ChipFlow(order = chips.map { it.id }, gap = 6.dp) {
                chips.forEach { chip ->
                    key(chip.id) {
                        DraggableChip(
                            chip = chip,
                            drag = drag,
                            rects = rects,
                            enabled = true,
                            order = { drag.draft ?: vm.positiveChips },
                            onTap = { openId = chip.id },
                            onCommit = { ids -> vm.reorderChips(ids) },
                        )
                    }
                }
            }
        }

        // タグ入力欄と、その下の補完の候補は 1 つのまとまりにして、候補が出る（変わる）たびに、まとまり全体がキーボードの上に見えるところまで
        // 画面をスクロールする（入力欄のカーソルだけでは、その下の候補がキーボードの裏に隠れる）。まとまりの下端より 52dp 下まで見える位置に動かして、
        // 一番下の「このまま追加する」がキーボードにくっつかず、完全に見えるようにする（見た目の余白は足さない）
        val bringIntoView = remember { BringIntoViewRequester() }
        val showSuggestions = vm.tagSuggestions.isNotEmpty() || vm.tagNoHit
        var groupSize by remember { mutableStateOf(IntSize.Zero) }
        // まとまりの下端より下まで見せる量。まとまりの下には、枠の下の余白（8dp）と行間（8dp）の次に LoRA 欄が続くので、それが見え始める手前の 16dp まで
        // （これ以上見せると LoRA 欄が見えてしまう）。レイアウトに余白は足さない（枠の中に空白が出るため）
        val extraBelowPx = with(LocalDensity.current) { 16.dp.toPx() }
        LaunchedEffect(vm.tagSuggestions.size, vm.tagNoHit) {
            if (showSuggestions) {
                // 候補のレイアウトと、キーボードで縮んだ表示領域が落ち着いてから。遅れて落ち着く場合もあるので、もう一度（2 回目）やる
                repeat(2) { n ->
                    delay(if (n == 0) 120L else 330L)
                    bringIntoView.bringIntoView(Rect(0f, 0f, groupSize.width.toFloat(), groupSize.height + extraBelowPx))
                }
            }
        }
        Column(Modifier.fillMaxWidth().onSizeChanged { groupSize = it }.bringIntoViewRequester(bringIntoView), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            CompactTextField(
                value = vm.positiveChipInput,
                onValueChange = { vm.onChipInputChange(it) },
                placeholder = "Positiveタグを入力",
                enabled = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { vm.commitTyped(vm.positiveChipInput) }),
                modifier = Modifier.fillMaxWidth(),
            )
            if (showSuggestions) Suggestions(vm)
        }
        vm.chipToast?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary) }
    }
    openId?.let { id -> ChipDialog(vm, id, onDismiss = { openId = null }) }
}

/**
 * チップを折り返して並べる自前のレイアウト。並び順は、子のノード順ではなく order（チップの id の並び）で直接指定する。
 * （標準の FlowRow では、並べ替えても子が並び直されなかったため）
 */
@Composable
private fun ChipFlow(order: List<String>, gap: androidx.compose.ui.unit.Dp, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Layout(content = content, modifier = modifier) { measurables, constraints ->
        val gapPx = gap.roundToPx()
        val byId = measurables.associateBy { it.layoutId as? String }
        val ordered = order.mapNotNull { byId[it] }
        val maxW = constraints.maxWidth
        val placeables = ordered.map { it.measure(Constraints(maxWidth = maxW)) }
        val positions = ArrayList<IntOffset>(placeables.size)
        var x = 0
        var y = 0
        var lineH = 0
        for (p in placeables) {
            if (x > 0 && x + p.width > maxW) { x = 0; y += lineH + gapPx; lineH = 0 }
            positions.add(IntOffset(x, y))
            x += p.width + gapPx
            lineH = maxOf(lineH, p.height)
        }
        layout(maxW, y + lineH) {
            placeables.forEachIndexed { i, p -> p.place(positions[i]) }
        }
    }
}

/**
 * 1 つのチップ。タップ → onTap。8px 以上動かすとドラッグ開始（SSXL と同じ）。
 * ドラッグ中は指に追従し、別のチップの上に来ると、その前後に差し込まれる。ほかのチップは FLIP（元の位置から新しい位置へ滑らかに移動）で入れ替わる。
 */
@Composable
private fun DraggableChip(
    chip: PromptChip,
    drag: DragState,
    rects: HashMap<String, Rect>,
    enabled: Boolean,
    order: () -> List<PromptChip>,
    onTap: () -> Unit,
    onCommit: (List<String>) -> Unit,
) {
    val scope = rememberCoroutineScope()
    val flip = remember(chip.id) { Animatable(Offset.Zero, Offset.VectorConverter) }
    val dragging = drag.dragId == chip.id

    // 掴んだチップが、いまどの位置の前後に入るかを決めて、仮の並びを入れ替える
    fun reorderIfNeeded() {
        val id = chip.id
        if (drag.awaitLayout && SystemClock.uptimeMillis() - drag.awaitSince < 200) return
        val me = rects[id] ?: return
        val center = Offset(me.left + drag.offset.x + me.width / 2, me.top + drag.offset.y + me.height / 2)
        val list = order()
        val over = list.firstOrNull { it.id != id && rects[it.id]?.contains(center) == true } ?: return
        val r = rects[over.id] ?: return
        val before = center.x < r.left + r.width / 2
        val from = list.indexOfFirst { it.id == id }
        var to = list.indexOfFirst { it.id == over.id } + if (before) 0 else 1
        if (from < to) to -= 1
        if (to == from || from < 0) return
        val next = list.toMutableList()
        next.add(to, next.removeAt(from))
        drag.awaitLayout = true
        drag.awaitSince = SystemClock.uptimeMillis()
        drag.draft = next
    }

    fun finishDrag() {
        val ids = drag.draft?.map { it.id }
        val released = drag.offset
        drag.dragId = null
        drag.offset = Offset.Zero
        drag.draft = null
        // 離した位置から、確定した位置へ滑らかに収まる
        scope.launch { flip.snapTo(released); flip.animateTo(Offset.Zero, spring(stiffness = Spring.StiffnessMediumLow)) }
        if (ids != null) onCommit(ids)
    }

    val container = MaterialTheme.colorScheme.secondaryContainer
    val content = MaterialTheme.colorScheme.onSecondaryContainer
    val weighted = !chip.raw && !PromptChips.weightIsOne(chip.weight)
    Row(
        Modifier
            .layoutId(chip.id)
            .zIndex(if (dragging) 1f else 0f)
            .onGloballyPositioned { c ->
                // レイアウト位置（graphicsLayer の移動は含まない）を記録。位置が変わったら FLIP で元の位置から滑らかに動かす
                val pos = c.positionInParent()
                val prev = rects[chip.id]?.topLeft
                rects[chip.id] = Rect(pos, Size(c.size.width.toFloat(), c.size.height.toFloat()))
                if (prev != null && prev != pos) {
                    val delta = prev - pos
                    if (drag.dragId == chip.id) { drag.offset += delta; drag.awaitLayout = false }   // 掴んでいるものは、指の下に留める。レイアウトが更新されたので次の判定を許可
                    else scope.launch { flip.snapTo(flip.value + delta); flip.animateTo(Offset.Zero, tween(200)) }
                }
            }
            .graphicsLayer {
                val o = if (dragging) drag.offset else flip.value
                translationX = o.x
                translationY = o.y
                val s = if (dragging) 1.06f else 1f
                scaleX = s
                scaleY = s
                shadowElevation = if (dragging) 12f else 0f
            }
            .clip(AppShape)
            .background(container)
            .let { if (chip.raw) it.border(1.dp, MaterialTheme.colorScheme.error, AppShape) else it }
            .pointerInput(chip.id, enabled) {
                if (!enabled) return@pointerInput
                detectDragGestures(
                    onDragStart = { drag.dragId = chip.id; drag.offset = Offset.Zero; drag.draft = order() },
                    onDrag = { change, amount -> change.consume(); drag.offset += amount; reorderIfNeeded() },
                    onDragEnd = { finishDrag() },
                    onDragCancel = { finishDrag() },
                )
            }
            .clickable(enabled = enabled, onClick = onTap)
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text(if (chip.raw) "raw: ${chip.label} ⚠" else chip.label, style = MaterialTheme.typography.bodyMedium, color = content)
        if (weighted) {
            Text(PromptChips.formatWeight(chip.weight), style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.tertiary)
        }
    }
}

/**
 * チップをタップしたときのウィンドウ。名前の編集・重み（− / 数値 / ＋）・コピー・Danbooru Wiki（📖）・削除。
 * 重みはその場で反映する。名前は「完了」か閉じたときに反映する。
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ChipDialog(vm: MainViewModel, id: String, onDismiss: () -> Unit) {
    val chip = vm.positiveChips.firstOrNull { it.id == id }
    if (chip == null) { onDismiss(); return }
    val clipboard = LocalClipboardManager.current
    val uri = LocalUriHandler.current
    var name by remember(id) { mutableStateOf(chip.label) }
    var weightText by remember(id) { mutableStateOf(PromptChips.formatWeight(chip.weight)) }
    var copied by remember(id) { mutableStateOf(false) }
    fun close() { if (name.trim().isNotEmpty() && name != chip.label) vm.renameChip(id, name); onDismiss() }
    fun setWeight(w: Double) {
        weightText = PromptChips.formatWeight(w)
        vm.applyChipWeight(id, w)
    }
    AlertDialog(
        onDismissRequest = { close() },
        title = { Text(if (chip.raw) "チップ（raw）" else "チップ") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(14.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("名前") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                if (!chip.raw) {
                    // weight: 丸い − / 数値 / ＋ を、同じ高さ・同じ中心線で中央に並べる
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("weight", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally)) {
                            RoundButton("−") { setWeight(((PromptChips.normalizeWeight(weightText) ?: chip.weight) - 0.05).coerceAtLeast(-2.0)) }
                            OutlinedTextField(
                                value = weightText,
                                onValueChange = { t ->
                                    weightText = t.filter { it.isDigit() || it == '.' || it == '-' }.take(6)
                                    PromptChips.normalizeWeight(weightText)?.let { vm.applyChipWeight(id, it) }
                                },
                                singleLine = true,
                                textStyle = MaterialTheme.typography.titleMedium.copy(textAlign = TextAlign.Center),
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                                modifier = Modifier.width(104.dp),
                            )
                            RoundButton("＋") { setWeight(((PromptChips.normalizeWeight(weightText) ?: chip.weight) + 0.05).coerceAtMost(3.0)) }
                        }
                    }
                }
                // コピー / Wiki / 削除: 同じ幅で、中央に均等に並べる
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally), verticalAlignment = Alignment.CenterVertically) {
                    ActionButton(if (copied) "コピー済み" else "コピー", Modifier.weight(1f)) { clipboard.setText(AnnotatedString(PromptChips.serializeChip(chip))); copied = true }
                    if (!chip.raw && chip.booruKey.isNotEmpty()) ActionButton("📖 Wiki", Modifier.weight(1f)) { uri.openUri(DanbooruClient.wikiUrl(chip.booruKey)) }
                    ActionButton("削除", Modifier.weight(1f), danger = true) { vm.deleteChip(id); onDismiss() }
                }
            }
        },
        confirmButton = { TextButton(shape = AppShape, onClick = { close() }) { Text("完了") } },
    )
}

/** 丸い − / ＋ ボタン（数値欄と同じ高さ 56dp）。 */
@Composable
private fun RoundButton(symbol: String, onClick: () -> Unit) {
    Box(
        Modifier
            .size(56.dp)
            .clip(CircleShape)
            .background(MaterialTheme.colorScheme.secondaryContainer)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Text(symbol, style = MaterialTheme.typography.headlineSmall, color = MaterialTheme.colorScheme.onSecondaryContainer)
    }
}

/** 同じ高さ・文字が中央のボタン。danger=true は削除用の赤。 */
@Composable
private fun ActionButton(label: String, modifier: Modifier = Modifier, danger: Boolean = false, onClick: () -> Unit) {
    OutlinedButton(shape = AppShape, 
        onClick = onClick,
        colors = if (danger) ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error) else ButtonDefaults.outlinedButtonColors(),
        contentPadding = PaddingValues(horizontal = 8.dp),
        modifier = modifier.height(44.dp),
    ) { Text(label, maxLines = 1, textAlign = TextAlign.Center) }
}

/**
 * Danbooru のタグ補完の候補（SSXL の tag-suggestions）。「旧名 → タグ名」と投稿数を出す。タップでチップになる。
 * 候補は 4 行ぶんの高さで、それ以上（最大 20 件）は中でスクロールする。
 * 一番下には、候補の数にかかわらず「このまま追加する」を固定で出す（入力した文字をそのままチップにする）。
 */
@Composable
private fun Suggestions(vm: MainViewModel) {
    val rowHeight = 44.dp
    Column(
        Modifier
            .fillMaxWidth()
            .clip(AppShape)
            .background(MaterialTheme.colorScheme.surfaceVariant),
    ) {
        if (vm.tagSuggestions.isNotEmpty()) {
            Column(Modifier.fillMaxWidth().heightIn(max = rowHeight * 4).verticalScroll(rememberScrollState())) {
                vm.tagSuggestions.take(20).forEach { tag ->
                    Row(
                        Modifier.fillMaxWidth().height(rowHeight).clickable { vm.addDanbooruTag(tag.name) }.padding(horizontal = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        val label = if (tag.antecedent.isNotEmpty()) "${tag.antecedent} → ${tag.name}" else tag.label
                        Text(label, style = MaterialTheme.typography.bodyMedium, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                        Text(String.format(Locale.US, "%,d", tag.postCount), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
        }
        Row(
            Modifier.fillMaxWidth().height(rowHeight).clickable { vm.commitTyped(vm.positiveChipInput) }.padding(horizontal = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            val typed = vm.positiveChipInput.trim()
            val prefix = if (vm.tagSuggestions.isEmpty()) "ヒットなし：" else ""
            Text(
                "$prefix「$typed」をこのまま追加する",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.primary,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}
