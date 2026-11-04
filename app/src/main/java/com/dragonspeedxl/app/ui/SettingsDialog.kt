package com.dragonspeedxl.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.dragonspeedxl.app.HISTORY_GRID_MAX
import com.dragonspeedxl.app.HISTORY_GRID_MIN
import com.dragonspeedxl.app.MainActivity
import com.dragonspeedxl.app.MainViewModel
import com.dragonspeedxl.app.security.Biometrics

private enum class Step { NONE, SET_PASSWORD, CHANGE_PASSWORD, DISABLE_PASSWORD }

/** 設定。上から、セキュリティ、履歴（横×縦、除外）、起動時のプロンプト、タグ補完、Civitai、性能。 */
@Composable
fun SettingsDialog(vm: MainViewModel, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val activity = context.findFragmentActivity()
    val bioAvailable = remember { Biometrics.available(context) }
    var step by remember { mutableStateOf(Step.NONE) }
    var note by remember { mutableStateOf<String?>(null) }
    val taesdMissing = vm.missingTaesdFile()
    LaunchedEffect(Unit) { vm.refreshTagDictionary() }

    // ウィンドウの高さはディスプレイの 80%。中身がはみ出すときはスクロールする
    val screenHeight = LocalConfiguration.current.screenHeightDp.dp
    AlertDialog(
        onDismissRequest = onDismiss,
        modifier = Modifier.height(screenHeight * 0.8f),
        title = { Text("設定") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("セキュリティ", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)

                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("パスワードロック")
                    }
                    Switch(checked = vm.hasPassword, onCheckedChange = { step = if (it) Step.SET_PASSWORD else Step.DISABLE_PASSWORD })
                }
                if (vm.hasPassword) {
                    OutlinedButton(shape = AppShape, onClick = { step = Step.CHANGE_PASSWORD }, modifier = Modifier.fillMaxWidth()) { Text("パスワードを変更") }
                }

                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("指紋認証ロック")
                    }
                    Switch(
                        checked = vm.biometricOn,
                        enabled = bioAvailable || vm.biometricOn,
                        onCheckedChange = { on ->
                            if (!on) {
                                vm.setBiometric(false); note = null
                            } else {
                                // 有効にする前に、一度認証して本当に使えることを確かめる
                                val a = activity
                                if (a == null) note = "画面を取得できませんでした" else Biometrics.prompt(
                                    a, "指紋認証ロックを有効にする", allowDeviceCredential = false,
                                    onSuccess = { vm.setBiometric(true); note = null },
                                    onError = { note = it },
                                )
                            }
                        },
                    )
                }

                OutlinedButton(shape = AppShape, onClick = { onDismiss(); vm.lockNow() }, enabled = vm.hasPassword || vm.biometricOn, modifier = Modifier.fillMaxWidth()) { Text("今すぐロック") }
                note?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }

                Text("履歴", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                // SSXL の「除外」（blurBlacklist）。オンのとき、下の語句を含む履歴を隠す
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("除外")
                        Text("以下のプロンプトを隠す（カンマ区切り）", style =MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Switch(checked = vm.blurBlacklist, onCheckedChange = { vm.updateBlurBlacklist(it) })
                }
                OutlinedTextField(
                    value = vm.imageBlacklistTags,
                    onValueChange = { vm.updateImageBlacklistTags(it) },
                    label = { Text("除外ワード") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text("1ページの件数")
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("横")
                    GridCountField(vm.historyCols, vm::updateHistoryCols)
                    Text("×")
                    Text("縦")
                    GridCountField(vm.historyRows, vm::updateHistoryRows)
                }
                Text(
                    "表示枚数：${vm.historyPageSize}枚",
                    style = MaterialTheme.typography.bodyMedium,
                )
                // 履歴を zip（画像と index.json）に書き出す。保存先はユーザーが選ぶ。裏で進み、通知に出す
                val activity = LocalContext.current.activity() as? MainActivity
                OutlinedButton(
                    onClick = { activity?.createFile(vm.historyExportName(), "application/zip") { uri -> if (uri != null) vm.exportHistory(uri) } },
                    enabled = vm.history.isNotEmpty() && activity != null,
                ) { Text("履歴を書き出す（画像と一覧・zip）") }
                // 書き出した zip を読み込んで、足りない項目を履歴に足す（同じ id は飛ばす）
                OutlinedButton(
                    onClick = { activity?.pickFile { uri -> if (uri != null) vm.importHistory(uri) } },
                    enabled = activity != null,
                ) { Text("履歴を読み込む（zip）") }

                Text("起動時のプロンプト", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                OutlinedTextField(
                    value = vm.defaultPrompt,
                    onValueChange = { vm.updateDefaultPrompt(it) },
                    label = { Text("デフォルトプロンプト") },
                    modifier = Modifier.fillMaxWidth(),
                )

                Text("タグ補完", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                OptionSwitch(
                    title = "ローカル辞書モード",
                    detail = "dskjalのDanbooruタグ一覧で補完します。\nオフでも、Danbooruにつながらないときは自動的にONになります。",
                    checked = vm.localTagMode,
                    onCheckedChange = { vm.updateLocalTagMode(it) },
                    missingFile = null,
                )
                val dictError = vm.tagDictionaryError
                Text(
                    when {
                        vm.tagDictionaryLoading -> "辞書: 取得中…"
                        vm.tagDictionaryName != null -> "辞書: ${vm.tagDictionaryName}（取得済み）"
                        dictError != null -> "辞書: 取得できませんでした（$dictError）"
                        else -> "辞書: 未取得（初めて使うときに dskjal.com から約3MBを取得します）"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = if (dictError != null && !vm.tagDictionaryLoading) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (dictError != null && !vm.tagDictionaryLoading && vm.tagDictionaryName == null) {
                    OutlinedButton(shape = AppShape, onClick = { vm.fetchTagDictionary() }, modifier = Modifier.fillMaxWidth()) { Text("辞書を取得し直す") }
                }

                Text("Civitai", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                OutlinedTextField(
                    value = vm.civitaiToken,
                    onValueChange = { vm.updateCivitaiToken(it) },
                    label = { Text("API トークン") },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    "Civitaiログインが必要なモデルのダウンロードに必要。空欄でも可",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Text("性能", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                OptionSwitch(
                    title = "生成中のプレビュー",
                    detail = "TAESDでステップごとに途中経過を出します。空いているGPUを使用するため、生成時間はほとんど変わりません。",
                    checked = vm.previewTaesd,
                    onCheckedChange = { vm.updatePreviewTaesd(it) },
                    missingFile = taesdMissing,
                )
                OptionSwitch(
                    title = "メモリ節約モード",
                    detail = "UNetを指定された解像度の分だけ読み込み、メモリロードを約1.4GB削減します。",
                    checked = vm.memorySaver,
                    onCheckedChange = { vm.updateMemorySaver(it) },
                    missingFile = null,
                )
                OptionSwitch(
                    title = "高速デコード",
                    detail = "TAESDでデコードすることで数秒速くなります。細部が少し劣化します。",
                    checked = vm.draftMode,
                    onCheckedChange = { vm.updateDraftMode(it) },
                    missingFile = taesdMissing,
                )
                // GPU の上限は常に出す。高速デコードが有効でも、生成中のプレビュー（TAESD）には効くため
                Column {
                    Text("GPU使用率の上限  ${if (vm.gpuCap >= 100) "制限なし" else "${vm.gpuCap}%"}")
                    Slider(
                        value = vm.gpuCap.toFloat(),
                        onValueChange = { vm.updateGpuCap(it.toInt() / 5 * 5) },
                        valueRange = 10f..100f,
                        steps = 17,
                    )
                    Text(
                        "VAEデコードのNPU・GPU分業と生成中のプレビューにおけるGPU使用率の上限です。OSごとフレームドロップする場合に下げてください。",
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        },
        confirmButton = { TextButton(shape = AppShape, onClick = onDismiss) { Text("閉じる") } },
    )

    when (step) {
        Step.SET_PASSWORD -> PasswordDialog(
            title = "パスワードを設定", askCurrent = false,
            onDone = { _, new -> vm.setPassword(new); step = Step.NONE }, onCancel = { step = Step.NONE }, vm = vm,
        )
        Step.CHANGE_PASSWORD -> PasswordDialog(
            title = "パスワードを変更", askCurrent = true,
            onDone = { _, new -> vm.setPassword(new); step = Step.NONE }, onCancel = { step = Step.NONE }, vm = vm,
        )
        Step.DISABLE_PASSWORD -> VerifyDialog(
            title = "パスワードロックを解除", vm = vm,
            onOk = { vm.clearPassword(); step = Step.NONE }, onCancel = { step = Step.NONE },
        )
        Step.NONE -> {}
    }
}

/** 履歴の列数・行数。1〜8 の整数。空欄のあいだは、直前の値のままにする。 */
@Composable
private fun GridCountField(value: Int, onValue: (Int) -> Unit) {
    var text by remember(value) { mutableStateOf(value.toString()) }
    CompactTextField(
        value = text,
        onValueChange = { raw ->
            val digits = raw.filter { it.isDigit() }.take(1)
            text = digits
            val n = digits.toIntOrNull() ?: return@CompactTextField
            val c = n.coerceIn(HISTORY_GRID_MIN, HISTORY_GRID_MAX)
            if (c != n) text = c.toString()
            onValue(c)
        },
        placeholder = "4",
        enabled = true,
        modifier = Modifier.width(72.dp),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
    )
}

/**
 * 性能のスイッチ。missingFile があるときはオフにして触れなくし、
 * 「＜ファイル名＞が存在しないため選択できません」を出す。保存してあるオン／オフは変えない。
 */
@Composable
private fun OptionSwitch(
    title: String,
    detail: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    missingFile: String?,
) {
    val available = missingFile == null
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title)
            Text(detail, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (missingFile != null) {
                Text(
                    "${missingFile}が存在しないため選択できません",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
        Switch(checked = available && checked, enabled = available, onCheckedChange = onCheckedChange)
    }
}

/** 新しいパスワード（2 回入力）。askCurrent=true なら、現在のパスワードも確認する。 */
@Composable
private fun PasswordDialog(title: String, askCurrent: Boolean, onDone: (String, String) -> Unit, onCancel: () -> Unit, vm: MainViewModel) {
    var current by remember { mutableStateOf("") }
    var p1 by remember { mutableStateOf("") }
    var p2 by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }
    AlertDialog(
        onDismissRequest = onCancel,
        title = { Text(title) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (askCurrent) PasswordField("現在のパスワード", current) { current = it; error = null }
                PasswordField("新しいパスワード（4 文字以上）", p1) { p1 = it; error = null }
                PasswordField("もう一度入力", p2) { p2 = it; error = null }
                error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            }
        },
        confirmButton = {
            TextButton(shape = AppShape, onClick = {
                when {
                    askCurrent && !vm.lock.verify(current) -> error = "現在のパスワードが違います"
                    p1.length < 4 -> error = "4 文字以上にしてください"
                    p1 != p2 -> error = "2 回の入力が一致しません"
                    else -> onDone(current, p1)
                }
            }) { Text("保存") }
        },
        dismissButton = { TextButton(shape = AppShape, onClick = onCancel) { Text("キャンセル") } },
    )
}

/** 現在のパスワードを確認してから解除する。 */
@Composable
private fun VerifyDialog(title: String, vm: MainViewModel, onOk: () -> Unit, onCancel: () -> Unit) {
    var pw by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }
    AlertDialog(
        onDismissRequest = onCancel,
        title = { Text(title) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                PasswordField("現在のパスワード", pw) { pw = it; error = null }
                error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            }
        },
        confirmButton = {
            TextButton(shape = AppShape, onClick = {
                if (vm.lock.verify(pw)) onOk() else error = "パスワードが違います"
            }) { Text("解除する") }
        },
        dismissButton = { TextButton(shape = AppShape, onClick = onCancel) { Text("キャンセル") } },
    )
}

@Composable
private fun PasswordField(label: String, value: String, onChange: (String) -> Unit) {
    OutlinedTextField(
        value = value, onValueChange = onChange, label = { Text(label) }, singleLine = true,
        visualTransformation = PasswordVisualTransformation(),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
        modifier = Modifier.fillMaxWidth(),
    )
}
