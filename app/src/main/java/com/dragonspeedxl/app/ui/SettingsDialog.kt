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
import com.dragonspeedxl.app.MainViewModel
import com.dragonspeedxl.app.security.Biometrics

private enum class Step { NONE, SET_PASSWORD, CHANGE_PASSWORD, DISABLE_PASSWORD }

/** 設定。上から、セキュリティ、履歴（横×縦、除外）、起動時のプロンプト、Civitai、性能。 */
@Composable
fun SettingsDialog(vm: MainViewModel, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val activity = context.findFragmentActivity()
    val bioAvailable = remember { Biometrics.available(context) }
    var step by remember { mutableStateOf(Step.NONE) }
    var note by remember { mutableStateOf<String?>(null) }
    val taesdMissing = vm.missingTaesdFile()

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
                        Text("アプリを開くときにパスワードを求める", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Switch(checked = vm.hasPassword, onCheckedChange = { step = if (it) Step.SET_PASSWORD else Step.DISABLE_PASSWORD })
                }
                if (vm.hasPassword) {
                    OutlinedButton(shape = AppShape, onClick = { step = Step.CHANGE_PASSWORD }, modifier = Modifier.fillMaxWidth()) { Text("パスワードを変更") }
                }

                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("指紋認証ロック")
                        Text(
                            if (bioAvailable) "指紋で解除できるようにする" else "指紋が登録されていないため使えません（端末の設定で登録）",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
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
                Text(
                    "ロックは、アプリを閉じる／別のアプリに切り替えたあと、開き直したときにかかります。",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                note?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }

                Text("履歴", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
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
                // SSXL の「除外」（blurBlacklist）。オンのとき、下の語句を含む履歴を隠す
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("除外")
                        Text("以下のプロンプトが含まれている履歴を隠す", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Switch(checked = vm.blurBlacklist, onCheckedChange = { vm.updateBlurBlacklist(it) })
                }
                OutlinedTextField(
                    value = vm.imageBlacklistTags,
                    onValueChange = { vm.updateImageBlacklistTags(it) },
                    label = { Text("除外ワード") },
                    modifier = Modifier.fillMaxWidth(),
                )

                Text("起動時のプロンプト", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                OutlinedTextField(
                    value = vm.defaultPrompt,
                    onValueChange = { vm.updateDefaultPrompt(it) },
                    label = { Text("デフォルトプロンプト") },
                    modifier = Modifier.fillMaxWidth(),
                )

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
                    title = "高速デコード",
                    detail = "TAESDでデコードすることで数秒速くなります。細部が少し劣化します。",
                    checked = vm.draftMode,
                    onCheckedChange = { vm.updateDraftMode(it) },
                    missingFile = taesdMissing,
                )
                OptionSwitch(
                    title = "メモリ節約モード",
                    detail = "UNetを今の向きの分だけ読み込み、メモリ(DMA-BUF)を約1.4GB減らします。縦横を切り替えるときに約2秒の読み直しが入ります。次の生成から反映されます。",
                    checked = vm.memorySaver,
                    onCheckedChange = { vm.updateMemorySaver(it) },
                    missingFile = null,
                )
                // GPU の上限は、高速デコードが実際に効いていないとき（VAE でデコードするとき）だけ出す。TAESD が無いときは保存がオンでも VAE になるので出す
                if (!vm.draftActive) {
                    Column {
                        Text("GPU使用率の上限  ${if (vm.gpuCap >= 100) "制限なし" else "${vm.gpuCap}%"}")
                        Slider(
                            value = vm.gpuCap.toFloat(),
                            onValueChange = { vm.updateGpuCap(it.toInt() / 5 * 5) },
                            valueRange = 10f..100f,
                            steps = 17,
                        )
                        Text(
                            "VAEデコードをNPUとGPUで分担するときのGPUの上限です。描画が重くなる場合に下げてください。",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
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
