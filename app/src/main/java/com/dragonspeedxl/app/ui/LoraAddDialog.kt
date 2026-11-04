package com.dragonspeedxl.app.ui

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.dragonspeedxl.app.MainActivity
import com.dragonspeedxl.app.MainViewModel
import com.dragonspeedxl.app.lora.CivitaiClient
import com.dragonspeedxl.app.lora.LoraConverter
import com.dragonspeedxl.app.qr.QrDecoder

/** ダイアログの Context は Activity のラップなので、Activity まで辿る。 */
internal tailrec fun Context.activity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.activity()
    else -> null
}

/**
 * LoRA を追加する。Civitai の URL から取るか、端末の safetensors を直接選んで、端末の中で adapter に変換する。
 * 変換できたら一覧に出る。activation text は Civitai なら自動、ファイルなら入力欄の値。
 */
@Composable
fun LoraAddDialog(vm: MainViewModel, onDismiss: () -> Unit) {
    var fromFile by remember { mutableStateOf(false) }
    var url by remember { mutableStateOf("") }
    var pickedUri by remember { mutableStateOf<Uri?>(null) }
    var pickedName by remember { mutableStateOf("") }
    var activation by remember { mutableStateOf("") }
    val context = LocalContext.current
    val activity = context.activity() as? MainActivity
    var qrError by remember { mutableStateOf<String?>(null) }
    var qrBusy by remember { mutableStateOf(false) }   // 撮った写真から QR を読んでいる最中
    val scope = rememberCoroutineScope()
    // QR を読み取る。端末のカメラアプリで撮ってもらい（アプリはカメラの権限を持たない）、その写真から読む。
    // 読めた値が Civitai のモデルの URL なら URL 欄に入れ、それ以外は案内を出す
    fun scanQr() {
        qrError = null
        val a = activity
        if (a == null) {
            qrError = "画面を取得できませんでした"
            return
        }
        val photo = File(File(context.cacheDir, "qr").apply { mkdirs() }, "qr.jpg")
        photo.delete()
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", photo)
        val launched = a.takePhoto(uri) { ok ->
            if (!ok) {
                photo.delete()
                return@takePhoto
            }
            qrBusy = true
            scope.launch {
                val value = withContext(Dispatchers.Default) {
                    try { QrDecoder.decode(photo) } finally { photo.delete() }
                }?.trim()
                qrBusy = false
                when {
                    value == null -> qrError = "QR コードが見つかりませんでした（QR を大きく、ピントを合わせて撮ってください）"
                    CivitaiClient.isCivitaiHost(value) && runCatching { CivitaiClient.parseUrl(value) }.isSuccess -> url = value
                    else -> qrError = "Civitai のモデルの URL ではありません（読み取った内容: ${value.take(80)}）"
                }
            }
        }
        if (!launched) qrError = "カメラアプリが見つかりません"
    }
    val onPicked: (Uri?) -> Unit = { uri ->
        if (uri != null) {
            pickedUri = uri
            pickedName = vm.loraDisplayName(uri)
        }
    }
    val missing = vm.missingLoraTemplate()
    // 端末のファイルの名前（追加後の LoRA の名前）が、既に一覧にあるか。Civitai は取ってみないと名前が分からないので、取り込み側で止める
    val pickedSafe = if (pickedUri == null) "" else LoraConverter.safeName(pickedName.substringBeforeLast('.'))
    val duplicate = fromFile && pickedSafe.isNotEmpty() && pickedSafe in vm.loraNames
    // 追加が成功したら、入力欄を空に戻す。追加ボタンは入力が無いあいだ無効になり、同じ入力で二度押して弾かれない
    LaunchedEffect(vm.loraImportedCount) {
        if (vm.loraImportedCount > 0) {
            url = ""
            pickedUri = null
            pickedName = ""
            activation = ""
        }
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("LoRA を追加") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(selected = !fromFile, onClick = { fromFile = false }, enabled = !vm.loraImporting, label = { Text("Civitai の URL") })
                    FilterChip(selected = fromFile, onClick = { fromFile = true }, enabled = !vm.loraImporting, label = { Text("ファイル") })
                }
                if (!fromFile) {
                    Text(
                        "Civitai のモデルのページの URL を貼ります（modelVersionId 付きならその版、無ければ最新版）。UNet の重みと、あればテキストエンコーダ（CLIP）の重みを使います。",
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    OutlinedTextField(
                        value = url, onValueChange = { url = it }, label = { Text("URL") }, singleLine = true, shape = AppShape,
                        enabled = !vm.loraImporting, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri), modifier = Modifier.fillMaxWidth(),
                    )
                    OutlinedButton(onClick = { scanQr() }, enabled = !vm.loraImporting && !qrBusy) {
                        Text(if (qrBusy) "QR コードを読み取り中…" else "QR コードを撮影して読み取る")
                    }
                    qrError?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error) }
                    Text(
                        if (vm.civitaiToken.isEmpty()) "API トークン: 未登録（設定で登録します。ログインが要るモデルには必要です）" else "API トークン: 登録済み（設定で変えられます）",
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    Text(
                        "端末の safetensors を選びます（名前は、ファイル名から作ります。英数字・_・.・- 以外は _ に直します）。activation text は空でもよいです。UNet の重みと、あればテキストエンコーダ（CLIP）の重みを使います。",
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    OutlinedButton(onClick = { activity?.pickFile(onPicked) }, enabled = !vm.loraImporting && activity != null) {
                        Text(if (pickedUri == null) "safetensors を選ぶ" else "別のファイルを選ぶ")
                    }
                    if (pickedName.isNotEmpty()) {
                        Text("選んだファイル: $pickedName", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    if (duplicate) {
                        Text("「$pickedSafe」は既に追加されています（同じ名前の LoRA があります）。", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                    }
                    OutlinedTextField(
                        value = activation, onValueChange = { activation = it }, label = { Text("activation text（トリガーの文）") }, minLines = 2, shape = AppShape,
                        enabled = !vm.loraImporting, modifier = Modifier.fillMaxWidth(),
                    )
                }
                if (missing != null) {
                    Text("ひな型 $missing が models/ にありません。", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                }
                if (vm.loraImporting) LinearProgressIndicator(progress = { vm.loraImportProgress }, modifier = Modifier.fillMaxWidth())
                if (vm.loraImportMessage.isNotEmpty()) {
                    Text(
                        vm.loraImportMessage, style = MaterialTheme.typography.bodySmall,
                        color = if (vm.loraImportFailed) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface,
                    )
                }
            }
        },
        confirmButton = {
            val ready = if (fromFile) pickedUri != null else url.isNotBlank()
            TextButton(
                // 名前はファイル名（拡張子を除く）から作る。英数字以外は safeName が直す
                onClick = { if (fromFile) pickedUri?.let { vm.importLoraFile(it, LoraConverter.safeName(pickedName.substringBeforeLast('.')), activation) } else vm.importLora(url) },
                enabled = !vm.loraImporting && !vm.busy && ready && missing == null && !duplicate,
            ) { Text("追加") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("閉じる") } },
    )
}
