package com.dragonspeedxl.app.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.dragonspeedxl.app.MainViewModel
import com.dragonspeedxl.app.R
import com.dragonspeedxl.app.download.ModelDownloader
import com.dragonspeedxl.app.download.ModelManifest

/**
 * 初回の案内（1 画面）。生成に使うモデルが端末に揃うまで、本体の画面は出さない。
 * 上に アプリ名（中央）、その下に説明・進み具合、下にボタンとライセンス。取り終えると、本体の画面に切り替わる。
 */
@Composable
fun ModelSetupScreen(vm: MainViewModel) {
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(
            Modifier
                .fillMaxSize()
                .safeDrawingPadding()
                .padding(horizontal = 24.dp, vertical = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(Modifier.weight(1f))

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Image(
                    painter = painterResource(R.drawable.ic_logo),
                    contentDescription = null,
                    modifier = Modifier.size(96.dp),
                )
                Text(
                    stringResource(R.string.app_name),
                    style = MaterialTheme.typography.headlineMedium,
                    textAlign = TextAlign.Center,
                )
            }

            Spacer(Modifier.height(40.dp))

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    "生成に必要なモデルをダウンロードします\n中断しても続きから再開されます\n※初回一度のみ。生成は完全オフラインでされます。",
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                vm.downloadProgress?.let { p ->
                    Spacer(Modifier.height(8.dp))
                    LinearProgressIndicator(
                        progress = { if (p.totalBytes > 0) p.doneBytes.toFloat() / p.totalBytes else 0f },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Text(
                        "${ModelDownloader.gb(p.doneBytes)} / ${ModelDownloader.gb(p.totalBytes)} GB  ·  ${p.fileIndex + 1} / ${p.fileCount}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                vm.downloadMessage?.let {
                    Text(
                        it,
                        style = MaterialTheme.typography.bodySmall,
                        textAlign = TextAlign.Center,
                        color = if (vm.downloadFailed) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            Spacer(Modifier.weight(1f))

            Button(
                shape = AppShape,
                enabled = !vm.downloading && vm.modelIdle,
                onClick = { vm.startModelDownload() },
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) {
                Text(
                    when {
                        vm.downloading -> "取得中…"
                        vm.downloadFailed -> "もう一度取る"
                        else -> "ダウンロード開始（約7.6GB）"
                    },
                )
            }

            Spacer(Modifier.height(16.dp))
            Text(
                "モデルは ${ModelManifest.REPO}（CC BY-NC-SA 4.0）",
                style = MaterialTheme.typography.labelSmall,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}
