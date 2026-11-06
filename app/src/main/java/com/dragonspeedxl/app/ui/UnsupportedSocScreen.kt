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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.dragonspeedxl.app.R
import com.dragonspeedxl.app.engine.SocSupport

/**
 * 起動時の案内: この端末の SoC は対応していない。モデル（約 7.6GB）を取る前に出す。
 * SoC の判定を誤っても閉め出さないよう、「それでも続ける」を置く（その起動の間だけ有効。起動し直すとまた出る）。
 */
@Composable
fun UnsupportedSocScreen(soc: SocSupport.Result, onContinue: () -> Unit) {
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
                Image(painterResource(R.drawable.ic_logo), contentDescription = null, modifier = Modifier.size(96.dp))
                Text(stringResource(R.string.app_name), style = MaterialTheme.typography.headlineMedium, textAlign = TextAlign.Center)
            }
            Spacer(Modifier.height(40.dp))
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("この端末には対応していません", style = MaterialTheme.typography.titleMedium, textAlign = TextAlign.Center)
                Text(
                    "対応しているのは Snapdragon 8 Elite Gen 5（SM8850）だけです。",
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    buildString {
                        append("この端末の SoC: ").append(soc.socModel)
                        soc.product?.let { append("（").append(it).append("）") }
                    },
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Spacer(Modifier.weight(1f))
            OutlinedButton(shape = AppShape, onClick = onContinue, modifier = Modifier.fillMaxWidth().height(52.dp)) {
                Text("それでも続ける（動作は保証しません）")
            }
        }
    }
}
