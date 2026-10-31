package com.dragonspeedxl.app.ui

import android.content.Context
import android.content.ContextWrapper
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import com.dragonspeedxl.app.MainViewModel
import com.dragonspeedxl.app.R
import com.dragonspeedxl.app.security.Biometrics

internal fun Context.findFragmentActivity(): FragmentActivity? {
    var c: Context? = this
    while (c is ContextWrapper) {
        if (c is FragmentActivity) return c
        c = c.baseContext
    }
    return null
}

/**
 * ロック中の画面。ロゴと最低限の文だけ。
 * 指紋認証が有効なときは、画面が前面に戻るたびに自動で指紋認証を出す（ボタンを押させない）。
 * 失敗しても何回でもやり直せる（認証を閉じた場合は、画面のどこかをタップすればもう一度出る）。
 */
@Composable
fun LockScreen(vm: MainViewModel) {
    val activity = LocalContext.current.findFragmentActivity()
    var password by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }

    fun tryPassword() {
        if (vm.lock.verify(password)) vm.unlock() else { password = ""; error = "パスワードが違います" }
    }

    fun askBiometric() {
        val a = activity ?: return
        // 端末の PIN / パターンでも通せるようにして、行き止まりを作らない
        Biometrics.prompt(a, "dragonspeedXL", allowDeviceCredential = true, onSuccess = { vm.unlock() }, onError = { })
    }

    // ロック画面が前面に来るたび（起動時・別アプリから戻ったとき）に指紋認証を自動で出す
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        if (vm.locked && vm.lock.biometricEnabled) askBiometric()
    }

    Surface(
        color = MaterialTheme.colorScheme.background,
        contentColor = MaterialTheme.colorScheme.onBackground,
        modifier = Modifier.fillMaxSize().clickable(enabled = vm.lock.biometricEnabled) { askBiometric() },   // 認証を閉じたあと、タップで再表示
    ) {
        Column(
            Modifier.fillMaxSize().padding(32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Image(painterResource(R.drawable.ic_launcher), contentDescription = "dragonspeedXL", modifier = Modifier.size(96.dp))
            Spacer(Modifier.height(12.dp))
            Text("dragonspeedXL", style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.primary)
            if (vm.lock.biometricEnabled) {
                Spacer(Modifier.height(4.dp))
                Text("指紋で解除", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }

            if (vm.lock.hasPassword) {
                Spacer(Modifier.height(28.dp))
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it; error = null },
                    label = { Text("パスワード") },
                    singleLine = true,
                    isError = error != null,
                    supportingText = error?.let { { Text(it) } },
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(onDone = { if (password.isNotEmpty()) tryPassword() }),
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(8.dp))
                Button(shape = AppShape, onClick = { tryPassword() }, enabled = password.isNotEmpty(), modifier = Modifier.fillMaxWidth().height(48.dp)) { Text("解除") }
            }
        }
    }
}
