package com.dragonspeedxl.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.snapshotFlow
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import com.dragonspeedxl.app.ui.AppTheme
import com.dragonspeedxl.app.ui.MainScreen
import com.dragonspeedxl.app.ui.ModelSetupScreen

class MainActivity : FragmentActivity() {   // BiometricPrompt のため FragmentActivity（ComponentActivity を継承しているので Compose はそのまま使える）
    private val vm: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // 取得の進み具合を通知に出すため（Android 13 以降は許可が要る）。断られても取得は続く（通知が出ないだけ）
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFY)
        }
        // ロックが有効な間は、アプリスイッチャー（最近使ったアプリ）の縮小画像に中身を残さない
        lifecycleScope.launch { snapshotFlow { vm.hasPassword || vm.biometricOn }.collect { hideFromRecents(it) } }
        setContent {
            // 生成に使うモデルが足りない間（初回など）は、取り終えるまで本体の画面を出さない
            AppTheme { if (vm.modelsNeedDownload) ModelSetupScreen(vm) else MainScreen(vm) }
        }
    }

    /**
     * hide=true で、アプリスイッチャーの縮小画像を出さない。
     * Android 13 以降は専用の API（スクリーンショット自体は妨げない）、12 以前は画面保護（FLAG_SECURE。スクショも不可になる）で代用する。
     */
    private fun hideFromRecents(hide: Boolean) {
        if (Build.VERSION.SDK_INT >= 33) {
            setRecentsScreenshotEnabled(!hide)
        } else if (hide) {
            window.setFlags(WindowManager.LayoutParams.FLAG_SECURE, WindowManager.LayoutParams.FLAG_SECURE)
        } else {
            window.clearFlags(WindowManager.LayoutParams.FLAG_SECURE)
        }
    }

    /** 別のアプリに切り替えた・閉じたとき。次に開いたときにロック画面が出るようにする。 */
    override fun onStop() {
        super.onStop()
        if (!pickingFile) vm.lockNow()   // 端末のファイル選択へ出ているときは、自分で開いた画面なのでロックしない（戻ったとき指紋認証で選択が消えるのを防ぐ）
    }

    private var onFilePicked: ((Uri?) -> Unit)? = null
    private var pickingFile = false

    /**
     * 端末のファイルを 1 つ選ばせ、選んだ Uri（キャンセルなら null）を callback に渡す。
     * ActivityResultLauncher は使わない: FragmentActivity では、その要求コードが 16 bit を超えて落ちるため（Can only use lower 16 bits for requestCode）。
     */
    fun pickFile(callback: (Uri?) -> Unit) {
        onFilePicked = callback
        pickingFile = true
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
        startActivityForResult(intent, REQUEST_PICK_FILE)
    }

    /** 書き出し先（ファイル名 name、型 mime）を、ユーザーに選ばせる。キャンセルなら null。pickFile と同じ理由で ActivityResultLauncher は使わない。 */
    fun createFile(name: String, mime: String, callback: (Uri?) -> Unit) {
        onFilePicked = callback
        pickingFile = true
        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType(mime).putExtra(Intent.EXTRA_TITLE, name)
        startActivityForResult(intent, REQUEST_CREATE_FILE)
    }

    @Suppress("DEPRECATION")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_PICK_FILE || requestCode == REQUEST_CREATE_FILE) {
            pickingFile = false
            val cb = onFilePicked
            onFilePicked = null
            cb?.invoke(if (resultCode == RESULT_OK) data?.data else null)
        }
    }

    companion object {
        private const val REQUEST_PICK_FILE = 0x4c52   // 16 bit に収まる小さい値
        private const val REQUEST_CREATE_FILE = 0x4c53
        private const val REQUEST_NOTIFY = 0x4e54
    }
}
