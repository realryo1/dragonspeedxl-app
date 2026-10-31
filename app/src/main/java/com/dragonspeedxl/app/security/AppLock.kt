package com.dragonspeedxl.app.security

import android.content.Context
import android.util.Base64
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricManager.Authenticators.BIOMETRIC_STRONG
import androidx.biometric.BiometricManager.Authenticators.DEVICE_CREDENTIAL
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec

/**
 * アプリのロック設定（パスワード / 指紋認証）。パスワードは PBKDF2-HMAC-SHA256（ソルト付き）のハッシュだけを保存し、平文は残さない。
 * 目的は「誤って開くのを防ぐ」ことなので、間違えても何回でもやり直せる（回数制限・待ち時間はない）。
 */
class AppLock(context: Context) {
    private val prefs = context.getSharedPreferences("app_lock", Context.MODE_PRIVATE)

    val hasPassword: Boolean get() = prefs.contains(KEY_HASH)
    var biometricEnabled: Boolean
        get() = prefs.getBoolean(KEY_BIO, false)
        set(v) = prefs.edit().putBoolean(KEY_BIO, v).apply()

    /** どちらかのロックが有効か */
    val enabled: Boolean get() = hasPassword || biometricEnabled

    fun setPassword(password: String) {
        val salt = ByteArray(16).also { SecureRandom().nextBytes(it) }
        prefs.edit()
            .putString(KEY_SALT, Base64.encodeToString(salt, Base64.NO_WRAP))
            .putString(KEY_HASH, Base64.encodeToString(hash(password, salt), Base64.NO_WRAP))
            .apply()
    }

    fun clearPassword() {
        prefs.edit().remove(KEY_SALT).remove(KEY_HASH).apply()
    }

    /** 入力が正しいか（定数時間比較）。 */
    fun verify(password: String): Boolean {
        val salt = prefs.getString(KEY_SALT, null)?.let { Base64.decode(it, Base64.NO_WRAP) } ?: return false
        val expected = prefs.getString(KEY_HASH, null)?.let { Base64.decode(it, Base64.NO_WRAP) } ?: return false
        return MessageDigest.isEqual(hash(password, salt), expected)
    }

    private fun hash(password: String, salt: ByteArray): ByteArray =
        SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(PBEKeySpec(password.toCharArray(), salt, ITERATIONS, 256)).encoded

    private companion object {
        const val KEY_SALT = "salt"
        const val KEY_HASH = "hash"
        const val KEY_BIO = "bio"
        const val ITERATIONS = 120_000
    }
}

/** 指紋（生体）認証。BiometricPrompt をラップする。 */
object Biometrics {
    /** この端末で指紋などの強い生体認証が使える（登録済み）か */
    fun available(context: Context): Boolean =
        BiometricManager.from(context).canAuthenticate(BIOMETRIC_STRONG) == BiometricManager.BIOMETRIC_SUCCESS

    /**
     * 認証ダイアログを出す。allowDeviceCredential=true なら、生体が失敗し続けたとき・使えないときに端末の PIN / パターンでも通せる
     * （行き止まりを作らない。ロックは誤って開くのを防ぐのが目的なので、端末の持ち主が通れて困ることはない）。
     */
    fun prompt(activity: FragmentActivity, title: String, allowDeviceCredential: Boolean, onSuccess: () -> Unit, onError: (String) -> Unit = {}) {
        val executor = ContextCompat.getMainExecutor(activity)
        val bp = BiometricPrompt(activity, executor, object : BiometricPrompt.AuthenticationCallback() {
            override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) = onSuccess()
            override fun onAuthenticationError(errorCode: Int, errString: CharSequence) = onError(errString.toString())
        })
        val info = BiometricPrompt.PromptInfo.Builder()
            .setTitle(title)
            .setConfirmationRequired(false)     // 顔認証などでも「確認」の追加タップを求めない
            .setAllowedAuthenticators(if (allowDeviceCredential) BIOMETRIC_STRONG or DEVICE_CREDENTIAL else BIOMETRIC_STRONG)
            .apply { if (!allowDeviceCredential) setNegativeButtonText("キャンセル") }
            .build()
        bp.authenticate(info)
    }
}
