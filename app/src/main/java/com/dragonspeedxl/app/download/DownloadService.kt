package com.dragonspeedxl.app.download

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.dragonspeedxl.app.MainActivity
import com.dragonspeedxl.app.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.launch

/**
 * 裏で取得している間、通知で進み具合を出す（前景サービス。アプリを閉じても、取得は続く）。
 * 開始の前に DownloadStatus.update を呼んでおく。finish で、完了の通知に変えてから止まる。
 */
class DownloadService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var job: Job? = null
    private var lastShown = 0L
    private var lastPercent = -1

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        ensureChannel(this)
        val first = DownloadStatus.state.value?.takeIf { !it.done }
        ServiceCompat.startForeground(
            this,
            NOTIFICATION_ID,
            buildOngoing(first?.title ?: "ダウンロード", first?.text ?: "準備中", first?.fraction),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
        )
        job = scope.launch {
            DownloadStatus.state.filterNotNull().collect { onState(it) }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_NOT_STICKY

    override fun onDestroy() {
        job?.cancel()
        scope.cancel()
        super.onDestroy()
    }

    private fun onState(s: DownloadStatus.State) {
        if (s.done) {
            stopForeground(STOP_FOREGROUND_DETACH)   // 完了の通知は残す（進み具合の通知を、完了の表示に置き換える）
            NotificationManagerCompat.from(this).notify(NOTIFICATION_ID, buildDone(s))
            stopSelf()
            return
        }
        // 通知の更新は、1 秒に 1 回か、％が変わったときだけ（取得の 1MB ごとの更新で通知を連打しない）
        val percent = s.fraction?.let { (it * 100).toInt().coerceIn(0, 100) } ?: -1
        val now = SystemClock.elapsedRealtime()
        if (percent == lastPercent && now - lastShown < 1_000L) return
        lastPercent = percent
        lastShown = now
        NotificationManagerCompat.from(this).notify(NOTIFICATION_ID, buildOngoing(s.title, s.text, s.fraction))
    }

    private fun buildOngoing(title: String, text: String, fraction: Float?): Notification {
        val b = base(title, text).setOngoing(true).setOnlyAlertOnce(true)
        if (fraction == null) b.setProgress(0, 0, true) else b.setProgress(100, (fraction * 100).toInt().coerceIn(0, 100), false)
        return b.build()
    }

    private fun buildDone(s: DownloadStatus.State): Notification =
        base(s.title, s.text).setOngoing(false).setAutoCancel(true).build()

    private fun base(title: String, text: String): NotificationCompat.Builder =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_logo)
            .setContentTitle(title)
            .setContentText(text)
            .setContentIntent(openApp())
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setPriority(NotificationCompat.PRIORITY_LOW)

    private fun openApp(): PendingIntent {
        val i = Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        return PendingIntent.getActivity(this, 0, i, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
    }

    companion object {
        private const val CHANNEL_ID = "downloads"
        private const val NOTIFICATION_ID = 0x4d44   // 16 bit に収まる値

        /** 通知付きの取得を始める。先に DownloadStatus.update で状態を入れてから呼ぶ。 */
        fun start(context: Context) {
            ContextCompat.startForegroundService(context, Intent(context, DownloadService::class.java))
        }

        private fun ensureChannel(context: Context) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
            val nm = context.getSystemService(NotificationManager::class.java)
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                nm.createNotificationChannel(
                    NotificationChannel(CHANNEL_ID, "ダウンロード", NotificationManager.IMPORTANCE_LOW).apply {
                        description = "モデルと LoRA の取得の進み具合"
                    },
                )
            }
        }
    }
}
