package com.dragonspeedxl.app.engine

import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Binder
import android.os.IBinder
import android.system.Os

/**
 * テキストエンコーダ（CLIP-L / CLIP-bigG）を動かす別プロセス（マニフェストの `android:process=":clip"`）。
 * cDSP は 32 ビットで、1 プロセスが使える仮想アドレス空間は約 3.75GB。4 グラフ共有の UNet（3.74GB）と同じプロセスには CLIP を載せられない
 * （載せると context の作成が失敗し、載っていた context も無効になる）。プロセスごとに DSP のセッションが別なので、CLIP はこちらに置く。
 * UNet 側のプロセスとは、ネイティブの UNIX ドメインソケットで話す（engine/clip_service.h）。
 */
class ClipService : Service() {
    // bind されている間は、このプロセスの優先度がアプリ本体（前面）と同じになる（BIND_IMPORTANT）。startService だけだと「キャッシュされたサービス」の
    // 優先度（oom_score_adj 500）で、メモリが逼迫したときに lowmemorykiller に落とされた（2026-10-18。CLIP の読み込み中に落ちて「通信に失敗」になった）。
    // 落とされても、bind が残っていれば（BIND_AUTO_CREATE）システムがプロセスを作り直して onBind を呼ぶ。
    override fun onBind(intent: Intent?): IBinder? {
        val modelDir = intent?.getStringExtra(EXTRA_MODEL_DIR)
        val libDir = intent?.getStringExtra(EXTRA_LIB_DIR)
        if (modelDir != null && libDir != null && ClipHost.loaded) {
            Os.setenv("ADSP_LIBRARY_PATH", libDir, true)   // Skel を探せるように、QNN 初期化より前に設定する
            ClipHost.nativeServe(modelDir, libDir)         // 別スレッドでソケットを待ち受ける（すぐ戻る）
        }
        return Binder()
    }

    companion object {
        private const val EXTRA_MODEL_DIR = "modelDir"
        private const val EXTRA_LIB_DIR = "libDir"
        private var bound = false
        private val conn = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder?) {}
            override fun onServiceDisconnected(name: ComponentName?) {}   // 落とされた。bind が残っているので、システムが作り直す
        }

        /** :clip プロセスの CLIP サービスに bind して起動する（まだなら）。何度呼んでもよい */
        @Synchronized
        fun start(context: Context, modelDir: String, libDir: String) {
            if (bound) return
            val intent = Intent(context, ClipService::class.java).putExtra(EXTRA_MODEL_DIR, modelDir).putExtra(EXTRA_LIB_DIR, libDir)
            bound = context.applicationContext.bindService(intent, conn, Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT)
        }
    }
}

/** :clip プロセスの JNI（libdsxl_native.so の ClipServeAsync） */
object ClipHost {
    var loaded = false
        private set

    init {
        try {
            System.loadLibrary("dsxl_native")
            loaded = true
        } catch (_: Throwable) {
        }
    }

    external fun nativeServe(modelDir: String, libDir: String)
}
