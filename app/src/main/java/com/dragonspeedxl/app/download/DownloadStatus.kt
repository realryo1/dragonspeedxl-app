package com.dragonspeedxl.app.download

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * 裏で進んでいる取得（モデル・LoRA）の状態。ViewModel が書き、DownloadService が通知に写す。
 * fraction が null のときは、進み具合が分からない（くるくる回す表示）。
 */
object DownloadStatus {
    data class State(
        val title: String,
        val text: String,
        val fraction: Float?,
        val done: Boolean = false,
        val failed: Boolean = false,
    )

    private val _state = MutableStateFlow<State?>(null)
    val state: StateFlow<State?> = _state.asStateFlow()

    fun update(title: String, text: String, fraction: Float?) {
        _state.value = State(title, text, fraction)
    }

    /** 終わった（成功・失敗）。通知を完了の表示に変えてから、サービスが止まる。 */
    fun finish(title: String, text: String, failed: Boolean = false) {
        _state.value = State(title, text, null, done = true, failed = failed)
    }
}
