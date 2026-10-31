package com.dragonspeedxl.app.download

import org.json.JSONObject

/**
 * 配布する 1 ファイル。
 * repoPath は Hugging Face の中のパス（先頭に models/ が付く）、localPath は端末の models/ の中の相対パス。
 */
data class ModelEntry(val repoPath: String, val localPath: String, val size: Long, val sha256: String)

/** Hugging Face の model repo に置いた manifest.json（ファイルの一覧・サイズ・sha256）。 */
class ModelManifest(val repo: String, val entries: List<ModelEntry>) {
    val totalBytes: Long get() = entries.sumOf { it.size }

    fun url(entry: ModelEntry): String = "https://huggingface.co/$repo/resolve/main/${entry.repoPath}"

    companion object {
        /** 公開のモデル repo（CC BY-NC-SA 4.0）。 */
        const val REPO = "realryo1/dragonspeedxl-models"
        /** APK の assets に同梱した manifest の名前（HF に上げたものと同じ。モデルを更新したら、ここも差し替える）。 */
        const val ASSET = "model_manifest.json"
        private const val PREFIX = "models/"

        /** manifest の JSON を読む。models/ の外を指すパスは受け付けない。 */
        fun parse(json: String): ModelManifest {
            val o = JSONObject(json)
            val files = o.getJSONArray("files")
            val entries = (0 until files.length()).map { i ->
                val f = files.getJSONObject(i)
                val repoPath = f.getString("path")
                require(repoPath.startsWith(PREFIX)) { "想定外のパス: $repoPath" }
                val local = repoPath.removePrefix(PREFIX)
                require(local.isNotEmpty() && !local.startsWith("/") && !local.split("/").contains("..")) { "不正なパス: $repoPath" }
                ModelEntry(repoPath, local, f.getLong("size"), f.getString("sha256").lowercase())
            }
            return ModelManifest(o.getString("repo"), entries)
        }
    }
}
