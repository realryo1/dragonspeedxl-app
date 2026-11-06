package com.dragonspeedxl.app.engine

/**
 * 対応 SoC の判定。同梱している bin は HTP v81（Snapdragon 8 Elite Gen 5 = SM8850）用だけ。
 * context binary は HTP の世代ごとに別物で、v79 以下は `extended_udma`（v81 以降のみ）が使えず、今の UNet（4 グラフ共有 bin）が作れない
 * （`docs/Research.md`）。このため、対応は v81 だけ。
 *
 * `Build.SOC_MODEL` は「SM8850」のほか、派生版（for Galaxy など）で「SM8850-AC」のように接尾辞つきで返るので、前方一致で見る。
 */
object SocSupport {
    enum class Status { SUPPORTED, UNSUPPORTED, UNKNOWN }

    /** @property prefix `Build.SOC_MODEL` の前方一致。@property htpArch HTP の世代。@property supported この世代の bin を配布しているか。 */
    private data class Soc(val prefix: String, val product: String, val htpArch: String, val supported: Boolean)

    private val table = listOf(
        Soc("SM8850", "Snapdragon 8 Elite Gen 5", "v81", supported = true),
        Soc("SM8750", "Snapdragon 8 Elite", "v79", supported = false),
        Soc("SM8650", "Snapdragon 8 Gen 3", "v75", supported = false),
    )

    data class Result(val status: Status, val socModel: String, val product: String?, val htpArch: String?)

    /** socModel は `Build.SOC_MODEL`（空や "unknown" もありうる）。 */
    fun detect(socModel: String?): Result {
        val raw = socModel?.trim().orEmpty()
        val soc = table.firstOrNull { raw.uppercase().startsWith(it.prefix) }
        val status = when {
            soc == null -> Status.UNKNOWN
            soc.supported -> Status.SUPPORTED
            else -> Status.UNSUPPORTED
        }
        return Result(status, raw.ifEmpty { "不明" }, soc?.product, soc?.htpArch)
    }
}
