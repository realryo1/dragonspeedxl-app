package com.dragonspeedxl.app.ui

import androidx.compose.material.icons.materialIcon
import androidx.compose.material.icons.materialPath
import androidx.compose.ui.graphics.vector.ImageVector

/**
 * material-icons-extended にしか無いアイコンのうち、使っている 2 つだけの写し（Icons.Filled.ContentCopy / Download と同じパス）。
 * extended を丸ごと入れると APK の dex が約 30MB 増えるため。ほかのアイコンは material-icons-core の Icons.Filled を使う。
 */
object AppIcons {
    val ContentCopy: ImageVector by lazy {
        materialIcon(name = "Filled.ContentCopy") {
            materialPath {
                moveTo(16.0f, 1.0f)
                lineTo(4.0f, 1.0f)
                curveToRelative(-1.1f, 0.0f, -2.0f, 0.9f, -2.0f, 2.0f)
                verticalLineToRelative(14.0f)
                horizontalLineToRelative(2.0f)
                lineTo(4.0f, 3.0f)
                horizontalLineToRelative(12.0f)
                lineTo(16.0f, 1.0f)
                close()
                moveTo(19.0f, 5.0f)
                lineTo(8.0f, 5.0f)
                curveToRelative(-1.1f, 0.0f, -2.0f, 0.9f, -2.0f, 2.0f)
                verticalLineToRelative(14.0f)
                curveToRelative(0.0f, 1.1f, 0.9f, 2.0f, 2.0f, 2.0f)
                horizontalLineToRelative(11.0f)
                curveToRelative(1.1f, 0.0f, 2.0f, -0.9f, 2.0f, -2.0f)
                lineTo(21.0f, 7.0f)
                curveToRelative(0.0f, -1.1f, -0.9f, -2.0f, -2.0f, -2.0f)
                close()
                moveTo(19.0f, 21.0f)
                lineTo(8.0f, 21.0f)
                lineTo(8.0f, 7.0f)
                horizontalLineToRelative(11.0f)
                verticalLineToRelative(14.0f)
                close()
            }
        }
    }

    val Download: ImageVector by lazy {
        materialIcon(name = "Filled.Download") {
            materialPath {
                moveTo(5.0f, 20.0f)
                horizontalLineToRelative(14.0f)
                verticalLineToRelative(-2.0f)
                horizontalLineTo(5.0f)
                verticalLineTo(20.0f)
                close()
                moveTo(19.0f, 9.0f)
                horizontalLineToRelative(-4.0f)
                verticalLineTo(3.0f)
                horizontalLineTo(9.0f)
                verticalLineToRelative(6.0f)
                horizontalLineTo(5.0f)
                lineToRelative(7.0f, 7.0f)
                lineTo(19.0f, 9.0f)
                close()
            }
        }
    }
}
