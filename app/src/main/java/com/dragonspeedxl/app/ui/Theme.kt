package com.dragonspeedxl.app.ui

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

/** 全要素で共通の角丸（基準はチップ入力エリアの外枠）。 */
val AppShape = RoundedCornerShape(12.dp)
private val AppShapes = Shapes(extraSmall = AppShape, small = AppShape, medium = AppShape, large = AppShape, extraLarge = AppShape)

// Android 12 未満（minSdk は 31 なので実際には使わない）向けの保険
private val FallbackDark = darkColorScheme(primary = Color(0xFF39C5BB), secondary = Color(0xFFE91E8C))
private val FallbackLight = lightColorScheme(primary = Color(0xFF00696A), secondary = Color(0xFFB0006A))

/** Material You（Material 3）。壁紙から作られる動的カラーを使い、明暗はシステムの設定に従う。 */
@Composable
fun AppTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val scheme = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> FallbackDark
        else -> FallbackLight
    }
    // 標準の background / surface はほぼ無彩色で、通知センターなどシステム UI より薄く見える。
    // システムと同じ系統の「色味の強い面」（surfaceContainer 系）を、背景・カード・補助面に当てる。
    val tuned = scheme.copy(
        background = scheme.surfaceContainer,
        surface = scheme.surfaceContainerHigh,
        surfaceVariant = scheme.surfaceContainerHighest,
    )
    // 背景との明暗差が強すぎないよう、文字色と枠線を背景側へ少し寄せる（色相は壁紙のまま）
    val bg = tuned.background
    val soft = tuned.copy(
        onBackground = lerp(tuned.onBackground, bg, 0.20f),
        onSurface = lerp(tuned.onSurface, bg, 0.20f),
        onSurfaceVariant = lerp(tuned.onSurfaceVariant, bg, 0.12f),
        outline = lerp(tuned.outline, bg, 0.45f),
        outlineVariant = lerp(tuned.outlineVariant, bg, 0.30f),
    )
    MaterialTheme(colorScheme = soft, shapes = AppShapes, content = content)
}
