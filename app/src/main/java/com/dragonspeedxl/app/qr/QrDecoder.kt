package com.dragonspeedxl.app.qr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.google.zxing.BarcodeFormat
import com.google.zxing.BinaryBitmap
import com.google.zxing.DecodeHintType
import com.google.zxing.MultiFormatReader
import com.google.zxing.RGBLuminanceSource
import com.google.zxing.common.HybridBinarizer
import java.io.File

/**
 * 写真（端末のカメラアプリで撮ったもの）から QR を読む。ZXing だけを使う（アプリはカメラの権限を持たない）。
 * 写真は数千万画素と大きいので、長辺を縮めてから読む。読めなければ、さらに小さくして読み直す。
 * QR は向きに依らず読めるので、写真の回転（EXIF）は見ない。重いので、メインスレッドでは呼ばない。
 */
object QrDecoder {
    /** 読むときの長辺の上限（この順に試す）。 */
    private val MAX_SIDES = intArrayOf(2048, 1024)

    private val HINTS = mapOf(
        DecodeHintType.POSSIBLE_FORMATS to listOf(BarcodeFormat.QR_CODE),
        DecodeHintType.TRY_HARDER to true,
    )

    /** 写真から QR の文字列を読む。見つからなければ null。 */
    fun decode(photo: File): String? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(photo.path, bounds)
        val longest = maxOf(bounds.outWidth, bounds.outHeight)
        if (longest <= 0) return null
        for (maxSide in MAX_SIDES) {
            var sample = 1
            while (longest / sample > maxSide) sample *= 2   // 2 のべき乗で縮める（読み込みの段階で縮むので、メモリを食わない）
            val bmp = BitmapFactory.decodeFile(photo.path, BitmapFactory.Options().apply { inSampleSize = sample }) ?: continue
            try {
                decode(bmp)?.let { return it }
            } finally {
                bmp.recycle()
            }
        }
        return null
    }

    private fun decode(bmp: Bitmap): String? {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        return try {
            MultiFormatReader().decode(BinaryBitmap(HybridBinarizer(RGBLuminanceSource(w, h, pixels))), HINTS).text
        } catch (_: Exception) {
            null
        }
    }
}
