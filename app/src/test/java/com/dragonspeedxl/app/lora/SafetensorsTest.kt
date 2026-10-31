package com.dragonspeedxl.app.lora

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class SafetensorsTest {
    @get:Rule
    val tmp = TemporaryFolder()

    /** safetensors の形（8 バイトの長さ、ヘッダー、データ）のファイルを作る。 */
    private fun write(header: String, data: ByteArray): File {
        val h = header.toByteArray(Charsets.UTF_8)
        val f = File(tmp.root, "t.safetensors")
        f.outputStream().use { o ->
            o.write(ByteArray(8) { i -> (h.size.toLong() shr (8 * i)).toByte() })
            o.write(h)
            o.write(data)
        }
        return f
    }

    /** 1.0f と 2.0f（リトルエンディアン） */
    private val twoFloats = byteArrayOf(0, 0, 0x80.toByte(), 0x3f, 0, 0, 0, 0x40)

    @Test
    fun readsFloat32InsideTheFile() {
        val f = write("""{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}""", twoFloats)
        Safetensors(f).use { sf -> assertArrayEquals(floatArrayOf(1f, 2f), sf.floats("w"), 0f) }
    }

    @Test
    fun rejectsOffsetsPastTheEndOfFile() {
        val f = write("""{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,64]}}""", twoFloats)
        assertThrows(IllegalArgumentException::class.java) { Safetensors(f).close() }
    }

    @Test
    fun rejectsNegativeStart() {
        val f = write("""{"w":{"dtype":"F32","shape":[2],"data_offsets":[-8,8]}}""", twoFloats)
        assertThrows(IllegalArgumentException::class.java) { Safetensors(f).close() }
    }

    @Test
    fun rejectsShapeThatDoesNotMatchSize() {
        val f = write("""{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}""", twoFloats)
        assertThrows(IllegalArgumentException::class.java) { Safetensors(f).close() }
    }

    @Test
    fun rejectsNegativeDimension() {
        val f = write("""{"w":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}}""", twoFloats)
        assertThrows(IllegalArgumentException::class.java) { Safetensors(f).close() }
    }
}
