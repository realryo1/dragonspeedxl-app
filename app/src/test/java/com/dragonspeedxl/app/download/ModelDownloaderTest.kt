package com.dragonspeedxl.app.download

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ModelDownloaderTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val json = """
        {"repo":"realryo1/dragonspeedxl-models","files":[
          {"path":"models/clip_l.bin","size":4,"sha256":"AB"},
          {"path":"models/gpu/vae_64_fp32.dlc","size":3,"sha256":"cd"},
          {"path":"models/vae_64_w8a16.bin","size":2,"sha256":"ef"}]}
    """.trimIndent()

    @Test
    fun sha256OfKnownText() {
        val f = tmp.newFile("abc.txt").apply { writeText("abc") }
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", sha256Of(f))
    }

    @Test
    fun missingListsAbsentAndWrongSizeOnly() {
        val dir = tmp.newFolder("models")
        File(dir, "clip_l.bin").writeBytes(ByteArray(4))          // 正しいサイズ → 足りない扱いにしない
        File(dir, "gpu").mkdirs()
        File(dir, "gpu/vae_64_fp32.dlc").writeBytes(ByteArray(1)) // サイズが違う → 足りない
        val d = ModelDownloader(dir)
        val missing = d.missing(ModelManifest.parse(json)).map { it.localPath }
        assertEquals(listOf("gpu/vae_64_fp32.dlc", "vae_64_w8a16.bin"), missing)
    }

    @Test
    fun bundledManifestIsValid() {
        // 同梱の manifest（テストは app モジュールのフォルダで動く）。形と、必要なファイルが入っていることを確かめる。
        val m = ModelManifest.parse(File("src/main/assets/model_manifest.json").readText())
        assertEquals(ModelManifest.REPO, m.repo)
        assertEquals(15, m.entries.size)
        assertTrue(m.entries.any { it.localPath == "vae_64_w8a16.bin" })
        assertTrue(m.entries.any { it.localPath == "gpu/vae_64_fp32.dlc" })
        assertTrue(m.entries.any { it.localPath == "unet_lora_4g.bin" })
    }
}
