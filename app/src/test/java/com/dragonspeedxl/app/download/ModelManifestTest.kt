package com.dragonspeedxl.app.download

import org.junit.Assert.assertEquals
import org.junit.Test

class ModelManifestTest {
    private val json = """
        {"repo":"realryo1/dragonspeedxl-models","files":[
          {"path":"models/clip_l.bin","size":10,"sha256":"AB"},
          {"path":"models/gpu/vae_64_fp32.dlc","size":20,"sha256":"cd"}]}
    """.trimIndent()

    @Test
    fun mapsRepoPathsToLocalPaths() {
        val m = ModelManifest.parse(json)
        assertEquals("clip_l.bin", m.entries[0].localPath)
        assertEquals("gpu/vae_64_fp32.dlc", m.entries[1].localPath)
        assertEquals("ab", m.entries[0].sha256)   // 小文字にそろえる
        assertEquals(30L, m.totalBytes)
    }

    @Test
    fun buildsResolveUrl() {
        val m = ModelManifest.parse(json)
        assertEquals(
            "https://huggingface.co/realryo1/dragonspeedxl-models/resolve/main/models/gpu/vae_64_fp32.dlc",
            m.url(m.entries[1]),
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsPathOutsideModels() {
        ModelManifest.parse("""{"repo":"x/y","files":[{"path":"other/a.bin","size":1,"sha256":"00"}]}""")
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsParentDirectory() {
        ModelManifest.parse("""{"repo":"x/y","files":[{"path":"models/../a.bin","size":1,"sha256":"00"}]}""")
    }
}
