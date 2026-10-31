package com.dragonspeedxl.app.lora

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class CivitaiClientTest {
    @Test
    fun tokenGoesOnlyToCivitaiOverHttps() {
        assertTrue(CivitaiClient.isCivitaiHost("https://civitai.com/api/v1/models/1"))
        assertTrue(CivitaiClient.isCivitaiHost("https://CIVITAI.com/api/download/models/2"))
        assertFalse(CivitaiClient.isCivitaiHost("https://evilcivitai.com/api"))
        assertFalse(CivitaiClient.isCivitaiHost("https://civitai.com.evil.example/"))
        assertFalse(CivitaiClient.isCivitaiHost("https://image.civitai.com/cdn/file"))   // CDN（サブドメイン）へは送らない
        assertFalse(CivitaiClient.isCivitaiHost("http://civitai.com/api"))
        assertFalse(CivitaiClient.isCivitaiHost("not a url"))
    }

    @Test
    fun parsesSafetensorsAndSha256() {
        val json = JSONObject(
            """
            {"id":7,"trainedWords":["a","b"],"files":[
              {"name":"x.pt","type":"Model","hashes":{"SHA256":"AA"}},
              {"name":"lora.safetensors","type":"Model","primary":true,"sizeKB":2,
               "hashes":{"SHA256":"ABCDEF0123"}}]}
            """.trimIndent(),
        )
        val v = CivitaiClient.parseVersion(json)
        assertEquals("lora.safetensors", v.fileName)
        assertEquals("abcdef0123", v.sha256)   // 小文字にそろえる
        assertEquals(listOf("a", "b"), v.trainedWords)
    }

    @Test
    fun missingHashIsNull() {
        val json = JSONObject("""{"id":8,"files":[{"name":"y.safetensors","type":"Model"}]}""")
        assertNull(CivitaiClient.parseVersion(json).sha256)
    }

    @Test
    fun rejectsVersionWithoutSafetensors() {
        val json = JSONObject("""{"id":9,"files":[{"name":"y.pt","hashes":{"SHA256":"00"}}]}""")
        assertThrows(IllegalArgumentException::class.java) { CivitaiClient.parseVersion(json) }
    }
}
