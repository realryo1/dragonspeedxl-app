package com.dragonspeedxl.app.engine

import com.dragonspeedxl.app.engine.SocSupport.Status
import org.junit.Assert.assertEquals
import org.junit.Test

class SocSupportTest {
    @Test fun sm8850IsSupported() {
        val r = SocSupport.detect("SM8850")
        assertEquals(Status.SUPPORTED, r.status)
        assertEquals("v81", r.htpArch)
    }

    @Test fun suffixedVariantIsSupported() {
        assertEquals(Status.SUPPORTED, SocSupport.detect("SM8850-AC").status)
        assertEquals(Status.SUPPORTED, SocSupport.detect(" sm8850-ac ").status)
    }

    @Test fun olderGenerationsAreKnownButUnsupported() {
        assertEquals(Status.UNSUPPORTED, SocSupport.detect("SM8750").status)
        assertEquals(Status.UNSUPPORTED, SocSupport.detect("SM8650-AB").status)
    }

    @Test fun otherOrEmptyIsUnknown() {
        assertEquals(Status.UNKNOWN, SocSupport.detect("SM8550").status)
        assertEquals(Status.UNKNOWN, SocSupport.detect("").status)
        assertEquals(Status.UNKNOWN, SocSupport.detect(null).status)
        assertEquals("不明", SocSupport.detect(null).socModel)
    }
}
