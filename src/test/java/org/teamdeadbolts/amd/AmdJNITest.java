package org.teamdeadbolts.amd;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class AmdJNITest {

    // Path to a dummy or real ONNX model
    private static final String MODEL_PATH = "/path/to/your/model.onnx";

    private static long detectorPtr;

    @BeforeAll
    static void setup() {
        try {
            // Load the native library
            System.loadLibrary("amd_jni");
        } catch (UnsatisfiedLinkError e) {
            fail("Failed to load amd_jni library: " + e.getMessage());
        }

        // Create a detector
        detectorPtr = AmdJNI.create(MODEL_PATH, 80, AmdJNI.ModelVersion.YOLO_V5.ordinal(), 0);
        assertTrue(detectorPtr >= 0, "Detector pointer should be non-negative");
    }

    @Test
    void testSetDevice() {
        int result = AmdJNI.setDevice(detectorPtr, 0); // dummy device mask
        assertEquals(0, result, "setDevice should return 0");
    }

    @Test
    void testIsQuantized() {
        boolean quant = AmdJNI.isQuantized(detectorPtr);
        assertFalse(quant, "Stubbed isQuantized should return false");
    }

    @Test
    void testDetectReturnsArray() {
        // Create dummy image pointer (0 for now, since our C++ stub ignores it)
        AmdJNI.AmdResult[] results = AmdJNI.detect(detectorPtr, 0, 0.5, 0.3);
        assertNotNull(results, "detect should return non-null array");
        assertEquals(0, results.length, "Stubbed detect should return empty array");
    }

    // @Test
    // void testDestroy() {
    //     long result = AmdJNI.destroy(detectorPtr);
    //     assertEquals(0, result, "destroy should return 0 on success");
    // }
}
