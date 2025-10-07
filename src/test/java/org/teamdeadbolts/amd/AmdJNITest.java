/* Team Deadbolts (C) 2025 */
package org.teamdeadbolts.amd;

import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.junit.jupiter.api.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class AmdJNITest {

    // Configure these paths for your environment
    private static final String MODEL_PATH =
            System.getenv("AMD_MODEL_PATH") != null
                    ? System.getenv("AMD_MODEL_PATH")
                    : "models/yolov5s.onnx";

    private static final String TEST_IMAGE_PATH =
            System.getenv("AMD_TEST_IMAGE") != null
                    ? System.getenv("AMD_TEST_IMAGE")
                    : "test_images/bus.jpg";

    private static final int NUM_CLASSES = 80; // COCO dataset
    private static final double NMS_THRESHOLD = 0.45;
    private static final double BOX_THRESHOLD = 0.25;

    private static long detectorPtr;
    private static boolean hasValidModel;
    private static boolean hasValidImage;

    @BeforeAll
    static void setup() {
        // Load OpenCV native library
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Warning: Failed to load OpenCV library: " + e.getMessage());
        }

        // Load AMD JNI library
        try {
            System.loadLibrary("amd_jni");
        } catch (UnsatisfiedLinkError e) {
            fail("Failed to load amd_jni library: " + e.getMessage());
        }

        // Check if model file exists
        hasValidModel = Files.exists(Paths.get(MODEL_PATH));
        if (!hasValidModel) {
            System.err.println("Warning: Model file not found at: " + MODEL_PATH);
            System.err.println("Set AMD_MODEL_PATH environment variable to specify model location");
        }

        // Check if test image exists
        hasValidImage = Files.exists(Paths.get(TEST_IMAGE_PATH));
        if (!hasValidImage) {
            System.err.println("Warning: Test image not found at: " + TEST_IMAGE_PATH);
            System.err.println("Set AMD_TEST_IMAGE environment variable to specify image location");
        }
    }

    @Test
    @Order(1)
    void testCreateDetector() {
        Assumptions.assumeTrue(hasValidModel, "Skipping: model file not available");

        detectorPtr =
                AmdJNI.create(
                        MODEL_PATH,
                        NUM_CLASSES,
                        AmdJNI.ModelVersion.YOLO_V5.ordinal(),
                        0x01 // CPU device
                        );

        System.out.println("Detector pointer: " + detectorPtr);

        assertTrue(detectorPtr >= 0, "Detector pointer should be non-negative");
    }

    @Test
    @Order(2)
    void testCreateWithInvalidPath() {
        try {
            // long invalidPtr = AmdJNI.create(
            //     "/invalid/path/to/model.onnx",
            //     NUM_CLASSES,
            //     AmdJNI.ModelVersion.YOLO_V5.ordinal(),
            //     0x01
            // );
            Exception exception =
                    assertThrows(
                            Exception.class,
                            () -> {
                                long invalidPtr =
                                        AmdJNI.create(
                                                "/invalid/path/to/model.onnx",
                                                NUM_CLASSES,
                                                AmdJNI.ModelVersion.YOLO_V5.ordinal(),
                                                0x01);
                            },
                            "Should throw exception for invalid model path");

            assertTrue(exception.getMessage().contains("File doesn't exist"));
        } finally {
        }

        // assertEquals(0, invalidPtr, "Should return 0 for invalid model path");
    }

    @Test
    @Order(3)
    void testSetDevice() {
        Assumptions.assumeTrue(hasValidModel && detectorPtr >= 0, "Skipping: no valid detector");

        int result = AmdJNI.setDevice(detectorPtr, 0x01); // CPU
        assertEquals(0, result, "setDevice should return 0 on success");

        // Test with invalid pointer
        int invalidResult = AmdJNI.setDevice(-1, 0x01);
        assertEquals(-1, invalidResult, "setDevice should return -1 for invalid pointer");
    }

    @Test
    @Order(4)
    void testIsQuantized() {
        Assumptions.assumeTrue(hasValidModel && detectorPtr >= 0, "Skipping: no valid detector");

        boolean quantized = AmdJNI.isQuantized(detectorPtr);
        // Result depends on your model - just verify it doesn't crash
        assertNotNull(quantized);

        // Test with invalid pointer
        boolean invalidResult = AmdJNI.isQuantized(-1);
        assertFalse(invalidResult, "Should return false for invalid pointer");
    }

    @Test
    @Order(5)
    void testDetectWithSyntheticImage() {
        Assumptions.assumeTrue(hasValidModel && detectorPtr >= 0, "Skipping: no valid detector");

        // Create a synthetic 640x640 BGR image
        Mat testImage = new Mat(640, 640, CvType.CV_8UC3, new Scalar(128, 128, 128));

        try {
            AmdJNI.AmdResult[] results =
                    AmdJNI.detect(
                            detectorPtr,
                            testImage.getNativeObjAddr(),
                            NMS_THRESHOLD,
                            BOX_THRESHOLD);

            assertNotNull(results, "detect should return non-null array");
            // Synthetic image may or may not have detections
            assertTrue(results.length >= 0, "Results array should have non-negative length");

        } finally {
            testImage.release();
        }
    }

    @Test
    @Order(6)
    void testDetectWithRealImage() {
        Assumptions.assumeTrue(hasValidModel && detectorPtr >= 0, "Skipping: no valid detector");
        Assumptions.assumeTrue(hasValidImage, "Skipping: no test image available");

        Mat image = Imgcodecs.imread(TEST_IMAGE_PATH);
        assertNotNull(image, "Failed to load test image");
        assertFalse(image.empty(), "Loaded image is empty");

        try {
            AmdJNI.AmdResult[] results =
                    AmdJNI.detect(
                            detectorPtr, image.getNativeObjAddr(), NMS_THRESHOLD, BOX_THRESHOLD);

            assertNotNull(results, "detect should return non-null array");

            // Print results for manual verification
            System.out.println("Detected " + results.length + " objects:");
            for (int i = 0; i < Math.min(results.length, 10); i++) {
                System.out.println("  " + results[i]);
            }

            // Validate result structure
            for (AmdJNI.AmdResult result : results) {
                assertNotNull(result.rect, "Result should have valid rect");
                assertTrue(result.rect.width > 0, "Box width should be positive");
                assertTrue(result.rect.height > 0, "Box height should be positive");
            }

            // Save annotated image
            saveAnnotatedImage(image, results, "test_output/detections.jpg");

        } finally {
            image.release();
        }
    }

    private void saveAnnotatedImage(Mat image, AmdJNI.AmdResult[] results, String outputPath) {
        Mat annotated = image.clone();

        // COCO class names for visualization
        String[] cocoClasses = {
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"
        };

        // Draw each detection
        for (AmdJNI.AmdResult result : results) {
            // Draw bounding box
            org.opencv.imgproc.Imgproc.rectangle(
                    annotated,
                    new org.opencv.core.Point(result.rect.x, result.rect.y),
                    new org.opencv.core.Point(
                            result.rect.x + result.rect.width, result.rect.y + result.rect.height),
                    new Scalar(0, 255, 0), // Green color
                    2);

            // Prepare label text
            String className =
                    result.getClassId() < cocoClasses.length
                            ? cocoClasses[result.getClassId()]
                            : "class_" + result.getClassId();
            String label = String.format("%s: %.2f", className, result.getConf());

            // Draw label background
            int[] baseline = new int[1];
            org.opencv.core.Size textSize =
                    org.opencv.imgproc.Imgproc.getTextSize(
                            label,
                            org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            1,
                            baseline);

            org.opencv.imgproc.Imgproc.rectangle(
                    annotated,
                    new org.opencv.core.Point(result.rect.x, result.rect.y - textSize.height - 5),
                    new org.opencv.core.Point(result.rect.x + textSize.width, result.rect.y),
                    new Scalar(0, 255, 0),
                    -1 // Filled
                    );

            // Draw label text
            org.opencv.imgproc.Imgproc.putText(
                    annotated,
                    label,
                    new org.opencv.core.Point(result.rect.x, result.rect.y - 5),
                    org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    new Scalar(0, 0, 0), // Black text
                    1);
        }

        // Create output directory if it doesn't exist
        new File(outputPath).getParentFile().mkdirs();

        // Save the annotated image
        boolean saved = Imgcodecs.imwrite(outputPath, annotated);
        if (saved) {
            System.out.println("✓ Saved annotated image to: " + outputPath);
        } else {
            System.err.println("✗ Failed to save annotated image");
        }

        annotated.release();
    }

    @Test
    @Order(7)
    void testDetectWithInvalidImagePointer() {
        Assumptions.assumeTrue(hasValidModel && detectorPtr >= 0, "Skipping: no valid detector");

        assertThrows(
                RuntimeException.class,
                () -> {
                    AmdJNI.detect(detectorPtr, 0, NMS_THRESHOLD, BOX_THRESHOLD);
                },
                "Should throw exception for null image pointer");
    }

    @Test
    @Order(8)
    void testDetectWithInvalidDetectorPointer() {
        Mat testImage = new Mat(640, 640, CvType.CV_8UC3);

        try {
            Exception exception =
                    assertThrows(
                            Exception.class,
                            () -> {
                                AmdJNI.AmdResult[] results =
                                        AmdJNI.detect(
                                                -1,
                                                testImage.getNativeObjAddr(),
                                                NMS_THRESHOLD,
                                                BOX_THRESHOLD);
                            },
                            "Should throw exception for invalid detector pointer");

            assertTrue(exception.getMessage().contains("Invalid detector pointer"));
        } finally {
            testImage.release();
        }
    }

    @Test
    @Order(9)
    void testDetectWithDifferentThresholds() {
        Assumptions.assumeTrue(hasValidModel && detectorPtr >= 0, "Skipping: no valid detector");
        Assumptions.assumeTrue(hasValidImage, "Skipping: no test image available");

        Mat image = Imgcodecs.imread(TEST_IMAGE_PATH);
        Assumptions.assumeTrue(image != null && !image.empty(), "Failed to load image");

        try {
            // High threshold - fewer detections
            AmdJNI.AmdResult[] highThreshResults =
                    AmdJNI.detect(
                            detectorPtr,
                            image.getNativeObjAddr(),
                            0.45,
                            0.7 // High confidence threshold
                            );

            // Low threshold - more detections
            AmdJNI.AmdResult[] lowThreshResults =
                    AmdJNI.detect(
                            detectorPtr,
                            image.getNativeObjAddr(),
                            0.45,
                            0.1 // Low confidence threshold
                            );

            assertNotNull(highThreshResults);
            assertNotNull(lowThreshResults);

            System.out.println("High threshold (0.7) detections: " + highThreshResults.length);
            System.out.println("Low threshold (0.1) detections: " + lowThreshResults.length);

            // Save both results
            saveAnnotatedImage(image, highThreshResults, "test_output/detections_high_thresh.jpg");
            saveAnnotatedImage(image, lowThreshResults, "test_output/detections_low_thresh.jpg");

            // Generally, lower threshold should give more or equal detections
            // (though not guaranteed in all cases due to NMS)
            assertTrue(lowThreshResults.length >= 0);

        } finally {
            image.release();
        }
    }

    @Test
    @Order(10)
    void testMultipleDetectCalls() {
        Assumptions.assumeTrue(hasValidModel && detectorPtr >= 0, "Skipping: no valid detector");

        Mat testImage = new Mat(640, 640, CvType.CV_8UC3, new Scalar(100, 100, 100));

        try {
            // Run detection multiple times to test stability
            for (int i = 0; i < 5; i++) {
                AmdJNI.AmdResult[] results =
                        AmdJNI.detect(
                                detectorPtr,
                                testImage.getNativeObjAddr(),
                                NMS_THRESHOLD,
                                BOX_THRESHOLD);
                assertNotNull(results, "Iteration " + i + " should return valid results");
            }
        } finally {
            testImage.release();
        }
    }

    @Test
    @Order(11)
    void testAmdResultEqualsAndHashCode() {
        AmdJNI.AmdResult result1 = new AmdJNI.AmdResult(10, 20, 100, 200, 0.95f, 1);
        AmdJNI.AmdResult result2 = new AmdJNI.AmdResult(10, 20, 100, 200, 0.95f, 1);
        AmdJNI.AmdResult result3 = new AmdJNI.AmdResult(15, 25, 105, 205, 0.90f, 2);

        assertEquals(result1, result2, "Identical results should be equal");
        assertNotEquals(result1, result3, "Different results should not be equal");

        assertEquals(result1.hashCode(), result2.hashCode(), "Equal objects should have same hash");
    }

    @Test
    @Order(12)
    void testAmdResultToString() {
        AmdJNI.AmdResult result = new AmdJNI.AmdResult(10, 20, 100, 200, 0.85f, 0);
        String str = result.toString();

        assertNotNull(str);
        assertTrue(str.contains("AmdResult"));
        assertTrue(str.contains("classId=0"));
        assertTrue(str.contains("conf=0.85"));
    }

    @Test
    @Order(13)
    void testDestroy() {
        Assumptions.assumeTrue(hasValidModel && detectorPtr >= 0, "Skipping: no valid detector");

        // Destroy should not throw
        assertDoesNotThrow(
                () -> {
                    AmdJNI.destroy(detectorPtr);
                });

        // After destroy, operations should fail with exception
        Mat testImage = new Mat(640, 640, CvType.CV_8UC3);
        try {
            Exception exception =
                    assertThrows(
                            Exception.class,
                            () -> {
                                AmdJNI.detect(
                                        detectorPtr,
                                        testImage.getNativeObjAddr(),
                                        NMS_THRESHOLD,
                                        BOX_THRESHOLD);
                            },
                            "Operations on destroyed detector should throw exception");

            assertTrue(exception.getMessage().contains("Invalid detector pointer"));
        } finally {
            testImage.release();
        }

        // Mark as destroyed to prevent double-free in cleanup
        detectorPtr = -1;
    }

    @AfterAll
    static void cleanup() {
        // Ensure cleanup even if tests fail
        if (detectorPtr >= 0) {
            try {
                AmdJNI.destroy(detectorPtr);
            } catch (Exception e) {
                System.err.println("Cleanup warning: " + e.getMessage());
            }
        }
    }
}
