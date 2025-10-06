
package org.teamdeadbolts.amd;

import org.opencv.core.Point;
import org.opencv.core.Rect2d;

public class AmdJNI {
  public static enum ModelVersion {
    YOLO_V5,
    YOLO_V8,
    YOLO_V11
  }

  public static class AmdResult {
    public final Rect2d rect;
    final float conf;
    final int classId;

    public AmdResult(int left, int top, int right, int bottom, float conf, int classId) {
      this.rect = new Rect2d(new Point(left, top), new Point(right, bottom));
      this.conf = conf;
      this.classId = classId;
    }

    @Override
    public String toString() {
      return String.format("AmdResult[classId=%d, conf=%.2f, rect=%s]", classId, conf, rect.toString());
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((rect == null) ? 0 : rect.hashCode());
      result = prime * result + Float.floatToIntBits(conf);
      result = prime * result + classId;
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) return true;
      if (obj == null) return false;
      if (getClass() != obj.getClass()) return false;
      AmdResult other = (AmdResult) obj;
      if (rect == null) {
        if (other.rect != null) return false;
      } else if (!rect.equals(other.rect)) return false;
      if (Float.floatToIntBits(conf) != Float.floatToIntBits(other.conf)) return false;
      if (classId != other.classId) return false;
      return true;
    }

    public float getConf() {
      return conf;
    }

    public int getClassId() {
      return classId;
    }
  }

  /**
   * Create an AMD (ONNX Runtime) detector. Returns valid pointer on success, or NULL on error.
   *
   * @param modelPath Absolute path to the ONNX model on disk
   * @param numClasses Number of classes. MUST MATCH or native code may segfault
   * @param modelVer Which YOLO model is being used
   * @param deviceMask Bitmask or flag for selecting NPU/GPU/CPU
   * @return Pointer to the detector in native memory
  */
  public static native long create(String modelPath, int numClasses, int modelVer, int deviceMask);

  /**
   * Change devie selection. May be NPU/GPU/CPU
   */
  public static native int setDevice(long ptr, int deviceMask);

  /** Delete all native resources associated with a detector */
  public static native void destroy(long ptr);

  /**
   * Run the detection
   * @param detectorPtr Pointer to detector created above
   * @param imagePtr Pointer to a cv::Mat input image
   * @param nmsThresh Non-maximum suppression threshold
   * @param boxThresh Minimum box confidence threshold
   * @return Array of detection results
  */
  public static native AmdResult[] detect(long ptr, long imgPtr, double nmsThresh, double boxThresh);

  /**
   * Check if a model is quantized
   * 
   * @param ptr Pointer to detector created above
   * @return true if quantized, false if not or on error
   */
  public static native boolean isQuantized(long ptr);
}
