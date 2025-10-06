#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "detector.h"
#include "org_teamdeadbolts_amd_AmdJNI.h"

// Global detector storage
std::vector<std::unique_ptr<Detector>> detectors;

extern "C" {

JNIEXPORT jlong JNICALL Java_org_teamdeadbolts_amd_AmdJNI_create(
    JNIEnv* env, jclass clazz, jstring modelPath, jint numClasses,
    jint modelVer, jint deviceMask) {
  const char* pathCStr = env->GetStringUTFChars(modelPath, nullptr);
  if (!pathCStr) {
    return 0;
  }

  try {
    auto detector =
        std::make_unique<Detector>(pathCStr, numClasses, modelVer, deviceMask);
    detectors.push_back(std::move(detector));
    env->ReleaseStringUTFChars(modelPath, pathCStr);

    return static_cast<jlong>(detectors.size() - 1);
  } catch (const std::exception& e) {
    env->ReleaseStringUTFChars(modelPath, pathCStr);

    // Optionally throw Java exception with error message
    jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
    if (exceptionClass) {
      env->ThrowNew(exceptionClass, e.what());
    }

    return 0;
  }
}

JNIEXPORT jint JNICALL Java_org_teamdeadbolts_amd_AmdJNI_setDevice(
    JNIEnv* env, jclass clazz, jlong ptr, jint desiredDevice) {
  if (ptr < 0 || ptr >= (jlong)detectors.size() || !detectors[ptr]) {
    return -1;
  }

  try {
    detectors[ptr]->setDevice(desiredDevice);
    return 0;
  } catch (...) {
    return -1;
  }
}

JNIEXPORT void JNICALL Java_org_teamdeadbolts_amd_AmdJNI_destroy(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong ptr) {
  if (ptr >= 0 && ptr < (jlong)detectors.size()) {
    detectors[ptr].reset();
  }
}

JNIEXPORT jobjectArray JNICALL Java_org_teamdeadbolts_amd_AmdJNI_detect(
    JNIEnv* env, jclass clazz, jlong ptr, jlong imagePtr, jdouble nmsThresh,
    jdouble boxThresh) {
  // Validate detector pointer
  if (ptr < 0 || ptr >= (jlong)detectors.size() || !detectors[ptr]) {
    jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
    if (exceptionClass) {
      env->ThrowNew(exceptionClass, "Invalid detector pointer");
    }
    return nullptr;
  }

  // Validate image pointer
  if (imagePtr == 0) {
    jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
    if (exceptionClass) {
      env->ThrowNew(exceptionClass, "Invalid image pointer");
    }
    return nullptr;
  }

  cv::Mat* image = reinterpret_cast<cv::Mat*>(imagePtr);

  try {
    // Run inference
    std::vector<DetectionBox> results =
        detectors[ptr]->detect(*image, (float)nmsThresh, (float)boxThresh);

    // Find AmdResult class and constructor
    jclass amdResultClass =
        env->FindClass("org/teamdeadbolts/amd/AmdJNI$AmdResult");
    if (!amdResultClass) {
      return nullptr;
    }

    jmethodID constructor =
        env->GetMethodID(amdResultClass, "<init>", "(IIIIFI)V");
    if (!constructor) {
      return nullptr;
    }

    // Create result array
    jobjectArray resultArray =
        env->NewObjectArray(results.size(), amdResultClass, nullptr);
    if (!resultArray) {
      return nullptr;
    }

    // Populate array with detections
    for (size_t i = 0; i < results.size(); i++) {
      const auto& det = results[i];

      // Convert center-width format to left-top-right-bottom
      int left = (int)(det.x - det.w / 2);
      int top = (int)(det.y - det.h / 2);
      int right = (int)(det.x + det.w / 2);
      int bottom = (int)(det.y + det.h / 2);

      // Create AmdResult object
      jobject resultObj =
          env->NewObject(amdResultClass, constructor, left, top, right, bottom,
                         det.confidence, det.classId);

      if (!resultObj) {
        continue;  // Skip this detection if creation failed
      }

      env->SetObjectArrayElement(resultArray, i, resultObj);
      env->DeleteLocalRef(resultObj);
    }

    return resultArray;

  } catch (const std::exception& e) {
    jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
    if (exceptionClass) {
      env->ThrowNew(exceptionClass, e.what());
    }
    return nullptr;
  }
}

JNIEXPORT jboolean JNICALL Java_org_teamdeadbolts_amd_AmdJNI_isQuantized(
    JNIEnv* env, jclass clazz, jlong ptr) {
  if (ptr < 0 || ptr >= (jlong)detectors.size() || !detectors[ptr]) {
    return JNI_FALSE;
  }

  try {
    return detectors[ptr]->isQuantized() ? JNI_TRUE : JNI_FALSE;
  } catch (...) {
    return JNI_FALSE;
  }
}

}  // extern "C"