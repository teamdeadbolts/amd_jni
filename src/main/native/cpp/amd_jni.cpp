#include "org_teamdeadbolts_amd_AmdJNI.h"
#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

struct Detector {
  Ort::Env env;
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;

  Detector(const std::string &model_path, int modelVer, int deviceMask) : env(ORT_LOGGING_LEVEL_INFO, "AMDJNI") {
    session_options.SetInterOpNumThreads(1);
    // TOOD: Device mask stuff

    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
  }
};

std::vector<std::unique_ptr<Detector>> detectors;


// ---------------- JNI Methods ----------------

extern "C" {
  JNIEXPORT jlong JNICALL Java_org_teamdeadbolts_amd_AmdJNI_create
    (JNIEnv *env, jclass clazz, jstring modelPath, jint numClasses, jint modelVer, jint deviceMask)
  {
      const char* pathCStr = env->GetStringUTFChars(modelPath, nullptr);
      if (!pathCStr) return 0;

      try {
          auto detector = std::make_unique<Detector>(pathCStr, modelVer, deviceMask);
          detectors.push_back(std::move(detector));
          env->ReleaseStringUTFChars(modelPath, pathCStr);

          // Return index as "pointer" (simplified)
          return static_cast<jlong>(detectors.size() - 1);
      } catch (...) {
          env->ReleaseStringUTFChars(modelPath, pathCStr);
          return 0;
      }
  }

  /*
  * Class:     org_teamdeadbolts_amd_AmdJNI
  * Method:    setDevice
  */
  JNIEXPORT jint JNICALL Java_org_teamdeadbolts_amd_AmdJNI_setDevice
    (JNIEnv *env, jclass clazz, jlong ptr, jint desiredDevice)
  {
      // TODO: implement GPU/NPU switching
      // For now, just return success
      return 0;
  }

  /*
  * Class:     org_teamdeadbolts_amd_AmdJNI
  * Method:    destroy
  */
  JNIEXPORT void JNICALL Java_org_teamdeadbolts_amd_AmdJNI_destroy
    (JNIEnv *, jclass, jlong ptr)
  {
    if (ptr >= 0 && ptr < (jlong)detectors.size()) {
        detectors[ptr].reset();
    }
  }

  /*
  * Class:     org_teamdeadbolts_amd_AmdJNI
  * Method:    detect
  */
  JNIEXPORT jobjectArray JNICALL Java_org_teamdeadbolts_amd_AmdJNI_detect
    (JNIEnv *env, jclass clazz, jlong ptr, jlong imagePtr, jdouble nmsThresh, jdouble boxThresh)
  {
      // TODO: Replace with real ONNX inference
      // For now, return empty array of AmdResult
      jclass amdResultClass = env->FindClass("org/teamdeadbolts/amd/AmdJNI$AmdResult");
      if (!amdResultClass) return nullptr;

      jobjectArray resultArray = env->NewObjectArray(0, amdResultClass, nullptr);
      return resultArray;
  }

  /*
  * Class:     org_teamdeadbolts_amd_AmdJNI
  * Method:    isQuantized
  */
  JNIEXPORT jboolean JNICALL Java_org_teamdeadbolts_amd_AmdJNI_isQuantized
    (JNIEnv *env, jclass clazz, jlong ptr)
  {
      // TODO: implement real quantization detection
      return JNI_FALSE;
  }
}
