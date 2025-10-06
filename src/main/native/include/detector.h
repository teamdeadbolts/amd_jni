#ifndef DETECTOR_H
#define DETECTOR_H

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct DetectionBox {
  float x, y, w, h;
  float confidence;
  int classId;
};

class Detector {
 public:
  Detector(const std::string& model_path, int numClasses, int modelVersion,
           int deviceMask);

  std::vector<DetectionBox> detect(cv::Mat& img, float nmsThresh,
                                   float boxThresh);
  bool isQuantized() const;
  void setDevice(int deviceMask);

  int getNumClasses() const { return numClasses; }
  int getModelVer() const { return modelVersion; }

 private:
  Ort::Env env;
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions allocator;

  int numClasses;
  int modelVersion;  // 0=YOLOv5, 1=YOLOv8, 2=YOLOv11
  int deviceMask;

  std::vector<Ort::AllocatedStringPtr> inputNamesAlloc;
  std::vector<Ort::AllocatedStringPtr> outputNamesAlloc;

  std::vector<const char*> inputNames;
  std::vector<const char*> outputNames;
  std::vector<int64_t> inputShape;
  std::vector<int64_t> outputShape;

  std::vector<DetectionBox> parseYOLOv5Output(Ort::Value& output, int origW,
                                              int origH, int inputW, int inputH,
                                              float boxThresh);
  std::vector<DetectionBox> parseYOLOv8Output(Ort::Value& output, int origW,
                                              int origH, int inputW, int inputH,
                                              float boxThresh);
  std::vector<DetectionBox> applyNMS(std::vector<DetectionBox>& boxes,
                                     float nmsThresh);
  float calculateIOU(const DetectionBox& a, const DetectionBox& b);

 private:
  std::vector<DetectionBox> parseOutputAndNMS(Ort::Value& output, int origW,
                                              int origH, int inputW, int inputH,
                                              float nmsThresh, float boxThresh);
};

#endif
