#include "detector.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>

Detector::Detector(const std::string& model_path, int numClasses,
                   int modelVersion, int deviceMask)
    : env(ORT_LOGGING_LEVEL_WARNING, "AMDJNI"),
      numClasses(numClasses),
      modelVersion(modelVersion),
      deviceMask(deviceMask) {
  //  static std::ofstream logFile("detector_debug.log", std::ios::app);
  //   std::cout.rdbuf(logFile.rdbuf());

  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Device selection based on mask
  if (deviceMask & 0x01) {         // CPU
                                   // Default provider
  } else if (deviceMask & 0x02) {  // GPU (CUDA if available)
                                   // OrtCUDAProviderOptions cuda_options;
    // session_options.AppendExecutionProvider_CUDA(cuda_options);
  } else if (deviceMask & 0x04) {  // NPU/VitisAI
    // session_options.AppendExecutionProvider("VitisAI");
  }

  session =
      std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

  // Get input/output info
  size_t numInputNodes = session->GetInputCount();
  size_t numOutputNodes = session->GetOutputCount();

  // Reserve space to avoid reallocation
  inputNamesAlloc.reserve(numInputNodes);
  inputNames.reserve(numInputNodes);
  outputNamesAlloc.reserve(numOutputNodes);
  outputNames.reserve(numOutputNodes);

  // Get input names and info
  for (size_t i = 0; i < numInputNodes; i++) {
    Ort::AllocatedStringPtr nameAlloc =
        session->GetInputNameAllocated(i, allocator);
    const char* namePtr = nameAlloc.get();

    inputNamesAlloc.push_back(std::move(nameAlloc));
    inputNames.push_back(
        inputNamesAlloc.back().get());  // Get pointer from stored allocation

    auto typeInfo = session->GetInputTypeInfo(i);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    inputShape = tensorInfo.GetShape();
  }

  // Get output names and info
  for (size_t i = 0; i < numOutputNodes; i++) {
    Ort::AllocatedStringPtr nameAlloc =
        session->GetOutputNameAllocated(i, allocator);
    const char* namePtr = nameAlloc.get();

    outputNamesAlloc.push_back(std::move(nameAlloc));
    outputNames.push_back(
        outputNamesAlloc.back().get());  // Get pointer from stored allocation

    auto typeInfo = session->GetOutputTypeInfo(i);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    outputShape = tensorInfo.GetShape();
  }
}

std::vector<DetectionBox> Detector::detect(cv::Mat& image, float nmsThresh,
                                           float boxThresh) {
  // Preprocess image
  cv::Mat inputBlob;
  int inputW = inputShape[3];
  int inputH = inputShape[2];

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(inputW, inputH));
  resized.convertTo(inputBlob, CV_32F, 1.0 / 255.0);

  // Convert HWC to CHW
  cv::Mat channels[3];
  cv::split(inputBlob, channels);

  // Check what type the model expects
  auto typeInfo = session->GetInputTypeInfo(0);
  auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
  auto elementType = tensorInfo.GetElementType();

  if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    // Model expects float16 - use Ort::Float16_t
    std::vector<Ort::Float16_t> inputTensorValues;
    inputTensorValues.reserve(inputW * inputH * 3);

    for (int c = 0; c < 3; c++) {
      float* channelData = (float*)channels[c].data;
      for (int i = 0; i < inputW * inputH; i++) {
        inputTensorValues.push_back(Ort::Float16_t(channelData[i]));
      }
    }

    // Create float16 tensor
    std::vector<int64_t> inputShapeVec = {1, 3, inputH, inputW};
    auto memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
        inputShapeVec.data(), inputShapeVec.size());

    // Run inference
    auto outputTensors =
        session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                     1, outputNames.data(), outputNames.size());

    return parseOutputAndNMS(outputTensors[0], image.cols, image.rows, inputW,
                             inputH, nmsThresh, boxThresh);

  } else {
    // Model expects float32 (original code)
    std::vector<float> inputTensorValues;
    inputTensorValues.reserve(inputW * inputH * 3);

    for (int c = 0; c < 3; c++) {
      inputTensorValues.insert(inputTensorValues.end(),
                               (float*)channels[c].data,
                               (float*)channels[c].data + inputW * inputH);
    }

    // Create float32 tensor
    std::vector<int64_t> inputShapeVec = {1, 3, inputH, inputW};
    auto memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
        inputShapeVec.data(), inputShapeVec.size());

    // Run inference
    auto outputTensors =
        session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                     1, outputNames.data(), outputNames.size());

    return parseOutputAndNMS(outputTensors[0], image.cols, image.rows, inputW,
                             inputH, nmsThresh, boxThresh);
  }
}

// Add helper method to avoid code duplication
std::vector<DetectionBox> Detector::parseOutputAndNMS(Ort::Value& output,
                                                      int origW, int origH,
                                                      int inputW, int inputH,
                                                      float nmsThresh,
                                                      float boxThresh) {
  std::vector<DetectionBox> detections;

  if (modelVersion == 0) {  // YOLOv5
    std::cout << "Using YOLOv5 parser..." << std::endl;
    detections =
        parseYOLOv5Output(output, origW, origH, inputW, inputH, boxThresh);
  } else {  // YOLOv8/v11
    std::cout << "Using YOLOv8/v11 parser..." << std::endl;
    detections =
        parseYOLOv8Output(output, origW, origH, inputW, inputH, boxThresh);
  }

  auto result = applyNMS(detections, nmsThresh);
  return result;
}

std::vector<DetectionBox> Detector::parseYOLOv5Output(Ort::Value& output,
                                                      int origW, int origH,
                                                      int inputW, int inputH,
                                                      float boxThresh) {
  std::vector<DetectionBox> boxes;
  auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
  auto outputType = output.GetTensorTypeAndShapeInfo().GetElementType();

  // YOLOv5 format: [1, num_anchors, 5+num_classes]
  int numBoxes = shape[1];
  int boxDim = shape[2];

  float scaleX = (float)origW / inputW;
  float scaleY = (float)origH / inputH;

  if (outputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    // Handle float16 output
    Ort::Float16_t* outputData = output.GetTensorMutableData<Ort::Float16_t>();

    for (int i = 0; i < numBoxes; i++) {
      Ort::Float16_t* box = outputData + i * boxDim;
      float objectness = static_cast<float>(box[4]);

      if (objectness < boxThresh) continue;

      // Find best class
      int bestClass = 0;
      float bestScore = 0;
      for (int c = 0; c < numClasses; c++) {
        float score = static_cast<float>(box[5 + c]) * objectness;
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < boxThresh) continue;

      DetectionBox det;
      det.x = static_cast<float>(box[0]) * scaleX;
      det.y = static_cast<float>(box[1]) * scaleY;
      det.w = static_cast<float>(box[2]) * scaleX;
      det.h = static_cast<float>(box[3]) * scaleY;
      det.confidence = bestScore;
      det.classId = bestClass;

      boxes.push_back(det);
    }
  } else {
    // Handle float32 output (original code)
    float* outputData = output.GetTensorMutableData<float>();

    for (int i = 0; i < numBoxes; i++) {
      float* box = outputData + i * boxDim;
      float objectness = box[4];

      if (objectness < boxThresh) continue;

      // Find best class
      int bestClass = 0;
      float bestScore = 0;
      for (int c = 0; c < numClasses; c++) {
        float score = box[5 + c] * objectness;
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < boxThresh) continue;

      DetectionBox det;
      det.x = box[0] * scaleX;
      det.y = box[1] * scaleY;
      det.w = box[2] * scaleX;
      det.h = box[3] * scaleY;
      det.confidence = bestScore;
      det.classId = bestClass;

      boxes.push_back(det);
    }
  }

  return boxes;
}

std::vector<DetectionBox> Detector::parseYOLOv8Output(Ort::Value& output,
                                                      int origW, int origH,
                                                      int inputW, int inputH,
                                                      float boxThresh) {
  std::vector<DetectionBox> boxes;
  float* outputData = output.GetTensorMutableData<float>();
  auto shape = output.GetTensorTypeAndShapeInfo().GetShape();

  // YOLOv8 format: [1, 4+num_classes, num_anchors]
  int boxDim = shape[1];
  int numBoxes = shape[2];

  float scaleX = (float)origW / inputW;
  float scaleY = (float)origH / inputH;

  for (int i = 0; i < numBoxes; i++) {
    // Find best class
    int bestClass = 0;
    float bestScore = 0;
    for (int c = 0; c < numClasses; c++) {
      float score = outputData[(4 + c) * numBoxes + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }

    if (bestScore < boxThresh) continue;

    DetectionBox det;
    det.x = outputData[0 * numBoxes + i] * scaleX;
    det.y = outputData[1 * numBoxes + i] * scaleY;
    det.w = outputData[2 * numBoxes + i] * scaleX;
    det.h = outputData[3 * numBoxes + i] * scaleY;
    det.confidence = bestScore;
    det.classId = bestClass;

    boxes.push_back(det);
  }

  return boxes;
}

std::vector<DetectionBox> Detector::applyNMS(std::vector<DetectionBox>& boxes,
                                             float nmsThresh) {
  // Sort by confidence
  std::sort(boxes.begin(), boxes.end(),
            [](const DetectionBox& a, const DetectionBox& b) {
              return a.confidence > b.confidence;
            });

  std::vector<bool> suppressed(boxes.size(), false);
  std::vector<DetectionBox> result;

  for (size_t i = 0; i < boxes.size(); i++) {
    if (suppressed[i]) continue;
    result.push_back(boxes[i]);

    for (size_t j = i + 1; j < boxes.size(); j++) {
      if (suppressed[j]) continue;
      if (boxes[i].classId != boxes[j].classId) continue;

      float iou = calculateIOU(boxes[i], boxes[j]);
      if (iou > nmsThresh) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

float Detector::calculateIOU(const DetectionBox& a, const DetectionBox& b) {
  float x1 = std::max(a.x - a.w / 2, b.x - b.w / 2);
  float y1 = std::max(a.y - a.h / 2, b.y - b.h / 2);
  float x2 = std::min(a.x + a.w / 2, b.x + b.w / 2);
  float y2 = std::min(a.y + a.h / 2, b.y + b.h / 2);

  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float union_area = a.w * a.h + b.w * b.h - intersection;

  return union_area > 0 ? intersection / union_area : 0;
}

bool Detector::isQuantized() const {
  try {
    auto typeInfo = session->GetInputTypeInfo(0);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    auto elementType = tensorInfo.GetElementType();

    // Check if model uses quantized types (int8, uint8)
    return (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
            elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
  } catch (...) {
    return false;
  }
}

void Detector::setDevice(int deviceMask) {
  this->deviceMask = deviceMask;
  // Note: Changing device at runtime requires recreating the session
}