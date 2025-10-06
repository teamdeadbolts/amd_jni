#include "yolo.h"

std::vector<DetectionBox> parseYOLOv5Output(Ort::Value& output, int origW,
                                            int origH, int inputW, int inputH,
                                            int numClasses, float boxThresh) {
  const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
  const auto outputType = output.GetTensorTypeAndShapeInfo().GetElementType();

  const int numBoxes = static_cast<int>(shape[1]);
  const int boxDim = static_cast<int>(shape[2]);

  if (outputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    auto* data = output.GetTensorMutableData<Ort::Float16_t>();
    return parseYOLOOutput(data, numBoxes, boxDim, origW, origH, inputW, inputH,
                           numClasses, boxThresh, false);
  } else {
    auto* data = output.GetTensorMutableData<float>();
    return parseYOLOOutput(data, numBoxes, boxDim, origW, origH, inputW, inputH,
                           numClasses, boxThresh, false);
  }
}

std::vector<DetectionBox> parseYOLOv8Output(Ort::Value& output, int origW,
                                            int origH, int inputW, int inputH,
                                            int numClasses, float boxThresh) {
  const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
  const auto outputType = output.GetTensorTypeAndShapeInfo().GetElementType();

  const int boxDim = static_cast<int>(shape[1]);
  const int numBoxes = static_cast<int>(shape[2]);

  if (outputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    auto* data = output.GetTensorMutableData<Ort::Float16_t>();
    return parseYOLOOutput(data, numBoxes, boxDim, origW, origH, inputW, inputH,
                           numClasses, boxThresh, true);
  } else {
    auto* data = output.GetTensorMutableData<float>();
    return parseYOLOOutput(data, numBoxes, boxDim, origW, origH, inputW, inputH,
                           numClasses, boxThresh, true);
  }
}