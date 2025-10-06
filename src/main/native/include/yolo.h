#pragma once

#include <onnxruntime_cxx_api.h>

#include <type_traits>
#include <vector>

struct DetectionBox {
  float x, y, w, h;
  float confidence;
  int classId;
};

template <typename T>
inline float toFloat(const T& value) {
  if constexpr (std::is_same_v<T, Ort::Float16_t>) {
    return static_cast<float>(value);
  } else {
    return value;
  }
}

template <typename T>
std::vector<DetectionBox> parseYOLOOutput(T* outputData, int numBoxes,
                                          int boxDim, int origW, int origH,
                                          int inputW, int inputH,
                                          int numClasses, float boxThresh,
                                          bool isV8Format) {
  std::vector<DetectionBox> boxes;
  boxes.reserve(numBoxes / 10);  // Reserve approximate capacity

  const float scaleX = static_cast<float>(origW) / inputW;
  const float scaleY = static_cast<float>(origH) / inputH;

  if (isV8Format) {
    // YOLOv8: [1, 4+num_classes, num_anchors]
    for (int i = 0; i < numBoxes; ++i) {
      // Find best class
      int bestClass = 0;
      float bestScore = 0.0f;

      for (int c = 0; c < numClasses; ++c) {
        const float score = toFloat(outputData[(4 + c) * numBoxes + i]);
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < boxThresh) continue;

      boxes.emplace_back(
          DetectionBox{.x = toFloat(outputData[0 * numBoxes + i]) * scaleX,
                       .y = toFloat(outputData[1 * numBoxes + i]) * scaleY,
                       .w = toFloat(outputData[2 * numBoxes + i]) * scaleX,
                       .h = toFloat(outputData[3 * numBoxes + i]) * scaleY,
                       .confidence = bestScore,
                       .classId = bestClass});
    }
  } else {
    // YOLOv5: [1, num_anchors, 5+num_classes]
    for (int i = 0; i < numBoxes; ++i) {
      const T* box = outputData + i * boxDim;
      const float objectness = toFloat(box[4]);

      if (objectness < boxThresh) continue;

      // Find best class
      int bestClass = 0;
      float bestScore = 0.0f;

      for (int c = 0; c < numClasses; ++c) {
        const float score = toFloat(box[5 + c]) * objectness;
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < boxThresh) continue;

      boxes.emplace_back(DetectionBox{.x = toFloat(box[0]) * scaleX,
                                      .y = toFloat(box[1]) * scaleY,
                                      .w = toFloat(box[2]) * scaleX,
                                      .h = toFloat(box[3]) * scaleY,
                                      .confidence = bestScore,
                                      .classId = bestClass});
    }
  }

  return boxes;
}

std::vector<DetectionBox> parseYOLOv5Output(Ort::Value& output, int origW,
                                            int origH, int inputW, int inputH,
                                            int numClasses, float boxThresh);

std::vector<DetectionBox> parseYOLOv8Output(Ort::Value& output, int origW,
                                            int origH, int inputW, int inputH,
                                            int numClasses, float boxThresh);