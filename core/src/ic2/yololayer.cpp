/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pch.h"
#include "dp.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <ios>

using namespace std;
using namespace snn;
using namespace snn::dp;

// DECLARE_LAYER(YOLO);

// Parameters for YOLO V3 Tiny layer
static const int32_t YOLOGridScale[]   = {32, 16};
static const int32_t YOLOGridChannel   = 3;
static const int32_t YOLONumberOfClass = 1;
static const int32_t YOLONumberOfFixed = 5;                                     // x, y, w, h, bbox
static const int32_t YOLOV3OutputNum   = YOLONumberOfClass + YOLONumberOfFixed; // x, y, w, h, bbox confidence, [class confidence]
static const float YOLOv3TinyAnchors[] = {10.000000, 14.000000, 23.000000,  27.000000,  37.000000,  58.000000,
                                          81.000000, 82.000000, 135.000000, 169.000000, 344.000000, 319.000000};
static const float YOLOv3TinyMasks[]   = {3.000000, 4.000000, 5.000000, 1.000000, 2.000000, 3.000000};

class BoundingBox {
public:
    BoundingBox(): classId(0), label(""), score(0), fx(0), fy(0), fw(0), fh(0) {}

    BoundingBox(int32_t _class_id, std::string _label, float _score, float _x, float _y, float _w, float _h)
        : classId(_class_id), label(_label), score(_score), fx(_x), fy(_y), fw(_w), fh(_h) {}

    int32_t classId;
    std::string label;
    float score = 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    float fw = 0.0f;
    float fh = 0.0f;
};

static float CalculateIoU(const BoundingBox& obj0, const BoundingBox& obj1) {
    float interx0 = (std::max)(obj0.fx, obj1.fx);
    float intery0 = (std::max)(obj0.fy, obj1.fy);
    float interx1 = (std::min)(obj0.fx + obj0.fw, obj1.fx + obj1.fw);
    float intery1 = (std::min)(obj0.fy + obj0.fh, obj1.fy + obj1.fh);
    if (interx1 < interx0 || intery1 < intery0) {
        return 0;
    }

    float area0     = obj0.fw * obj0.fh;
    float area1     = obj1.fw * obj1.fh;
    float areaInter = (interx1 - interx0) * (intery1 - intery0);
    float areaSum   = area0 + area1 - areaInter;

    return static_cast<float>(areaInter) / areaSum;
}

static void Nms(std::vector<BoundingBox>& bboxList, std::vector<BoundingBox>& bboxNmsList, float nmsIouThreshold, bool checkClassId) {
    std::sort(bboxList.begin(), bboxList.end(), [](BoundingBox const& lhs, BoundingBox const& rhs) {
        if (lhs.score > rhs.score) {
            return true;
        }
        return false;
    });

    std::unique_ptr<bool[]> isMerged(new bool[bboxList.size()]);
    for (size_t i = 0; i < bboxList.size(); i++) {
        isMerged[i] = false;
    }
    for (size_t indexHighScore = 0; indexHighScore < bboxList.size(); indexHighScore++) {
        std::vector<BoundingBox> candidates;

        if (isMerged[indexHighScore]) {
            continue;
        }

        candidates.push_back(bboxList[indexHighScore]);
        for (size_t index_low_score = indexHighScore + 1; index_low_score < bboxList.size(); index_low_score++) {
            if (isMerged[index_low_score]) {
                continue;
            }

            if (checkClassId && bboxList[indexHighScore].classId != bboxList[index_low_score].classId) {
                continue;
            }

            if (CalculateIoU(bboxList[indexHighScore], bboxList[index_low_score]) > nmsIouThreshold) {
                candidates.push_back(bboxList[index_low_score]);
                isMerged[index_low_score] = true;
            }
        }

        bboxNmsList.push_back(candidates[0]);
    }
}

static inline float sigmoidStd(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float scaleSigmoidStd(float x, float s) { return s * sigmoidStd(x) - (s - 1.0f) * 0.5f; }

static bool getBoundingBox(const float* data, float scaleX, float scaleY, int32_t gridWidth, int32_t gridHeight, std::vector<BoundingBox>& bboxList,
                           float boxConfidence, int yoloIdx) {
    int32_t netWidth  = (int) (scaleX * gridWidth);
    int32_t netHeight = (int) (scaleY * gridHeight);
    int32_t index     = 0;

    for (int32_t gridY = 0; gridY < gridHeight; gridY++) {
        for (int32_t gridX = 0; gridX < gridWidth; gridX++) {
            for (int32_t gridC = 0; gridC < YOLOGridChannel; gridC++) {
                int classId;
                float maxClsLogit = -FLT_MAX;
                for (int i = YOLONumberOfFixed; i < YOLOV3OutputNum; ++i) {
                    float l_ = *(data + index + i);
                    if (l_ > maxClsLogit) {
                        maxClsLogit = l_;
                        classId     = i - YOLONumberOfFixed;
                    }
                }

                // printf("Index %d:%d:%d, original: %f, %f, %f, %f - %f, %f, index:%d\n", gridY, gridX, gridC,
                //     data[index + 0], data[index + 1], data[index + 2], data[index + 3], data[index + 4], data[index + 5], index);

                int anchorIndex        = static_cast<int>(YOLOv3TinyMasks[gridC + yoloIdx * YOLOGridChannel]);
                const float biasWidth  = YOLOv3TinyAnchors[anchorIndex * 2];
                const float biasHeight = YOLOv3TinyAnchors[anchorIndex * 2 + 1];

                float maxClsProb = 1.f / ((1.f + exp(-data[index + 4]) * (1.f + exp(-maxClsLogit))));

                // printf("Index %d:%d:%d, calc: %d, %f, %f, net: %d, %d\n", gridY, gridX, gridC,
                //     biases_index, bias_w, bias_h, net_w, net_h);

                if (maxClsProb > boxConfidence) {
                    // scaleXY in YOLO v4
                    // float cx = (gridX + scaleSigmoidStd(data[index + 0], scaleXY)) / grid_w;
                    // float cy = (gridY + scaleSigmoidStd(data[index + 1], scaleXY)) / grid_h;
                    float cx = (gridX + sigmoidStd(data[index + 0])) / gridWidth;
                    float cy = (gridY + sigmoidStd(data[index + 1])) / gridHeight;

                    // printf("Index %d:%d:%d, cx: %f-%f, cy: %f-%f\n", gridY, gridX, gridC, sigmoidGPU(data[index + 0]), cx, sigmoidGPU(data[index + 1]), cy);

                    float w_ = std::exp(data[index + 2]) * biasWidth / netWidth;
                    float h_ = std::exp(data[index + 3]) * biasHeight / netHeight;

                    float x_ = cx - w_ / 2;
                    float y_ = cy - h_ / 2;

                    printf("Index %d:%d:%d, score: %f coord: %f, %f, %f, %f\n", gridY, gridX, gridC, maxClsProb, x_, y_, x_ + w_, y_ + h_);

                    bboxList.push_back(BoundingBox(classId, "", maxClsProb, x_, y_, w_, h_));
                }

                index += YOLOV3OutputNum;
            }
            index += (ROUND_UP_DIV_4(YOLOGridChannel * YOLOV3OutputNum) - YOLOGridChannel * YOLOV3OutputNum); // Texture aligned to 4 chs, so
        }
    }
    return 0;
}

void snn::dp::YOLODesc::parse(ModelParser& parser, int layerId) {
    try {
        auto layerObj   = parser.getJsonObject("Layer_" + std::to_string(layerId));
        numOutputPlanes = static_cast<int>(layerObj["outputPlanes"].get<double_t>());
        numInputPlanes  = static_cast<int>(layerObj["inputPlanes"].get<double_t>());
    } catch (std::exception& e) {
        SNN_LOGE("ModelParser::getYOLOLayer : Issues parsing layer %d, %s", layerId, e.what());
        return;
    }
}

void snn::dp::YOLOLayer::computeImageTexture(FixedSizeArray<snn::ImageTexture>& inputTex, FixedSizeArray<snn::ImageTexture>& outputTex) {
    printf("%%%%%%%% %s:%d %zu\n", __FUNCTION__, __LINE__, inputTex.size());
    int32_t imgWidth          = 416;
    int32_t imgHeight         = 416;
    int32_t inputWidth        = 416;
    int32_t inputHeight       = 416;
    float confidenceThreshold = 0.35f;
    float iouThreshold        = 0.45f;
    std::vector<BoundingBox> bboxList;
    int index = 0;
    for (const auto& gridScale : YOLOGridScale) {
        int32_t gridWidth  = inputWidth / gridScale;
        int32_t gridHeight = inputHeight / gridScale;
        float scaleX       = static_cast<float>(gridScale) * imgWidth / inputWidth; /* scale to original image */
        float scaleY       = static_cast<float>(gridScale) * imgHeight / inputHeight;

        uint32_t texWidth  = inputTex[index].texture(0)->getDesc().width;
        uint32_t texHeight = inputTex[index].texture(0)->getDesc().height;
        uint32_t texDepth  = inputTex[index].texture(0)->getDesc().depth;
        printf("%%%%%%%% %s:%d %d dim: %d, %d, %d\n", __FUNCTION__, __LINE__, index, texWidth, texHeight, texDepth);

        int numPixels = texWidth * texHeight * texDepth * 4;
        uint8_t* ret  = (uint8_t*) malloc(numPixels * sizeof(float));
        // std::vector<uint8_t> ret(numPixels);
        inputTex[index].download();
        inputTex[index].getCVMatData(ret);
        printf("scaleX:%f, scaleY:%f, gridWidth: %d, gridHeight:%d\n", scaleX, scaleY, gridWidth, gridHeight);
        getBoundingBox((float*) ret, scaleX, scaleY, gridWidth, gridHeight, bboxList, confidenceThreshold, index);

        free(ret);
        index += 1;
    }
    printf("Before NMS Res: %zu\n", bboxList.size());
    for (auto& box : bboxList) {
        printf("Bounding box: %d,  score: %f, coord: %f, %f, %f, %f\n", box.classId, box.score, box.fx, box.fy, box.fw, box.fh);
    }

    std::vector<BoundingBox> nmsResult;
    Nms(bboxList, nmsResult, iouThreshold, true);

    vector<vector<float>> vec; //(nmsResult.size(), vector<float>(6));

    printf("After NMS Res: %zu\n", nmsResult.size());
    for (auto& box : nmsResult) {
        printf("Bounding box: %d,  score: %f, coord: %f, %f, %f, %f\n", box.classId, box.score, box.fx, box.fy, box.fw, box.fh);
        vector<float> v {(float) box.classId, box.score, box.fx, box.fy, box.fw, box.fh};
        vec.push_back(v);
    }

    outputTex[0].outputMat = vec;

    printf("%%%%%%%% %s:%d :%s\n", __FUNCTION__, __LINE__, name.c_str());
}
