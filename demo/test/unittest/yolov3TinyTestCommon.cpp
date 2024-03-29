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

static ncnn::Mat create_mat_from(const float* src, int length) {
    ncnn::Mat ret(length);
    memcpy(ret.data, src, length * sizeof(float));
    return ret;
}

static ncnn::Mat MyRandomMat(int w, int h, int c) {
    ncnn::Mat m(w, h, c);
    Randomize(m, -15.f, 1.5f);
    return m;
}

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

float CalculateIoU(const BoundingBox& obj0, const BoundingBox& obj1) {
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

void Nms(std::vector<BoundingBox>& bboxList, std::vector<BoundingBox>& bboxNmsList, float nmsIouThreshold, bool checkClassId) {
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

inline float sigmoidStd(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float scaleSigmoidStd(float x, float s) { return s * sigmoidStd(x) - (s - 1.0f) * 0.5f; }

bool getBoundingBox(const float* data, float scaleX, float scaleY, int32_t gridWidth, int32_t gridHeight, std::vector<BoundingBox>& bboxList,
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
                    float fl = *(data + index + i);
                    if (fl > maxClsLogit) {
                        maxClsLogit = fl;
                        classId     = i - YOLONumberOfFixed;
                    }
                }

                int anchorIndex        = static_cast<int>(YOLOv3TinyMasks[gridC + yoloIdx * YOLOGridChannel]);
                const float biasWidth  = YOLOv3TinyAnchors[anchorIndex * 2];
                const float biasHeight = YOLOv3TinyAnchors[anchorIndex * 2 + 1];

                float maxClsProb = 1.f / ((1.f + exp(-data[index + 4]) * (1.f + exp(-maxClsLogit))));

                if (maxClsProb > boxConfidence) {
                    float cx = (gridX + sigmoidStd(data[index + 0])) / gridWidth;
                    float cy = (gridY + sigmoidStd(data[index + 1])) / gridHeight;

                    float fw = std::exp(data[index + 2]) * biasWidth / netWidth;
                    float fh = std::exp(data[index + 3]) * biasHeight / netHeight;

                    float fx = cx - fw / 2;
                    float fy = cy - fh / 2;

                    printf("Index %d:%d:%d, score: %f coord: %f, %f, %f, %f\n", gridY, gridX, gridC, maxClsProb, fx, fy, fx + fw, fy + fh);

                    bboxList.push_back(BoundingBox(classId, "", maxClsProb, fx, fy, fw, fh));
                }

                index += YOLOV3OutputNum;
            }
        }
    }
    return 0;
}

static std::vector<BoundingBox> getBoundBoxFromYOLO(std::vector<cv::Mat>& yolos, int32_t imgWidth, int32_t imgHeight, int32_t inputWidth, int32_t inputHeight,
                                                    float confidenceThreshold, float iouThreshold) {
    std::vector<BoundingBox> bboxList;
    int index = 0;
    for (const auto& gridScale : YOLOGridScale) {
        int32_t gridWidth  = inputWidth / gridScale;
        int32_t gridHeight = inputHeight / gridScale;
        float scaleX       = static_cast<float>(gridScale) * imgWidth / inputWidth; /* scale to original image */
        float scaleY       = static_cast<float>(gridScale) * imgHeight / inputHeight;
        float* output_data = (float*) yolos[index].data;
        printf("scaleX:%f, scaleY:%f, gridWidth: %d, gridHeight:%d\n", scaleX, scaleY, gridWidth, gridHeight);
        getBoundingBox(output_data, scaleX, scaleY, gridWidth, gridHeight, bboxList, confidenceThreshold, index);
        index += 1;
    }
    printf("Before NMS Res: %zu\n", bboxList.size());
    for (auto& box : bboxList) {
        printf("Bounding box: %d,  score: %f, coord: %f, %f, %f, %f\n", box.classId, box.score, box.fx, box.fy, box.fw, box.fh);
    }

    std::vector<BoundingBox> nmsResult;
    Nms(bboxList, nmsResult, iouThreshold, true);

    printf("After NMS Res: %zu\n", nmsResult.size());
    for (auto& box : nmsResult) {
        printf("Bounding box: %d,  score: %f, coord: %f, %f, %f, %f\n", box.classId, box.score, box.fx, box.fy, box.fw, box.fh);
    }
    return nmsResult;
}

static int test_snn_input_v3tiny(ncnn::Mat mat13, ncnn::Mat mat26, ncnn::Mat& matOut) {
    const float b[] = {10.000000, 14.000000, 23.000000, 27.000000, 37.000000, 58.000000, 81.000000, 82.000000, 135.000000, 169.000000, 344.000000, 319.000000};
    const float m[] = {3.000000, 4.000000, 5.000000, 1.000000, 2.000000, 3.000000};
    const float s[] = {32.000000, 16.000000};

    ncnn::Mat biases        = create_mat_from(b, sizeof(b) / sizeof(b[0]));
    ncnn::Mat mask          = create_mat_from(m, sizeof(m) / sizeof(m[0]));
    ncnn::Mat anchors_scale = create_mat_from(s, sizeof(s) / sizeof(s[0]));

    std::vector<ncnn::Mat> a(2);
    a[0] = mat13;
    a[1] = mat26;

    ncnn::ParamDict pd;
    pd.set(0, 1); // Class
    pd.set(1, 3); // Num of box
    pd.set(2, 0.35f);
    pd.set(3, 0.45f);
    pd.set(4, biases);
    pd.set(5, mask);
    pd.set(6, anchors_scale);
    pd.set(7, 2);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ncnnOutput(1);
    int ret = test_layer_naive(ncnn::layer_to_index("Yolov3DetectionOutput"), pd, weights, a, 1, ncnnOutput, (void (*)(ncnn::Yolov3DetectionOutput*)) 0, 0);
    if (ret != 0) {
        fprintf(stderr, "test_layer_naive failed\n");
        return ret;
    } else {
        matOut = ncnnOutput[0];
    }

    return 0;
}
