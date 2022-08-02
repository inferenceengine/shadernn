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
#include "layer/convolution.h"
#include "layer/padding.h"
#include "layer/pooling.h"
#include "layer/interp.h"

#include "cpu.h"
#include "net.h"

#include "layer/yolov3detectionoutput.h"
#include "testutil.h"
#include "matutil.h"

#define NCNN_MODEL_NAME "yolov3-tiny_dummy"
#define SNN_MODEL_NAME  "yolov3-tiny_dummy.json"
// #define SNN_MODEL_NAME "yolov3_tiny_bb_layers.json"
// #define TEST_IMAGE "%s/images/rose_1_416x416.png"
#define TEST_IMAGE "%s/assets/images/arduino.png"

// #define FRAGMENT_SHADER 1
#define COMPARE_THRESHOLD 0.01

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

static ncnn::Mat test_snn_input_v3tiny(ncnn::Mat mat13, ncnn::Mat mat26) {
    const float b[] = {10.000000, 14.000000, 23.000000, 27.000000, 37.000000, 58.000000, 81.000000, 82.000000, 135.000000, 169.000000, 344.000000, 319.000000};
    const float m[] = {3.000000, 4.000000, 5.000000, 1.000000, 2.000000, 3.000000};
    const float s[] = {32.000000, 16.000000};

    ncnn::Mat biases        = create_mat_from(b, sizeof(b) / sizeof(b[0]));
    ncnn::Mat mask          = create_mat_from(m, sizeof(m) / sizeof(m[0]));
    ncnn::Mat anchors_scale = create_mat_from(s, sizeof(s) / sizeof(s[0]));

    std::vector<ncnn::Mat> a(2);
    // a[0] = MyRandomMat(13, 13, 255);
    // a[1] = MyRandomMat(26, 26, 255);
    a[0] = mat13;
    a[1] = mat26;

    ncnn::ParamDict pd;
    pd.set(0, 1); // Class
    pd.set(1, 3); // Num of box
    // pd.set(2, 0.6f);
    // pd.set(3, 0.45f);
    pd.set(2, 0.35f);
    pd.set(3, 0.45f);
    pd.set(4, biases);
    pd.set(5, mask);
    pd.set(6, anchors_scale);
    pd.set(7, 2);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ncnnOutput(1);
    {
        int ret = test_layer_naive(ncnn::layer_to_index("Yolov3DetectionOutput"), pd, weights, a, 1, ncnnOutput, (void (*)(ncnn::Yolov3DetectionOutput*)) 0, 0);
        if (ret != 0) {
            fprintf(stderr, "test_layer_naive failed\n");
        }
    }
    pretty_print_ncnn(ncnnOutput[0]);

    return ncnnOutput[0];
}

int main(int argc, char** argv) {
    SRAND(7767517);

    ncnn::Mat ncnnMat, ncnnMat2, snnMat;
    int ret = 0;

    snn::MRTMode mrtMode = snn::MRTMode::SINGLE_PLANE;

    if (argc > 1) {
        if (strcmp("--use_2ch_mrt", argv[1]) == 0) {
            mrtMode = snn::MRTMode::DOUBLE_PLANE;
            printf("MRT_MODE SET TO: DOUBLE_PLANE\nSHADER MODE: FRAGMENT\n");
        }

        if (strcmp("--use_4ch_mrt", argv[1]) == 0) {
            mrtMode = snn::MRTMode::QUAD_PLANE;
            printf("MRT_MODE SET TO: QUAD_PLANE\nSHADER MODE: FRAGMENT\n");
        }

        if (strcmp("--use_compute", argv[1]) == 0) {
            mrtMode = (snn::MRTMode) 0;
            printf("MRT_MODE SET TO: NULL\nSHADER MODE: COMPUTE\n");
        }
    } else {
        printf("DEFAULT MRT MODE SET TO: snn::MRTMode::SINGLE_PLANE\n");
        printf("DEFAULT SHADER MODE: FRAGMENT SHADER\n");
        printf("To change this mode, use the following options. Note that they are mutually exclusive:\n");
        printf("\t--use_2ch_mrt : Use 2 render targets (DOUBLE_PLANE MRT)\n");
        printf("\t--use_4ch_mrt : Use 4 render targets (QUAD_PLANE MRT)\n");
        printf("\t--use_compute : Use compute shader\n");
    }

    /*ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE,
    ASSETS_DIR).c_str(), "input_4_blob", 416); snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [01] Conv2D_3x3 pass[3]_input.dump",
    DUMP_DIR,SNN_MODEL_NAME).c_str(), true); ret = CompareMat(ncnnMat, snnMat); if (ret) {
        //pretty_print_ncnn(ncnnMat);
        //pretty_print_ncnn(snnMat);
    }
    printf("---------------------------------Conv_layer_1 layer input res: %d\n", ret);*/
    ////pretty_print_ncnn(snnMat);

    // compareNCNNLayerSNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), 1,
    //     formatString("%s/Yolov3-tiny/%s layer [01] Conv2D_7x7 pass[15]_input.dump", DUMP_DIR,SNN_MODEL_NAME).c_str(),
    //     3, 64, 32, 7, 0, 2, true);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_12_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [01] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(3, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        pretty_print_ncnn(ncnnMat);
        pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Conv_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "max_pooling2d_7_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [02] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(3, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Pooling_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_13_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [03] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block batch_norm_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "max_pooling2d_8_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [04] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(7, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block Add_layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_14_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [05] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "max_pooling2d_9_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [06] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(15, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_3 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_15_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [07] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "max_pooling2d_10_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [08] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_4 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_16_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [09] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_5 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "max_pooling2d_11_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [10] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Conv2D_layer_6 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_17_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [11] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------1st Block Add_layer_2 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "max_pooling2d_12_blob", 416);
    snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [12] MaxPooling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret    = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------activation 12 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_18_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [13] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(255, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------activation 13 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_19_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [14] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------activation 14 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_21_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [15] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(31, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------activation 15 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "up_sampling2d_2_blob", 416);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [16] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [16] UpSampling2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 31).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Final Upsample layer output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "concatenate_2_blob", 416);
    if (mrtMode == (snn::MRTMode) 0) {
        snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [17] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 0).c_str());
    } else {
        snnMat = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [17] Concatenate pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, 95).c_str());
    }
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Final concate layer output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_20_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [18] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(127, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Final 18th conv2d output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "leaky_re_lu_22_blob", 416);
    snnMat  = getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [19] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(63, mrtMode)).c_str());
    ret     = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (ret) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Final 19th conv2d output res: %d\n", ret);

    // ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE,
    // ASSETS_DIR).c_str(), "flatten_blob", 416); snnMat = getSNNLayerText(formatString("%s/Yolov3-tiny/%s layer [31] Flatten cpu layer.txt",
    // DUMP_DIR,SNN_MODEL_NAME).c_str()); ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD); if (ret) {
    //     //pretty_print_ncnn(ncnnMat);
    //     //pretty_print_ncnn(snnMat);
    // }
    // printf("-----------------------------Flatten_Layer_1 output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "conv2d_23_blob", 416);
    snnMat =
        getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [20] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (1) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Output 20th conv2d output res: %d\n", ret);

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "conv2d_26_blob", 416);
    snnMat =
        getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [21] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    ret = CompareMat(ncnnMat, snnMat, COMPARE_THRESHOLD);
    if (1) {
        // pretty_print_ncnn(ncnnMat);
        // pretty_print_ncnn(snnMat);
    }
    printf("-----------------------------Output 21st conv2d output res: %d\n", ret);

    auto ncnnMat13 = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                                  formatString(TEST_IMAGE, ASSETS_DIR).c_str(), "conv2d_23_blob", 416);
    auto ncnnMat26 = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(),
                                  formatString(TEST_IMAGE, ASSETS_DIR).c_str(), "conv2d_26_blob", 416);
    auto snnMat13 =
        getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [20] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    auto snnMat26 =
        getSNNLayer(formatString("%s/Yolov3-tiny/%s layer [21] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    // test_snn_input_v3tiny(ncnnMat13, ncnnMat26);
    auto layerMat = test_snn_input_v3tiny(ncnnMat13, ncnnMat26);

    // std::vector<ncnn::Mat> mats;
    // mats.push_back(snnMat13);
    // mats.push_back(snnMat26);
    // auto vectors = ncnnMat2Vectors(mats);

    // std::array<struct anchor_t, 9> anchors{{{10, 14}, {23, 27}, {37, 58}, {81, 82}, {135, 169}, {344, 319}}};
    // struct shape_t image_shape {416, 416};
    // yolo_output_t yoloOutput;
    // yoloOutput.push_back(vectors);
    // auto snnRes = yolo_eval(yoloOutput,
    // anchors,
    // 1,
    // image_shape,
    // 30,
    // 0.35,
    // 0.45
    // );

    // printf("Res of SNN YOLO layer:%zu\n", snnRes.size());
    // for (size_t i = 0; i < snnRes.size(); i++) {
    //     auto pred = snnRes[i];
    //     printf("%zu: %d, %f, %f, %f, %f, %f\n", i, pred.class_label, pred.score, pred.xmin, pred.ymin, pred.xmax, pred.ymax);
    // }

    ncnnMat = getNCNNLayer(formatString("%s/../../modelzoo/Yolov3-tiny/%s", ASSETS_DIR, NCNN_MODEL_NAME).c_str(), formatString(TEST_IMAGE, ASSETS_DIR).c_str(),
                           "detection_out", 416);
    // pretty_print_ncnn(ncnnMat);

    // auto ncnnTest = snnMat26.reshape(snnMat26.w * snnMat26.h * snnMat26.c);
    // pretty_print_ncnn(ncnnTest);

    std::vector<cv::Mat> cvMats;
    auto cvMat13 = getCVMatFromDump(
        formatString("%s/Yolov3-tiny/%s layer [20] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    auto cvMat26 = getCVMatFromDump(
        formatString("%s/Yolov3-tiny/%s layer [21] Conv2D pass[%d].dump", DUMP_DIR, SNN_MODEL_NAME, getPassIndex(4, mrtMode)).c_str(), false, 18);
    // auto cvMat13 = NCNNMat2CVMat(ncnnMat13);
    // auto cvMat26 = NCNNMat2CVMat(ncnnMat26);

    // print_3d_cvmat(cvMat13);
    // print_3d_cvmat(cvMat26);
    cvMats.push_back(cvMat13);
    cvMats.push_back(cvMat26);
    getBoundBoxFromYOLO(cvMats, 416, 416, 416, 416, 0.35f, 0.45f);

    return 0;
}
