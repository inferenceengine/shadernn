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
#include "snn/imageTexture.h"

using namespace snn;

void snn::readTexture(int buf_size, GLuint textureId) {
    char* outBuffer = (char*) malloc(buf_size * 16);
    glActiveTexture(GL_TEXTURE0);
    CHECK_GL_ERROR("glActiveTexture");
    glBindTexture(GL_TEXTURE_2D_ARRAY, textureId);
    CHECK_GL_ERROR("glBindTexture");

    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_FLOAT, outBuffer);
    CHECK_GL_ERROR("glGetTexImage");

    float* dest = (float*) outBuffer;
    printf(">>>>>>>>>>readTexture>>>>>>>>>>>>>\n");
    for (int i = 0; i < buf_size * 4; i += 4) {
        printf("%d, %f\n", i, *(dest + i));
    }
    printf("<<<<<<<<<<<<<<<<<<<<<<<\n");

    free(outBuffer);
}

std::vector<float> snn::readTexture(int texDimSize, GLuint textureId, uint32_t w, uint32_t h, uint32_t d, uint32_t p) {
    std::vector<float> ret;
    ret.resize(texDimSize * 4);
    glActiveTexture(GL_TEXTURE0);
    CHECK_GL_ERROR("glActiveTexture");
    if (d <= 1) {
        glBindTexture(GL_TEXTURE_2D, textureId);
    } else {
        glBindTexture(GL_TEXTURE_2D_ARRAY, textureId);
    }
    CHECK_GL_ERROR("glBindTexture");

    float* outBuffer = (float*) ret.data();
    if (d <= 1) {
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, outBuffer);
        CHECK_GL_ERROR("glGetTexImage");
    } else {
        glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_FLOAT, outBuffer);
        CHECK_GL_ERROR("glGetTexImage");
    }

    float* dest = (float*) outBuffer;
    printf(">>>>>>>>>>readTexture>>>>>>>>>>>>>\n");
    for (int i = 0; i < texDimSize * 4; i += 1) {
        printf("%d, %f\n", i, *(dest + i));
    }
    printf("<<<<<<<<<<<<<<<<<<<<<<<\n");

    std::vector<uint32_t> dims {w, h, d, p};
    snn::ImageTexture img(dims, ColorFormat::RGBA32F, dest);
    img.printOutWH();

    return ret;
}
