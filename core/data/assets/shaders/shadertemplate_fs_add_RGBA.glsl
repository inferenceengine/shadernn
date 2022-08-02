/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*/
#define FLOAT_PRECISION _PLACEHOLDER_PRECISION_

precision FLOAT_PRECISION float;
precision FLOAT_PRECISION sampler2D;
precision FLOAT_PRECISION sampler2DArray;

#if PLANE_COUNT > 0
layout(location = 0) out vec4 o_pixel;
#endif
#if PLANE_COUNT > 1
layout(location = 1) out vec4 o_pixel1;
#endif
#if PLANE_COUNT > 2
layout(location = 2) out vec4 o_pixel2;
#endif
#if PLANE_COUNT > 3
layout(location = 3) out vec4 o_pixel3;
#endif

_PLACEHOLDER_UNIFORMS_DECLARATION_

void main() {
    int lod = 0;
    FLOAT_PRECISION vec4 s = vec4(0.0f);
    FLOAT_PRECISION vec4 s1 = vec4(0.0f);
    FLOAT_PRECISION vec4 s2 = vec4(0.0f);
    FLOAT_PRECISION vec4 s3 = vec4(0.0f);

    ivec2 outLoc = ivec2(gl_FragCoord.xy);

    int layer = _PLACEHOLDER_LAYER_;
#if PLANE_COUNT > 1
    int layer1 = layer + 1;
#endif
#if PLANE_COUNT > 2
    int layer2 = layer + 2;
#endif
#if PLANE_COUNT > 3
    int layer3 = layer + 3;
#endif

#ifdef INPUT_TEXTURE_2D
    vec4 in1 = texelFetch(inputTextures0, ivec2(outLoc), 0);
    vec4 in2 = texelFetch(inputTextures1, ivec2(outLoc), 0);
#else
    vec4 in1 = texelFetch(inputTextures0, ivec3(outLoc, layer), 0);
    vec4 in2 = texelFetch(inputTextures1, ivec3(outLoc, layer), 0);
#endif

    s = in1 + in2;
#if PLANE_COUNT > 1
#ifdef INPUT_TEXTURE_2D
    in1 = texelFetch(inputTextures0, ivec2(outLoc), 0);
    in2 = texelFetch(inputTextures1, ivec2(outLoc), 0);
#else
    in1 = texelFetch(inputTextures0, ivec3(outLoc, layer1), 0);
    in2 = texelFetch(inputTextures1, ivec3(outLoc, layer1), 0);
#endif
    s1 = in1 + in2;
#endif
#if PLANE_COUNT > 2
#ifdef INPUT_TEXTURE_2D
    in1 = texelFetch(inputTextures0, ivec2(outLoc), 0);
    in2 = texelFetch(inputTextures1, ivec2(outLoc), 0);
#else
    in1 = texelFetch(inputTextures0, ivec3(outLoc, layer2), 0);
    in2 = texelFetch(inputTextures1, ivec3(outLoc, layer2), 0);
#endif
    s2 = in1 + in2;
#endif
#if PLANE_COUNT > 2
#ifdef INPUT_TEXTURE_2D
    in1 = texelFetch(inputTextures0, ivec2(outLoc), 0);
    in2 = texelFetch(inputTextures1, ivec2(outLoc), 0);
#else
    in1 = texelFetch(inputTextures0, ivec3(outLoc, layer3), 0);
    in2 = texelFetch(inputTextures1, ivec3(outLoc, layer3), 0);
#endif
    s3 = in1 + in2;
#endif
    _PLACEHOLDER_ACTIVATION_
    o_pixel = s;
#if PLANE_COUNT > 1
    o_pixel1 = s1;
#endif
#if PLANE_COUNT > 2
    o_pixel2 = s2;
#endif
#if PLANE_COUNT > 3
    o_pixel3 = s3;
#endif
}
