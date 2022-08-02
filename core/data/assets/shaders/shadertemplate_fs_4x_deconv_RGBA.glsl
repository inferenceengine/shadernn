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

precision highp int;

layout(location = 0)out vec4 o_pixel;

uniform vec4 bias;

#ifdef INPUT_TEXTURE_2D
uniform sampler2D inputTextures;
#define TEXTURE(t, p) texture((t), (p).xy)
#define TEXTURE_GATHER(t, p, c) textureGather((t), (p).xy, (c))
#define TEXEL_FETCH(t, p, l) texelFetch((t), (p).xy, (l))
#else
uniform sampler2DArray inputTextures;
#define TEXTURE(t, p) texture((t), (p))
#define TEXTURE_GATHER(t, p, c) textureGather((t), (p), (c))
#define TEXEL_FETCH(t, p, l) texelFetch((t), (p), (l))
#endif

//These should be defined inside the shader loader code
//#define INPUT_WIDTH 540.0
//#define INPUT_HEIGHT 960.0

//#define USE_COMPONENT_G
//#define USE_COMPONENT_B
//#define USE_COMPONENT_A

const vec4 weightMatrix1[] = vec4[](
_PLACEHOLDER_WEIGHT1_VEC_CONSTANTS_);

#ifdef USE_COMPONENT_G
const vec4 weightMatrix2[] = vec4[](
_PLACEHOLDER_WEIGHT2_VEC_CONSTANTS_);
#endif

#ifdef USE_COMPONENT_B
const vec4 weightMatrix3[] = vec4[](
_PLACEHOLDER_WEIGHT3_VEC_CONSTANTS_);
#endif

#ifdef USE_COMPONENT_A
const vec4 weightMatrix4[] = vec4[](
_PLACEHOLDER_WEIGHT4_VEC_CONSTANTS_);
#endif

#ifdef USE_BATCH_NORMALIZATION
const vec4 beta = _PLACEHOLDER_BETA_VEC_CONSTANTS_;
const vec4 gamma = _PLACEHOLDER_GAMMA_VEC_CONSTANTS_;
const vec4 movingMean = _PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_;
const vec4 movingVariance = _PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_;
#endif

// =============================================================================
// Debug Print (Begin)
// =============================================================================
// Change this to the pixel coordinate that you want to investigate
// Set to -1, -1 to disble the debug output.
#define DEBUG_PIXEL_COORDINATE_X -1
#define DEBUG_PIXEL_COORDINATE_Y -1
#define DEBUG_PRINT_ENABLED (DEBUG_PIXEL_COORDINATE_X >= 0) && (DEBUG_PIXEL_COORDINATE_Y >= 0)
#if DEBUG_PRINT_ENABLED
layout(std430, binding = 15) buffer DebugBuffer
{
    int   i_debug_counter;
    float i_debug_buffer[];
};

void DebugPrint(float value)
{
    if (ivec2(gl_FragCoord) == ivec2(DEBUG_PIXEL_COORDINATE_X, DEBUG_PIXEL_COORDINATE_Y))
    {
        int offset = atomicAdd(i_debug_counter, 1);
        if (offset < i_debug_buffer.length()) {
            i_debug_buffer[offset] = value;
        }
    }
}
#else
void DebugPrint(float value) {}
#endif

void DebugPrintEndL() {
    DebugPrint(intBitsToFloat(-1));
}

void DebugPrintB(bool value)
{
    DebugPrint(float(value));
}

void DebugPrintI(int value)
{
    DebugPrint(float(value));
}

void DebugPrintVec2(vec2 value)
{
    DebugPrint(value.x);
    DebugPrint(value.y);
}

void DebugPrintVec3(vec3 value)
{
    DebugPrint(value.x);
    DebugPrint(value.y);
    DebugPrint(value.z);
}

void DebugPrintVec4(vec4 value)
{
    DebugPrint(value.x);
    DebugPrint(value.y);
    DebugPrint(value.z);
    DebugPrint(value.w);
}

void DebugPrintMat4(mat4 value)
{
    for(int c = 0; c < 4; ++c) {
        DebugPrintVec4(value[c]);
    }
}
// =============================================================================
// Debug Print (End)
// =============================================================================

#define dotSum(wm, s) (dot(t0, wm[0 + s]) + \
    dot(t1, wm[1 + s]) + \
    dot(t2, wm[2 + s]) + \
    dot(t3, wm[3 + s]) + \
    dot(t4, wm[4 + s]) + \
    dot(t5, wm[5 + s]) + \
    dot(t6, wm[6 + s]) + \
    dot(t7, wm[7 + s]) + \
    dot(t8, wm[8 + s]) + \
    dot(t9, wm[9 + s]) + \
    dot(t10, wm[10 + s]) + \
    dot(t11, wm[11 + s]) + \
    dot(t12, wm[12 + s]) + \
    dot(t13, wm[13 + s]) + \
    dot(t14, wm[14 + s]) + \
    dot(t15, wm[15 + s]));


#define addToR(si) s.r += dotSum(weightMatrix1, si);

#ifdef USE_COMPONENT_G
#define addToG(si) s.g += dotSum(weightMatrix2, si);
#else
#define addToG(si)
#endif

#ifdef USE_COMPONENT_B
#define addToB(si) s.b += dotSum(weightMatrix3, si);
#else
#define addToB(si)
#endif

#ifdef USE_COMPONENT_A
#define addToA(si) s.a += dotSum(weightMatrix4, si);
#else
#define addToA(si)
#endif

#define RGBA_DOT_SUM(startingIndex) \
addToR(startingIndex) \
addToG(startingIndex) \
addToB(startingIndex) \
addToA(startingIndex)

highp const vec2 maxUV = vec2(INPUT_WIDTH, INPUT_HEIGHT);
highp const vec2 maxUV_stride = float(NUM_STRIDE) * vec2(maxUV);
highp const vec2 inv_maxUV_stride = 1.0f / maxUV_stride;
highp const vec2 boundary = maxUV + (maxUV - 1.0) * (float(NUM_STRIDE) - 1.0);
float checkTexel(highp vec2 coord) {
#if NUM_STRIDE == 1
    return 1.0;
#elif NUM_STRIDE == 2
    int x = int(coord.x) % NUM_STRIDE;
    int y = int(coord.y) % NUM_STRIDE;
    return float(x * y);
#else
    bool b = (int((coord.x - 0.5)) % NUM_STRIDE == 0) && (int((coord.y - 0.5)) % NUM_STRIDE == 0);
    return float(b);
#endif
}

vec4 getTexel(vec2 tcoord, int layer)
{
    return checkTexel(tcoord) * TEXTURE(inputTextures, vec3(tcoord * inv_maxUV_stride, layer));
}

void main()
{
// Convolution Process
vec4 s = vec4(0.0f);

// Pre-calculate coordinates here, note that
// textureGather: pixel should be in the center of the four surrounding pixels, use zero offset
// texture: Pixel should be in the center of the pixel since GL_NEAREST, use 0.5 offset
// It's also verified that using texture varying and have vertex shader generate those has same performance.
// for stride 1: offset = vec2(0.0f, 0.0f);
highp vec2 baseCoord = gl_FragCoord.xy;

highp vec2 texCoord_0 = (vec2(baseCoord) + vec2(-1.5, -1.5));
highp vec2 texCoord_1 = (vec2(baseCoord) + vec2(-0.5, -1.5));
highp vec2 texCoord_2 = (vec2(baseCoord) + vec2(0.5, -1.5));
highp vec2 texCoord_3 = (vec2(baseCoord) + vec2(1.5, -1.5));

highp vec2 texCoord_4 = (vec2(baseCoord) + vec2(-1.5, -0.5));
highp vec2 texCoord_5 = (vec2(baseCoord) + vec2(-0.5, -0.5));
highp vec2 texCoord_6 = (vec2(baseCoord) + vec2(0.5, -0.5));
highp vec2 texCoord_7 = (vec2(baseCoord) + vec2(1.5, -0.5));

highp vec2 texCoord_8 = (vec2(baseCoord) + vec2(-1.5, 0.5));
highp vec2 texCoord_9 = (vec2(baseCoord) + vec2(-0.5, 0.5));
highp vec2 texCoord_10 = (vec2(baseCoord) + vec2(0.5, 0.5));
highp vec2 texCoord_11 = (vec2(baseCoord) + vec2(1.5, 0.5));

highp vec2 texCoord_12 = (vec2(baseCoord) + vec2(-1.5, 1.5));
highp vec2 texCoord_13 = (vec2(baseCoord) + vec2(-0.5, 1.5));
highp vec2 texCoord_14 = (vec2(baseCoord) + vec2(0.5, 1.5));
highp vec2 texCoord_15 = (vec2(baseCoord) + vec2(1.5, 1.5));

#if DEBUG_PRINT_ENABLED
DebugPrintEndL();
DebugPrintI(INPUT_WIDTH);
DebugPrintI(INPUT_HEIGHT);
DebugPrintI(NUM_INPUT_PLANES);
DebugPrintI(NUM_STRIDE);
DebugPrintVec2(baseCoord);
DebugPrintEndL();
vec2 forprint[] = vec2[](
    texCoord_0,
    texCoord_1,
    texCoord_2,
    texCoord_3,
    texCoord_4,
    texCoord_5,
    texCoord_6,
    texCoord_7,
    texCoord_8,
    texCoord_9,
    texCoord_10,
    texCoord_11,
    texCoord_12,
    texCoord_13,
    texCoord_14,
    texCoord_15
);
for(int i = 0; i < 16; ++i) {
    if (checkTexel(forprint[i]) != 0.) {
        DebugPrintI(i);
        DebugPrintVec2(forprint[i] / float(NUM_STRIDE));
        DebugPrintVec4(weightMatrix1[i]);
        //DebugPrintVec4(weightMatrix2[i]);
        //DebugPrintVec4(weightMatrix3[i]);
    #ifdef USE_COMPONENT_A
        //DebugPrintVec4(weightMatrix4[i]);
    #endif
        DebugPrintEndL();
    }
}
#endif

for (int i = 0; i < NUM_INPUT_PLANES; i += 4) {
    int layer = i >> 2;

    vec4 t0 = getTexel(texCoord_0, layer);
    vec4 t1 = getTexel(texCoord_1, layer);
    vec4 t2 = getTexel(texCoord_2, layer);
    vec4 t3 = getTexel(texCoord_3, layer);
    vec4 t4 = getTexel(texCoord_4, layer);
    vec4 t5 = getTexel(texCoord_5, layer);
    vec4 t6 = getTexel(texCoord_6, layer);
    vec4 t7 = getTexel(texCoord_7, layer);
    vec4 t8 = getTexel(texCoord_8, layer);
    vec4 t9 = getTexel(texCoord_9, layer);
    vec4 t10 = getTexel(texCoord_10, layer);
    vec4 t11 = getTexel(texCoord_11, layer);
    vec4 t12 = getTexel(texCoord_12, layer);
    vec4 t13 = getTexel(texCoord_13, layer);
    vec4 t14 = getTexel(texCoord_14, layer);
    vec4 t15 = getTexel(texCoord_15, layer);

    if (i == 0) {
        RGBA_DOT_SUM(0);
    }

#if NUM_INPUT_PLANES > 4
    else if (i == 4) {
        RGBA_DOT_SUM(16);
    }
#endif

#if NUM_INPUT_PLANES > 8
    else if (i == 8) {
        RGBA_DOT_SUM(32);
    }
#endif

#if NUM_INPUT_PLANES > 12
    else if (i == 12) {
        RGBA_DOT_SUM(48);
    }
#endif

#if NUM_INPUT_PLANES > 16
    else if (i == 16) {
        RGBA_DOT_SUM(64);
    }
#endif

#if NUM_INPUT_PLANES > 20
    else if (i == 20) {
        RGBA_DOT_SUM(80);
    }
#endif

#if NUM_INPUT_PLANES > 24
    else if (i == 24) {
        RGBA_DOT_SUM(96);
    }
#endif

#if NUM_INPUT_PLANES > 28
    else if (i == 28) {
        RGBA_DOT_SUM(112);
    }
#endif

#if NUM_INPUT_PLANES > 32
    else if (i == 32) {
        RGBA_DOT_SUM(128);
    }
#endif

#if NUM_INPUT_PLANES > 36
    else if (i == 36) {
        RGBA_DOT_SUM(144);
    }
#endif

#if NUM_INPUT_PLANES > 40
    else if (i == 40) {
        RGBA_DOT_SUM(160);
    }
#endif

#if NUM_INPUT_PLANES > 44
    else if (i == 44) {
        RGBA_DOT_SUM(176);
    }
#endif

#if NUM_INPUT_PLANES > 48
    else if (i == 48) {
        RGBA_DOT_SUM(192);
    }
#endif

#if NUM_INPUT_PLANES > 52
    else if (i == 52) {
        RGBA_DOT_SUM(208);
    }
#endif

#if NUM_INPUT_PLANES > 56
    else if (i == 56) {
        RGBA_DOT_SUM(224);
    }
#endif

#if NUM_INPUT_PLANES > 60
    else {
        RGBA_DOT_SUM(240);
    }
#endif
}

// Leaky ReLU Process
s += bias;

#ifdef USE_BATCH_NORMALIZATION
s = ((gamma / sqrt(movingVariance + vec4(0.001f))) * (s - movingMean)) + beta;
#endif
//}


