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
#define FLOAT_PRECISION mediump

precision FLOAT_PRECISION int;
precision FLOAT_PRECISION float;
precision FLOAT_PRECISION sampler2D;
precision FLOAT_PRECISION sampler2DArray;
precision FLOAT_PRECISION image2D;
precision FLOAT_PRECISION image2DArray;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
#if NUM_OUTPUT_PLANES > 4
layout(rgba32f, binding = 0) uniform writeonly image2DArray outTexture;
#else
layout(rgba32f, binding = 0) uniform writeonly image2D outTexture;
#endif

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

layout(std140, binding=0) readonly buffer WeightMatrixX1 { vec4 weightMatrix1[]; };
#ifdef USE_COMPONENT_G
layout(std140, binding=1) readonly buffer WeightMatrixX2 { vec4 weightMatrix2[]; };
#endif
#ifdef USE_COMPONENT_B
layout(std140, binding=2) readonly buffer WeightMatrixX3 { vec4 weightMatrix3[]; };
#endif
#ifdef USE_COMPONENT_A
layout(std140, binding=3) readonly buffer WeightMatrixX4 { vec4 weightMatrix4[]; };
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

const vec2 maxUV = vec2(INPUT_WIDTH, INPUT_HEIGHT);//textureSize(inputTextures[0], lod);
const int NUM_LAYERS = (NUM_INPUT_PLANES + 3) / 4;

void main()
{
// Convolution Process
vec4 s = bias;

// Pre-calculate coordinates here, note that
// textureGather: pixel should be in the center of the four surrounding pixels, use zero offset
// texture: Pixel should be in the center of the pixel since GL_NEAREST, use 0.5 offset
// It's also verified that using texture varying and have vertex shader generate those has same performance.
// for stride 1: offset = vec2(0.0f, 0.0f);
vec2 baseCoord = floor(vec2(gl_GlobalInvocationID.xy + uvec2(1, 1)) / 2.0);
vec2 texCoord_0 = (baseCoord + vec2(-0.5, -0.5)) / maxUV;
vec2 texCoord_1 = (baseCoord + vec2( 0.5, -0.5)) / maxUV;
vec2 texCoord_2 = (baseCoord + vec2(-0.5,  0.5)) / maxUV;
vec2 texCoord_3 = (baseCoord + vec2( 0.5,  0.5)) / maxUV;
ivec4 baseWeight = ivec4(0, 2, 8,10) + ivec4((int(gl_GlobalInvocationID.x) % 2) + 4 * (int(gl_GlobalInvocationID.y) % 2));

#if DEBUG_PRINT_ENABLED
DebugPrintEndL();
DebugPrintI(INPUT_WIDTH);
DebugPrintI(INPUT_HEIGHT);
DebugPrintI(NUM_INPUT_PLANES);
DebugPrintI(NUM_STRIDE);
DebugPrintVec2(gl_FragCoord.xy); DebugPrintVec2(baseCoord); DebugPrintEndL();
DebugPrintI(baseWeight.x); DebugPrintVec2(texCoord_0 * maxUV); DebugPrintVec4(weightMatrix1[baseWeight.x]); DebugPrintEndL();
DebugPrintI(baseWeight.y); DebugPrintVec2(texCoord_1 * maxUV); DebugPrintVec4(weightMatrix1[baseWeight.y]); DebugPrintEndL();
DebugPrintI(baseWeight.z); DebugPrintVec2(texCoord_2 * maxUV); DebugPrintVec4(weightMatrix1[baseWeight.z]); DebugPrintEndL();
#ifdef USE_COMPONENT_A
DebugPrintI(baseWeight.w); DebugPrintVec2(texCoord_3 * maxUV); DebugPrintVec4(weightMatrix1[baseWeight.w]); DebugPrintEndL();
#endif
#endif

for (int layer = 0; layer < NUM_LAYERS; ++layer) {
    vec4 t0  = TEXTURE(inputTextures, vec3(texCoord_0, layer));
    vec4 t1  = TEXTURE(inputTextures, vec3(texCoord_1, layer));
    vec4 t2  = TEXTURE(inputTextures, vec3(texCoord_2, layer));
    vec4 t3  = TEXTURE(inputTextures, vec3(texCoord_3, layer));
    ivec4 weights = baseWeight + ivec4(layer * 16);
    s.r += dot(t0, weightMatrix1[weights.x]) + dot(t1, weightMatrix1[weights.y]) + dot(t2, weightMatrix1[weights.z]) + dot(t3, weightMatrix1[weights.w]);
#ifdef USE_COMPONENT_G
    s.g += dot(t0, weightMatrix2[weights.x]) + dot(t1, weightMatrix2[weights.y]) + dot(t2, weightMatrix2[weights.z]) + dot(t3, weightMatrix2[weights.w]);
#endif
#ifdef USE_COMPONENT_B
    s.b += dot(t0, weightMatrix3[weights.x]) + dot(t1, weightMatrix3[weights.y]) + dot(t2, weightMatrix3[weights.z]) + dot(t3, weightMatrix3[weights.w]);
#endif
#ifdef USE_COMPONENT_A
    s.a += dot(t0, weightMatrix4[weights.x]) + dot(t1, weightMatrix4[weights.y]) + dot(t2, weightMatrix4[weights.z]) + dot(t3, weightMatrix4[weights.w]);
#endif
}

#ifdef USE_BATCH_NORMALIZATION
const vec4 beta = _PLACEHOLDER_BETA_VEC_CONSTANTS_;
const vec4 gamma = _PLACEHOLDER_GAMMA_VEC_CONSTANTS_;
const vec4 movingMean = _PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_;
const vec4 movingVariance = _PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_;
s = ((gamma / sqrt(movingVariance + vec4(0.001f))) * (s - movingMean)) + beta;
#endif


//}


