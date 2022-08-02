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
#define N_DIMS NUM_KERNEL_SIZE*NUM_KERNEL_SIZE

precision FLOAT_PRECISION float;
precision FLOAT_PRECISION sampler2DArray;

#if PLANE_COUNT > 0 // ends on line 25
layout(location = 0)out vec4 o_pixel;
#endif // end if from line 23
#if PLANE_COUNT > 1 // ends on line 28
layout(location = 1)out vec4 o_pixel1;
#endif // end if from line 26
#if PLANE_COUNT > 2 // ends on line 31
layout(location = 2)out vec4 o_pixel2;
#endif // end if from line 29
#if PLANE_COUNT > 3 // ends on line 34
layout(location = 3)out vec4 o_pixel3;
#endif // end if from line 34

//uniform FLOAT_PRECISION vec4 bias;
uniform FLOAT_PRECISION uint kernelSize;
uniform FLOAT_PRECISION sampler2DArray inputTextures;

#ifdef USE_UNIFORM_WEIGHTS
#ifdef USE_WEIGHT_TEXTURES
layout (binding = 2) uniform sampler2D weightMatrix1;
#if PLANE_COUNT > 1
layout (binding = 3) uniform sampler2D weightMatrix2;
#endif
#if PLANE_COUNT > 2
layout (binding = 4) uniform sampler2D weightMatrix3;
#endif
#if PLANE_COUNT > 3
layout (binding = 5) uniform sampler2D weightMatrix4;
#endif
#endif
#ifdef USE_WEIGHT_BUFFERS
layout (STORAGE_FORMAT, binding = 2) VARIABLE_SPECIFIER weightMatrix1;
#if PLANE_COUNT > 1
layout (STORAGE_FORMAT, binding = 3) VARIABLE_SPECIFIER weightMatrix2;
#endif
#if PLANE_COUNT > 2
layout (STORAGE_FORMAT, binding = 4) VARIABLE_SPECIFIER weightMatrix3;
#endif
#if PLANE_COUNT > 3
layout (STORAGE_FORMAT, binding = 5) VARIABLE_SPECIFIER weightMatrix4;
#endif
#endif
uniform vec4 bias[PLANE_COUNT];
#ifdef USE_BATCH_NORMALIZATION
uniform vec4 beta[PLANE_COUNT];
uniform vec4 gamma[PLANE_COUNT];
uniform vec4 movingMean[PLANE_COUNT];
uniform vec4 movingVariance[PLANE_COUNT];
#endif
#endif

#ifdef USE_WEIGHT_CONSTANTS
#if PLANE_COUNT > 0
const FLOAT_PRECISION vec4 weightMatrix1[] = vec4[](
	_PLACEHOLDER_WEIGHT1_VEC_CONSTANTS_);
#endif
#if PLANE_COUNT > 1
const FLOAT_PRECISION vec4 weightMatrix2[] = vec4[](
	_PLACEHOLDER_WEIGHT2_VEC_CONSTANTS_);
#endif
#if PLANE_COUNT > 2
const FLOAT_PRECISION vec4 weightMatrix3[] = vec4[](
	_PLACEHOLDER_WEIGHT3_VEC_CONSTANTS_);
#endif
#if PLANE_COUNT > 3
const FLOAT_PRECISION vec4 weightMatrix4[] = vec4[](
	_PLACEHOLDER_WEIGHT4_VEC_CONSTANTS_);
#endif


_PLACEHOLDER_BIAS_CONSTANTS_

#ifdef USE_BATCH_NORMALIZATION
const FLOAT_PRECISION vec4 beta[] = vec4[](_PLACEHOLDER_BETA_VEC_CONSTANTS_);
const FLOAT_PRECISION vec4 gamma[] = vec4[](_PLACEHOLDER_GAMMA_VEC_CONSTANTS_);
const FLOAT_PRECISION vec4 movingMean[] = vec4[](_PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_);
const FLOAT_PRECISION vec4 movingVariance[] = vec4[](_PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_);
#endif

#endif

bool checkValid (FLOAT_PRECISION vec2 coords) {
	bool retVal = true;
    retVal = retVal && !((coords.x >= 1.0f || coords.x < 0.0f));
    retVal = retVal && !((coords.y >= 1.0f || coords.y < 0.0f));
	return retVal;
}

FLOAT_PRECISION vec2 replicatePadding(FLOAT_PRECISION vec2 sourceCoords) {
    FLOAT_PRECISION vec2 repCoords = vec2(0.0, 0.0);
    FLOAT_PRECISION vec2 offsetCoords = vec2(0.5, 0.5);
    repCoords.x = (sourceCoords.x >= 1.0f) ? 2.0 - sourceCoords.x - offsetCoords.x : 0.0 - sourceCoords.x + offsetCoords.x;
    repCoords.y = (sourceCoords.y >= 1.0f) ? 2.0 - sourceCoords.y - offsetCoords.y : 0.0 - sourceCoords.y + offsetCoords.y;
    return repCoords;
}

FLOAT_PRECISION vec2 checkboardPadding(FLOAT_PRECISION vec2 sourceCoords) {
    FLOAT_PRECISION vec2 repCoords = vec2(0.0, 0.0);
    FLOAT_PRECISION vec2 offsetCoords = vec2(0.5, 0.5);
    repCoords.x = (sourceCoords.x >= 1.0f) ? sourceCoords.x - 1.0f + offsetCoords.x : sourceCoords.x + 1.0f + offsetCoords.x;
    repCoords.y = (sourceCoords.y >= 1.0f) ? sourceCoords.y - 1.0f + offsetCoords.y : sourceCoords.y + 1.0f + offsetCoords.y;
    return repCoords;
}

void main()
{
	// Convolution Process
	const int lod = 0;
	FLOAT_PRECISION vec2 maxUV = vec2(INPUT_WIDTH, INPUT_HEIGHT);//textureSize(inputTextures[0], lod);
	FLOAT_PRECISION vec4 s = vec4(0.0f);
	FLOAT_PRECISION vec4 s1 = vec4(0.0f);
	FLOAT_PRECISION vec4 s2 = vec4(0.0f);
	FLOAT_PRECISION vec4 s3 = vec4(0.0f);

	ivec2 baseCoord_old = ivec2(gl_FragCoord.xy);
	ivec2 baseCoord = ivec2(int(baseCoord_old.x * NUM_STRIDE), int(baseCoord_old.y * NUM_STRIDE));
	baseCoord += ivec2(1, 1);

	// Pre-calculate coordinates here, note that
	// textureGather: pixel should be in the center of the four surrounding pixels, use zero offset
	// texture: Pixel should be in the center of the pixel since GL_NEAREST, use 0.5 offset
	// It's also verified that using texture varying and have vertex shader generate those has same performance.

#ifdef USE_UNIFORM_WEIGHTS
    vec2 baseCoordFloat = vec2(baseCoord);

    FLOAT_PRECISION vec2 texCoords[N_DIMS];
    FLOAT_PRECISION vec4 texVals[PLANE_COUNT * N_DIMS];
    FLOAT_PRECISION vec2 weightsCoords[N_DIMS];
    FLOAT_PRECISION float offsetsT[NUM_KERNEL_SIZE];
    FLOAT_PRECISION float offsetsB[NUM_KERNEL_SIZE];

    float initValT = -0.5 - float(PADDING_T);
    float initValL = -0.5 - float(PADDING_L);

    for (int i = 0; i < NUM_KERNEL_SIZE; i++) {
        offsetsT[i] = initValT;
        initValT += 1.0f;
    }

    for (int i = 0; i < NUM_KERNEL_SIZE; i++) {
        offsetsB[i] = initValL;
        initValL += 1.0f;
    }

	for (int i = 0; i < NUM_KERNEL_SIZE; i++) {
		for (int j = 0; j < NUM_KERNEL_SIZE; j++) {
			texCoords[i*NUM_KERNEL_SIZE + j] = (baseCoordFloat + vec2(offsetsT[i], offsetsB[j]) ) / maxUV;
			weightsCoords[i*NUM_KERNEL_SIZE + j] = (vec2(i, j) + vec2(0.5, 0.5)) / vec2(NUM_KERNEL_SIZE, NUM_KERNEL_SIZE);
		}
	}

	int i = OUTPUTPLANE_INDEX;
	int layer = i >> 2;
	for (int layerIdx = layer; layerIdx < layer + PLANE_COUNT; layerIdx++) {
		for (int i = 0; i < NUM_KERNEL_SIZE; i++) {
			for (int j = 0; j < NUM_KERNEL_SIZE; j++) {
				int arrayAccess = N_DIMS * (layerIdx - layer) + i * NUM_KERNEL_SIZE + j;
#ifdef CLAMPED_PADDING
				texVals[arrayAccess] = texture(inputTextures, vec3(texCoords[arrayAccess], layerIdx));
#endif
#ifdef CONST_PADDING
				texVals[arrayAccess] = (checkValid(texCoords[arrayAccess])) ? texture(inputTextures, vec3(texCoords[arrayAccess], layerIdx)) : vec4(PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE);
#endif
#ifdef REPLICATE_PADDING
				FLOAT_PRECISION vec2 repCoords = replicatePadding(texCoords[arrayAccess]);
                texVals[arrayAccess] = (checkValid(texCoords[arrayAccess])) ? texture(inputTextures, vec3(texCoords[arrayAccess], layerIdx)) : texture(inputTextures, vec3(repCoords, layerIdx));
#endif
#ifdef CHECKBOARD_PADDING
				FLOAT_PRECISION vec2 repCoords = checkboardPadding(texCoords[arrayAccess]);
                texVals[arrayAccess] = (checkValid(texCoords[arrayAccess])) ? texture(inputTextures, vec3(texCoords[arrayAccess], layer)) : texture(inputTextures, vec3(repCoords, layer));
#endif
			}
		}
	}

#if PLANE_COUNT > 0
	for (int i = 0; i < N_DIMS; i++) {
		s += texVals[i] * texture(weightMatrix1, weightsCoords[i]);
	}
#endif
#if PLANE_COUNT > 1
	for (int i = N_DIMS; i < 2*N_DIMS; i++) {
		s1 += texVals[i] * texture(weightMatrix2, weightsCoords[i - N_DIMS]);
	}
#endif
#if PLANE_COUNT > 2
	for (int i = 2*N_DIMS; i < 3*N_DIMS; i++) {
		s2 += texVals[i] * texture(weightMatrix3, weightsCoords[i - 2*N_DIMS]);
	}
#endif
#if PLANE_COUNT > 3
	for (int i = 3*N_DIMS; i < 4*N_DIMS; i++) {
		s3 += texVals[i] * texture(weightMatrix4, weightsCoords[i - 3*N_DIMS]);
	}
#endif
	
#else
	
		
_PLACEHOLDER_ELEMENT_ACCESS_

_PLACEHOLDER_CALC_

#endif
#if PLANE_COUNT > 0
    s += bias[0];
#endif
#if PLANE_COUNT > 1
    s1 += bias[1];
#endif
#if PLANE_COUNT > 2
    s2 += bias[2];
#endif
#if PLANE_COUNT > 3
    s3 += bias[3];
#endif

#ifdef USE_BATCH_NORMALIZATION
#if PLANE_COUNT > 0
    FLOAT_PRECISION vec4 sqrtVar = sqrt(movingVariance[0] + vec4(0.001f));
    sqrtVar = max(sqrtVar, vec4(0.0001f));
    s = ((gamma[0]/sqrtVar) * (s - movingMean[0])) + beta[0];
#endif
#if PLANE_COUNT > 1
    sqrtVar = sqrt(movingVariance[1] + vec4(0.001f));
    sqrtVar = max(sqrtVar, vec4(0.0001f));
    s1 = ((gamma[1]/sqrtVar) * (s1 - movingMean[1])) + beta[1];
#endif
#if PLANE_COUNT > 2
    sqrtVar = sqrt(movingVariance[2] + vec4(0.001f));
    sqrtVar = max(sqrtVar, vec4(0.0001f));
    s2 = ((gamma[2]/sqrtVar) * (s2 - movingMean[2])) + beta[2];
#endif
#if PLANE_COUNT > 3
    sqrtVar = sqrt(movingVariance[3] + vec4(0.001f));
    sqrtVar = max(sqrtVar, vec4(0.0001f));
    s3 = ((gamma[3]/sqrtVar) * (s - movingMean[3])) + beta[3];
#endif
#endif
	//o_pixel = s;
//}

