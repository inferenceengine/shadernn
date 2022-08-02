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

_PLACEHOLDER_DEFINES_

precision FLOAT_PRECISION float;
precision FLOAT_PRECISION sampler2D;
precision FLOAT_PRECISION sampler2DArray;
precision FLOAT_PRECISION image2D;
precision FLOAT_PRECISION image2DArray;

layout(location = 0) out vec4 o_pixel;
#if PLANE_COUNT > 1
layout(location = 1) out vec4 o_pixel1;
#endif
#if PLANE_COUNT > 2
layout(location = 2) out vec4 o_pixel2;
#endif
#if PLANE_COUNT > 3
layout(location = 3) out vec4 o_pixel3;
#endif

#ifdef INPUT_TEXTURE_2D
uniform sampler2D inputTextures;
#define TEXTURE(t, c) texture((t), (c).xy)
#else
uniform sampler2DArray inputTextures;
#define TEXTURE(t, c) texture((t), (c))
#endif

_PLACEHOLDER_UNIFORMS_DECLARATION_

bool checkValid (FLOAT_PRECISION vec2 coords) {
	float upperX = 1.0f;
	float upperY = 1.0f;
	bool retVal = true;
	retVal = retVal && !((coords.x >= upperX || coords.x < 0.0f));
	retVal = retVal && !((coords.y >= upperY || coords.y < 0.0f));
	return retVal;
}

void main()
{
    int lod = 0; 
    FLOAT_PRECISION vec2 maxUV = vec2(INPUT_WIDTH, INPUT_HEIGHT);
#if PLANE_COUNT > 0
	FLOAT_PRECISION vec4 s = vec4(0.0f);
    float maxR_0 = -127.5f;
    float maxB_0 = -127.5f;
    float maxG_0 = -127.5f;
    float maxA_0 = -127.5f;
#endif
#if PLANE_COUNT > 1
	FLOAT_PRECISION vec4 s1 = vec4(0.0f);
    float maxR_1 = -127.5f;
    float maxB_1 = -127.5f;
    float maxG_1 = -127.5f;
    float maxA_1 = -127.5f;
#endif
#if PLANE_COUNT > 2
	FLOAT_PRECISION vec4 s2 = vec4(0.0f);
    float maxR_2 = -127.5f;
    float maxB_2 = -127.5f;
    float maxG_2 = -127.5f;
    float maxA_2 = -127.5f;
#endif
#if PLANE_COUNT > 3
	FLOAT_PRECISION vec4 s3 = vec4(0.0f);
    float maxR_3 = -127.5f;
    float maxB_3 = -127.5f;
    float maxG_3 = -127.5f;
    float maxA_3 = -127.5f;
#endif

    ivec2 baseCoord_old = ivec2(gl_FragCoord.xy);
	ivec2 baseCoord = ivec2(int(baseCoord_old.x * NUM_STRIDE), int(baseCoord_old.y * NUM_STRIDE));
	baseCoord += ivec2(1, 1);

_PLACEHOLDER_TEXTURE_READ_
_PLACEHOLDER_CALCULATION_

#if PLANE_COUNT > 0 
    s = vec4(maxR_0, maxG_0, maxB_0, maxA_0);
#endif
#if PLANE_COUNT > 1
	s1 = vec4(maxR_1, maxG_1, maxB_1, maxA_1);
#endif
#if PLANE_COUNT > 2
	s2 = vec4(maxR_2, maxG_2, maxB_2, maxA_2);
#endif
#if PLANE_COUNT > 3
	s3 = vec4(maxR_3, maxG_3, maxB_3, maxA_3);
#endif
    