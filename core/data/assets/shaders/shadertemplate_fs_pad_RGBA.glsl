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

// Defines to be added
// PADDING_T : padding on the top side of texture
// PADDING_B : padding on the bottom side of texture
// PADDING_L : padding on the left side of the texture
// PADDING_R : padding on the right side of the texture

precision FLOAT_PRECISION float;
precision FLOAT_PRECISION sampler2DArray;
precision FLOAT_PRECISION sampler2D;

layout(location = 0)out vec4 o_pixel;

#ifdef INPUT_TEXTURE_2D
layout(binding = 0) uniform sampler2D inputTextures;
#define TEXTURE(t, c) texture((t), (c).xy)
#else
layout(binding = 0) uniform sampler2DArray inputTextures;
#define TEXTURE(t, c) texture((t), (c))
#endif

//These should be defined inside the shader loader code
//#define INPUT_WIDTH 540.0
//#define INPUT_HEIGHT 960.0

//#define USE_COMPONENT_G
//#define USE_COMPONENT_B
//#define USE_COMPONENT_A

bool checkValid (FLOAT_PRECISION vec2 coords) {
    bool retVal = true;
    retVal = retVal && !((coords.x >= 1.0f || coords.x < 0.0f));
    retVal = retVal && !((coords.y >= 1.0f || coords.y < 0.0f));
    return retVal;
}

FLOAT_PRECISION vec2 replicatePadding(FLOAT_PRECISION vec2 sourceCoords, FLOAT_PRECISION vec2 offsetCoords) {
    FLOAT_PRECISION vec2 repCoords = vec2(sourceCoords.x, sourceCoords.y);
    if (sourceCoords.x >= 1.0f) repCoords.x = 1.0f - offsetCoords.x;
    if (sourceCoords.x < 0.0f)  repCoords.x = 0.0f + offsetCoords.x;
    if (sourceCoords.y >= 1.0f) repCoords.y = 1.0f - offsetCoords.y;
    if (sourceCoords.y < 0.0f)  repCoords.y = 0.0f + offsetCoords.y;
    //repCoords = vec2(8.0/8.0, 8.0/8.0);
    return repCoords;
}

FLOAT_PRECISION vec2 reflectPadding(FLOAT_PRECISION vec2 sourceCoords, FLOAT_PRECISION vec2 offsetCoords) {
    FLOAT_PRECISION vec2 repCoords = vec2(sourceCoords.x, sourceCoords.y);
    if (sourceCoords.x >= 1.0f) repCoords.x = 2.0f - sourceCoords.x - 2.0*offsetCoords.x;
    if (sourceCoords.x < 0.0f)  repCoords.x = (0.0f - sourceCoords.x) + 2.0*offsetCoords.x;
    if (sourceCoords.y >= 1.0f) repCoords.y = 2.0f - sourceCoords.y  - 2.0*offsetCoords.y;
    if (sourceCoords.y < 0.0f)  repCoords.y = (0.0f - sourceCoords.y) + 2.0*offsetCoords.y;
    return repCoords;
}

FLOAT_PRECISION vec2 checkboardPadding(FLOAT_PRECISION vec2 sourceCoords) {
    FLOAT_PRECISION vec2 repCoords = vec2(0.0, 0.0);
    if (sourceCoords.x >= 1.0f) repCoords.x = sourceCoords.x - 1.0f;
    if (sourceCoords.x < 0.0f)  repCoords.x = sourceCoords.x + 1.0f;
    if (sourceCoords.y >= 1.0f) repCoords.y = sourceCoords.y + 1.0f;
    if (sourceCoords.y < 0.0f)  repCoords.y = sourceCoords.y + 1.0f;
    return repCoords;
}

void main()
{
    // Convolution Process
    const int lod = 0;
    FLOAT_PRECISION vec2 maxUV = vec2(INPUT_WIDTH, INPUT_HEIGHT);//textureSize(inputTextures[0], lod);
    FLOAT_PRECISION vec4 s = vec4(0.0f);

    // Pre-calculate coordinates here, note that
    // textureGather: pixel should be in the center of the four surrounding pixels, use zero offset
    // texture: Pixel should be in the center of the pixel since GL_NEAREST, use 0.5 offset
    // It's also verified that using texture varying and have vertex shader generate those has same performance.
	ivec2 baseCoord_old = ivec2(gl_FragCoord.xy);
	ivec2 baseCoord = ivec2(int(baseCoord_old.x), int(baseCoord_old.y));
	baseCoord += ivec2(1, 1);
	
    vec2 baseCoordFloat = vec2(baseCoord);

    FLOAT_PRECISION vec2 texCoords;
    FLOAT_PRECISION vec4 texVals;
    FLOAT_PRECISION float offsetsT;
    FLOAT_PRECISION float offsetsB;

    float initValT = -0.5 - float(PADDING_T);
    float initValL = -0.5 - float(PADDING_L);

    offsetsT = initValT;
    offsetsB = initValL;

    if (1==0) {
        s = vec4(0.0, 0.0, 0.0, 0.0);
    } else {
        texCoords = (baseCoordFloat + vec2(offsetsT, offsetsB) ) / maxUV;

        LAYER_CALCULATION    
        texVals = vec4(0.0, 0.0, 0.0, 0.0);
        if (checkValid(texCoords)) {
            texVals = TEXTURE(inputTextures, vec3(texCoords, layer));
        } else {

#ifdef CONST_PADDING
            texVals = (checkValid(texCoords)) ? TEXTURE(inputTextures, vec3(texCoords, layer)) : vec4(PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE);
#endif
#ifdef REPLICATE_PADDING
            FLOAT_PRECISION vec2 repCoords = replicatePadding(texCoords, vec2(0.5f,0.5f)/maxUV);
            texVals = (checkValid(texCoords)) ? TEXTURE(inputTextures, vec3(texCoords, layer)) : TEXTURE(inputTextures, vec3(repCoords, layer));
	    //texVals = vec4(repCoords.y);
#endif
#ifdef REFLECT_PADDING
            FLOAT_PRECISION vec2 repCoords = reflectPadding(texCoords, vec2(0.5f,0.5f)/maxUV);
            texVals = (checkValid(texCoords)) ? TEXTURE(inputTextures, vec3(texCoords, layer)) : TEXTURE(inputTextures, vec3(repCoords, layer));
	    //texVals = vec4(repCoords.x);
#endif
#ifdef CHECKBOARD_PADDING
            FLOAT_PRECISION vec2 repCoords = checkboardPadding(texCoords);
            texVals = (checkValid(texCoords)) ? TEXTURE(inputTextures, vec3(texCoords, layer)) : TEXTURE(inputTextures, vec3(repCoords, layer));
#endif
        }

        s.r = texVals.x;
#ifdef USE_COMPONENT_G
        s.g = texVals.y;
#endif
#ifdef USE_COMPONENT_B
        s.b = texVals.z;
#endif
#ifdef USE_COMPONENT_A
        s.a = texVals.w;
#endif
    }
    o_pixel = s;
}
