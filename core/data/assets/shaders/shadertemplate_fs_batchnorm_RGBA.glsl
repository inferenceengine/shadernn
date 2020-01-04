#define FLOAT_PRECISION _PLACEHOLDER_PRECISION_

// Defines to be added
// PADDING_T : padding on the top side of texture
// PADDING_B : padding on the bottom side of texture
// PADDING_L : padding on the left side of the texture
// PADDING_R : padding on the right side of the texture

precision FLOAT_PRECISION float;
precision FLOAT_PRECISION sampler2DArray;

layout(location = 0)out vec4 o_pixel;

#ifdef INPUT_TEXTURE_2D
layout(binding = 0) uniform sampler2D inputTextures;
#define TEXTURE(t, c) texture((t), (c).xy)
#else
layout(binding = 0) uniform sampler2DArray inputTextures;
#define TEXTURE(t, c) texture((t), (c))
#endif

const FLOAT_PRECISION vec4 beta = _PLACEHOLDER_BETA_VEC_CONSTANTS_;
const FLOAT_PRECISION vec4 gamma = _PLACEHOLDER_GAMMA_VEC_CONSTANTS_;
const FLOAT_PRECISION vec4 movingMean = _PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_;
const FLOAT_PRECISION vec4 movingVariance = _PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_;

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

    texCoords = (baseCoordFloat + vec2(-0.5, -0.5) ) / maxUV;

    LAYER_CALCULATION    
    texVals = vec4(0.0, 0.0, 0.0, 0.0);
    texVals = TEXTURE(inputTextures, vec3(texCoords, layer));
    s = texVals;
    FLOAT_PRECISION vec4 sqrtVar = sqrt(movingVariance + vec4(0.001f));
    sqrtVar = max(sqrtVar, vec4(0.0001f));
    s = ((gamma/sqrtVar) * (s - movingMean)) + beta;
    o_pixel = s;
}
