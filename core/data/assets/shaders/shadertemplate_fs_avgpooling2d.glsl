
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
    FLOAT_PRECISION vec4 s = vec4(0.0f);
    FLOAT_PRECISION vec4 s1 = vec4(0.0f);
    FLOAT_PRECISION vec4 s2 = vec4(0.0f);
    FLOAT_PRECISION vec4 s3 = vec4(0.0f);
    FLOAT_PRECISION vec2 maxUV = vec2(INPUT_WIDTH, INPUT_HEIGHT);

    ivec2 baseCoord_old = ivec2(gl_FragCoord.xy);
    ivec2 baseCoord = ivec2(int(baseCoord_old.x * NUM_STRIDE), int(baseCoord_old.y * NUM_STRIDE));
    baseCoord += ivec2(1, 1);

_PLACEHOLDER_TEXTURE_READ_
_PLACEHOLDER_CALCULATION_

