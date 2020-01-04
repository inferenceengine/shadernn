#define N_DIMS NUM_KERNEL_SIZE*NUM_KERNEL_SIZE
#define FLOAT_PRECISION _PLACEHOLDER_PRECISION_

// Defines to be added
// PADDING_T : padding on the top side of texture
// PADDING_B : padding on the bottom side of texture
// PADDING_L : padding on the left side of the texture
// PADDING_R : padding on the right side of the texture

// CONST_PADDING : Padding with a const value
// REPLICATE_PADDING : Padding in a reflective fashion
// CHECKBOARD_PADDING : Padding in a checkboard fashion

// USE_UNIFORM_WEIGHTS : Use uniform weights or old method

// NEW PLACEHOLDERS:
// _PLACEHOLDER_UNIFORM_WEIGHTS : Add logic for binding weights and biases as uniforms
// _PLACEHOLDER_OUT_LAYER_ : 4 * output layer where the current output will be stored

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

#ifdef INPUT_TEXTURE_2D // ends on line 42
layout(binding = 0) uniform sampler2D inputTextures;
#define TEXTURE(t, c) texture((t), (c).xy)
#else // else for if on line 36
layout(binding = 0) uniform sampler2DArray inputTextures;
#define TEXTURE(t, c) texture((t), (c))
#endif // end if from line 36

#if NUM_INPUT_PLANES > 4
#define WEIGHT_SAMPLER sampler2DArray
#define weightFetch(t, c) texture((t), (c))
#else
#define WEIGHT_SAMPLER sampler2D
#define weightFetch(t, c) texture((t), (c).xy)
#endif

#ifdef USE_UNIFORM_WEIGHTS // ends on line 60
uniform vec4 bias[PLANE_COUNT];
#ifdef USE_BATCH_NORMALIZATION
uniform vec4 beta[PLANE_COUNT];
uniform vec4 gamma[PLANE_COUNT];
uniform vec4 movingMean[PLANE_COUNT];
uniform vec4 movingVariance[PLANE_COUNT];
#endif // end if from line 46
#else  // else forif from line 44
_PLACEHOLDER_BIAS_CONSTANTS_
#ifdef USE_BATCH_NORMALIZATION
const vec4 beta[] = vec4[](_PLACEHOLDER_BETA_VEC_CONSTANTS_);
const vec4 gamma[] = vec4[](_PLACEHOLDER_GAMMA_VEC_CONSTANTS_);
const vec4 movingMean[] = vec4[](_PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_);
const vec4 movingVariance[] = vec4[](_PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_);
#endif // end if from line 54
#endif // end if from line 44

#ifdef USE_WEIGHT_BUFFERS // ends on line 143
#ifdef USE_COMPONENT_R_PLANE_0 // ends on line 67
layout(STORAGE_FORMAT, binding = 2) VARIABLE_SPECIFIER weightMatrix1 {
    FLOAT_PRECISION vec4 weights1[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 63
#ifdef USE_COMPONENT_G_PLANE_0 // ends on line 72
layout(STORAGE_FORMAT, binding = 3) VARIABLE_SPECIFIER weightMatrix2 {
    FLOAT_PRECISION vec4 weights2[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 68
#ifdef USE_COMPONENT_B_PLANE_0 // ends on line 77
layout(STORAGE_FORMAT, binding = 4) VARIABLE_SPECIFIER weightMatrix3 {
    FLOAT_PRECISION vec4 weights3[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 73
#ifdef USE_COMPONENT_A_PLANE_0 // ends on line 
layout(STORAGE_FORMAT, binding = 5) VARIABLE_SPECIFIER weightMatrix4 {
    FLOAT_PRECISION vec4 weights4[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 82
#ifdef USE_COMPONENT_R_PLANE_1
layout(STORAGE_FORMAT, binding = 6) VARIABLE_SPECIFIER weightMatrix5 {
    FLOAT_PRECISION vec4 weights5[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 83
#ifdef USE_COMPONENT_G_PLANE_1
layout(STORAGE_FORMAT, binding = 7) VARIABLE_SPECIFIER weightMatrix6 {
    FLOAT_PRECISION vec4 weights6[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 88
#ifdef USE_COMPONENT_B_PLANE_1
layout(STORAGE_FORMAT, binding = 8) VARIABLE_SPECIFIER weightMatrix7 {
    FLOAT_PRECISION vec4 weights7[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 93
#ifdef USE_COMPONENT_A_PLANE_1
layout(STORAGE_FORMAT, binding = 9) VARIABLE_SPECIFIER weightMatrix8 {
    FLOAT_PRECISION vec4 weights8[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 98
#ifdef USE_COMPONENT_R_PLANE_2
layout(STORAGE_FORMAT, binding = 10) VARIABLE_SPECIFIER weightMatrix9 {
    FLOAT_PRECISION vec4 weights9[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 103
#ifdef USE_COMPONENT_G_PLANE_2
layout(STORAGE_FORMAT, binding = 11) VARIABLE_SPECIFIER weightMatrix10 {
    FLOAT_PRECISION vec4 weights10[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 108
#ifdef USE_COMPONENT_B_PLANE_2
layout(STORAGE_FORMAT, binding = 12) VARIABLE_SPECIFIER weightMatrix11 {
    FLOAT_PRECISION vec4 weights11[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 113
#ifdef USE_COMPONENT_A_PLANE_2
layout(STORAGE_FORMAT, binding = 13) VARIABLE_SPECIFIER weightMatrix12 {
    FLOAT_PRECISION vec4 weights12[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 118
#ifdef USE_COMPONENT_R_PLANE_3
layout(STORAGE_FORMAT, binding = 14) VARIABLE_SPECIFIER weightMatrix13 {
    FLOAT_PRECISION vec4 weights13[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 123
#ifdef USE_COMPONENT_G_PLANE_3
layout(STORAGE_FORMAT, binding = 15) VARIABLE_SPECIFIER weightMatrix14 {
    FLOAT_PRECISION vec4 weights14[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 128
#ifdef USE_COMPONENT_B_PLANE_3
layout(STORAGE_FORMAT, binding = 16) VARIABLE_SPECIFIER weightMatrix15 {
    FLOAT_PRECISION vec4 weights15[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 133
#ifdef USE_COMPONENT_A_PLANE_3
layout(STORAGE_FORMAT, binding = 17) VARIABLE_SPECIFIER weightMatrix16 {
    FLOAT_PRECISION vec4 weights16[NUM_INPUT_PLANES * N_DIMS];
};
#endif // end if from line 138
#endif // end if from line 62
#ifdef USE_WEIGHT_TEXTURES //ends on line
#ifdef USE_COMPONENT_R_PLANE_0
layout(binding = 2) uniform WEIGHT_SAMPLER weightMatrix1;
#endif // end if from line 145
// FLOAT_PRECISION vec4 rDot[N_VEC4];
#ifdef USE_COMPONENT_G_PLANE_0
layout(binding = 3) uniform WEIGHT_SAMPLER weightMatrix2;
// FLOAT_PRECISION vec4 gDot[N_VEC4];
#endif // end if from line 150
#ifdef USE_COMPONENT_B_PLANE_0
layout(binding = 4) uniform WEIGHT_SAMPLER weightMatrix3;
// FLOAT_PRECISION vec4 bDot[N_VEC4];
#endif // end if from line 153
#ifdef USE_COMPONENT_A_PLANE_0
layout(binding = 5) uniform WEIGHT_SAMPLER weightMatrix4;
#endif // end if from line 157
#ifdef USE_COMPONENT_R_PLANE_1
layout(binding = 6) uniform WEIGHT_SAMPLER weightMatrix5;
#endif // end if from line 160
// FLOAT_PRECISION vec4 rDot[N_VEC4];
#ifdef USE_COMPONENT_G_PLANE_1
layout(binding = 7) uniform WEIGHT_SAMPLER weightMatrix6;
// FLOAT_PRECISION vec4 gDot[N_VEC4];
#endif // end if from line 164
#ifdef USE_COMPONENT_B_PLANE_1
layout(binding = 8) uniform WEIGHT_SAMPLER weightMatrix7;
// FLOAT_PRECISION vec4 bDot[N_VEC4];
#endif // end if from line 168
#ifdef USE_COMPONENT_A_PLANE_1
layout(binding = 9) uniform WEIGHT_SAMPLER weightMatrix8;
#endif // end if from line 172
#ifdef USE_COMPONENT_R_PLANE_2
layout(binding = 10) uniform WEIGHT_SAMPLER weightMatrix9;
#endif // end if from line 177
// FLOAT_PRECISION vec4 rDot[N_VEC4];
#ifdef USE_COMPONENT_G_PLANE_2
layout(binding = 11) uniform WEIGHT_SAMPLER weightMatrix10;
// FLOAT_PRECISION vec4 gDot[N_VEC4];
#endif // end if from line 179
#ifdef USE_COMPONENT_B_PLANE_2
layout(binding = 12) uniform WEIGHT_SAMPLER weightMatrix11;
// FLOAT_PRECISION vec4 bDot[N_VEC4];
#endif // end if from line 183
#ifdef USE_COMPONENT_A_PLANE_2
layout(binding = 13) uniform WEIGHT_SAMPLER weightMatrix12;
#endif // end if from line 187
#ifdef USE_COMPONENT_R_PLANE_3
layout(binding = 14) uniform WEIGHT_SAMPLER weightMatrix13;
#endif // end if from line 190
// FLOAT_PRECISION vec4 rDot[N_VEC4];
#ifdef USE_COMPONENT_G_PLANE_3
layout(binding = 15) uniform WEIGHT_SAMPLER weightMatrix14;
// FLOAT_PRECISION vec4 gDot[N_VEC4];
#endif // end if from line 194
#ifdef USE_COMPONENT_B_PLANE_3
layout(binding = 16) uniform WEIGHT_SAMPLER weightMatrix15;
// FLOAT_PRECISION vec4 bDot[N_VEC4];
#endif // end if from line 198
#ifdef USE_COMPONENT_A_PLANE_3
layout(binding = 17) uniform WEIGHT_SAMPLER weightMatrix16;
#endif // end if from line 202
#endif // end if from line 144
#ifdef USE_WEIGHT_CONSTANTS
#ifdef USE_COMPONENT_R_PLANE_0
const FLOAT_PRECISION vec4 weights1[] = vec4[](
_PLACEHOLDER_WEIGHT1_VEC_CONSTANTS_
);
#endif // end if from line 207
#ifdef USE_COMPONENT_G_PLANE_0
const FLOAT_PRECISION vec4 weights2[] = vec4[](
_PLACEHOLDER_WEIGHT2_VEC_CONSTANTS_
);
#endif // end if from line 212
#ifdef USE_COMPONENT_B_PLANE_0
const FLOAT_PRECISION vec4 weights3[] = vec4[](
_PLACEHOLDER_WEIGHT3_VEC_CONSTANTS_
);
#endif // end if from line 217
#ifdef USE_COMPONENT_A_PLANE_0
const FLOAT_PRECISION vec4 weights4[] = vec4[](
_PLACEHOLDER_WEIGHT4_VEC_CONSTANTS_
);
#endif // end if from line 222
#ifdef USE_COMPONENT_R_PLANE_1
const FLOAT_PRECISION vec4 weights5[] = vec4[](
_PLACEHOLDER_WEIGHT5_VEC_CONSTANTS_
);
#endif // end if from line 227
#ifdef USE_COMPONENT_G_PLANE_1
const FLOAT_PRECISION vec4 weights6[] = vec4[](
_PLACEHOLDER_WEIGHT6_VEC_CONSTANTS_
);
#endif // end if from line 232
#ifdef USE_COMPONENT_B_PLANE_1
const FLOAT_PRECISION vec4 weights7[] = vec4[](
_PLACEHOLDER_WEIGHT7_VEC_CONSTANTS_
);
#endif // end if from line 237
#ifdef USE_COMPONENT_A_PLANE_1
const FLOAT_PRECISION vec4 weights8[] = vec4[](
_PLACEHOLDER_WEIGHT8_VEC_CONSTANTS_
);
#endif // end if from line 242
#ifdef USE_COMPONENT_R_PLANE_2
const FLOAT_PRECISION vec4 weights9[] = vec4[](
_PLACEHOLDER_WEIGHT9_VEC_CONSTANTS_
);
#endif // end if from line 247
#ifdef USE_COMPONENT_G_PLANE_2
const FLOAT_PRECISION vec4 weights10[] = vec4[](
_PLACEHOLDER_WEIGHT10_VEC_CONSTANTS_
);
#endif // end if from line 252
#ifdef USE_COMPONENT_B_PLANE_2
const FLOAT_PRECISION vec4 weights11[] = vec4[](
_PLACEHOLDER_WEIGHT11_VEC_CONSTANTS_
);
#endif // end if from line 257
#ifdef USE_COMPONENT_A_PLANE_2
const FLOAT_PRECISION vec4 weights12[] = vec4[](
_PLACEHOLDER_WEIGHT12_VEC_CONSTANTS_
);
#endif // end if from line 262
#ifdef USE_COMPONENT_R_PLANE_3
const FLOAT_PRECISION vec4 weights13[] = vec4[](
_PLACEHOLDER_WEIGHT13_VEC_CONSTANTS_
);
#endif // end if from line 267
#ifdef USE_COMPONENT_G_PLANE_3
const FLOAT_PRECISION vec4 weights14[] = vec4[](
_PLACEHOLDER_WEIGHT14_VEC_CONSTANTS_
);
#endif // end if from line 272
#ifdef USE_COMPONENT_B_PLANE_3
const FLOAT_PRECISION vec4 weights15[] = vec4[](
_PLACEHOLDER_WEIGHT15_VEC_CONSTANTS_
);
#endif // end if from line 277
#ifdef USE_COMPONENT_A_PLANE_3
const FLOAT_PRECISION vec4 weights16[] = vec4[](
_PLACEHOLDER_WEIGHT16_VEC_CONSTANTS_
);
#endif // end if from line 282
#endif // end if from line 206

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
    FLOAT_PRECISION vec4 s2= vec4(0.0f);
    FLOAT_PRECISION vec4 s3 = vec4(0.0f);

    // Pre-calculate coordinates here, note that
    // textureGather: pixel should be in the center of the four surrounding pixels, use zero offset
    // texture: Pixel should be in the center of the pixel since GL_NEAREST, use 0.5 offset
    // It's also verified that using texture varying and have vertex shader generate those has same performance.
	ivec2 baseCoord_old = ivec2(gl_FragCoord.xy);
	ivec2 baseCoord = ivec2(int(baseCoord_old.x * NUM_STRIDE), int(baseCoord_old.y * NUM_STRIDE));
	baseCoord += ivec2(1, 1);

#ifdef USE_UNIFORM_WEIGHTS
    vec2 baseCoordFloat = vec2(baseCoord);

    FLOAT_PRECISION vec2 texCoords[N_DIMS];
    FLOAT_PRECISION vec4 texVals[N_DIMS];
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

    for (int plane = 0; plane < NUM_INPUT_PLANES; plane+=4) {
        int layer = plane >> 2;
#ifdef USE_MULTI_INPUTS
        layer = (plane + 4 * int((NUM_INPUT_PLANES + 3) / 4)) >> 2;
#endif // end if from line 358
        for (int i = 0; i < NUM_KERNEL_SIZE; i++) {
            for (int j = 0; j < NUM_KERNEL_SIZE; j++) {
                int arrayAccess = i * NUM_KERNEL_SIZE + j;
#ifdef CLAMPED_PADDING
                texVals[arrayAccess] = TEXTURE(inputTextures, vec3(texCoords[arrayAccess], layer));
#endif
#ifdef CONST_PADDING
                texVals[arrayAccess] = (checkValid(texCoords[arrayAccess])) ? TEXTURE(inputTextures, vec3(texCoords[arrayAccess], layer)) : vec4(PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE);
#endif // end if from line 364
#ifdef REPLICATE_PADDING
                FLOAT_PRECISION vec2 repCoords = replicatePadding(texCoords[arrayAccess]);
                texVals[arrayAccess] = (checkValid(texCoords[arrayAccess])) ? TEXTURE(inputTextures, vec3(texCoords[arrayAccess], layer)) : TEXTURE(inputTextures, vec3(repCoords, layer));
#endif // end if from line 367
#ifdef CHECKBOARD_PADDING
                FLOAT_PRECISION vec2 repCoords = checkboardPadding(texCoords[arrayAccess]);
                texVals[arrayAccess] = (checkValid(texCoords[arrayAccess])) ? TEXTURE(inputTextures, vec3(texCoords[arrayAccess], layer)) : TEXTURE(inputTextures, vec3(repCoords, layer));
#endif // end if from line 371
#ifdef REMOVE_ZERO
                texVals[arrayAccess] = max(texVals[arrayAccess], vec4(0.001));
#endif // end if from line 375
            }
        }

#ifdef USE_WEIGHT_TEXTURES
        for (int i = 0; i < N_DIMS; i++) {
            // rDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix1, vec3(weightsCoords[i], layer)));
#ifdef USE_COMPONENT_R_PLANE_0
            s.r += dot(texVals[i], weightFetch(weightMatrix1, vec3(weightsCoords[i], layer)));
#endif // end if from line 384
#ifdef USE_COMPONENT_G_PLANE_0
            // gDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix2, vec3(weightsCoords[i], layer)));
            s.g += dot(texVals[i], weightFetch(weightMatrix2, vec3(weightsCoords[i], layer)));
#endif // end if from line 387
#ifdef USE_COMPONENT_B_PLANE_0
            // bDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix3, vec3(weightsCoords[i], layer)));
            s.b += dot(texVals[i], weightFetch(weightMatrix3, vec3(weightsCoords[i], layer)));
#endif // end if from line 391
#ifdef USE_COMPONENT_A_PLANE_0
            // aDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix4, vec3(weightsCoords[i], layer)));
            s.a += dot(texVals[i], weightFetch(weightMatrix4, vec3(weightsCoords[i], layer)));
#endif // end if from line 395
#ifdef USE_COMPONENT_R_PLANE_1
            s1.r += dot(texVals[i], weightFetch(weightMatrix5, vec3(weightsCoords[i], layer)));
#endif // end if from line 399
#ifdef USE_COMPONENT_G_PLANE_1
            // gDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix2, vec3(weightsCoords[i], layer)));
            s1.g += dot(texVals[i], weightFetch(weightMatrix6, vec3(weightsCoords[i], layer)));
#endif // end if from line 402
#ifdef USE_COMPONENT_B_PLANE_1
            // bDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix3, vec3(weightsCoords[i], layer)));
            s1.b += dot(texVals[i], weightFetch(weightMatrix7, vec3(weightsCoords[i], layer)));
#endif // end if from line 406
#ifdef USE_COMPONENT_A_PLANE_1
            // aDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix4, vec3(weightsCoords[i], layer)));
            s1.a += dot(texVals[i], weightFetch(weightMatrix8, vec3(weightsCoords[i], layer)));
#endif // end if from line 410
#ifdef USE_COMPONENT_R_PLANE_2
            s2.r += dot(texVals[i], weightFetch(weightMatrix9, vec3(weightsCoords[i], layer)));
#endif // end if from line 414
#ifdef USE_COMPONENT_G_PLANE_2
            // gDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix2, vec3(weightsCoords[i], layer)));
            s2.g += dot(texVals[i], weightFetch(weightMatrix10, vec3(weightsCoords[i], layer)));
#endif // end if from line 417
#ifdef USE_COMPONENT_B_PLANE_2
            // bDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix3, vec3(weightsCoords[i], layer)));
            s2.b += dot(texVals[i], weightFetch(weightMatrix11, vec3(weightsCoords[i], layer)));
#endif // end if from line 421
#ifdef USE_COMPONENT_A_PLANE_2
            // aDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix4, vec3(weightsCoords[i], layer)));
            s2.a += dot(texVals[i], weightFetch(weightMatrix12, vec3(weightsCoords[i], layer)));
#endif // end if from line 425
#ifdef USE_COMPONENT_R_PLANE_3
            s3.r += dot(texVals[i], weightFetch(weightMatrix13, vec3(weightsCoords[i], layer)));
#endif // end if from line 429
#ifdef USE_COMPONENT_G_PLANE_3
            // gDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix2, vec3(weightsCoords[i], layer)));
            s3.g += dot(texVals[i], weightFetch(weightMatrix14, vec3(weightsCoords[i], layer)));
#endif // end if from line 432
#ifdef USE_COMPONENT_B_PLANE_3
            // bDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix3, vec3(weightsCoords[i], layer)));
            s3.b += dot(texVals[i], weightFetch(weightMatrix15, vec3(weightsCoords[i], layer)));
#endif // end if from line 436
#ifdef USE_COMPONENT_A_PLANE_3
            // aDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix4, vec3(weightsCoords[i], layer)));
            s3.a += dot(texVals[i], weightFetch(weightMatrix16, vec3(weightsCoords[i], layer)));
#endif // end if from line 440
        }
#endif // end if from line 381
#ifdef USE_WEIGHT_BUFFERS
        for (int i = 0; i < N_DIMS; i++) {
            // rDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix1, vec3(weightsCoords[i], layer)));
#ifdef USE_COMPONENT_R_PLANE_0
            s.r += dot(texVals[i], weights1[layer * N_DIMS + i]);
#endif // end if from line 449
#ifdef USE_COMPONENT_G_PLANE_0
            // gDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix2, vec3(weightsCoords[i], layer)));
            s.g += dot(texVals[i], weights2[layer * N_DIMS + i]);
#endif // end if from line 452
#ifdef USE_COMPONENT_B_PLANE_0
            // bDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix3, vec3(weightsCoords[i], layer)));
            s.b += dot(texVals[i], weights3[layer * N_DIMS + i]);
#endif // end if from line 456
#ifdef USE_COMPONENT_A_PLANE_0
            // aDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix4, vec3(weightsCoords[i], layer)));
            s.a += dot(texVals[i], weights4[layer * N_DIMS + i]);
#endif
#ifdef USE_COMPONENT_R_PLANE_1
            s1.r += dot(texVals[i], weights5[layer * N_DIMS + i]);
#endif // end if from line 464
#ifdef USE_COMPONENT_G_PLANE_1
            // gDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix2, vec3(weightsCoords[i], layer)));
            s1.g += dot(texVals[i], weights6[layer * N_DIMS + i]);
#endif // end if from line 467
#ifdef USE_COMPONENT_B_PLANE_1
            // bDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix3, vec3(weightsCoords[i], layer)));
            s1.b += dot(texVals[i], weights7[layer * N_DIMS + i]);
#endif // end if from line 471
#ifdef USE_COMPONENT_A_PLANE_1
            // aDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix4, vec3(weightsCoords[i], layer)));
            s1.a += dot(texVals[i], weights8[layer * N_DIMS + i]);
#endif // end if from line 475
#ifdef USE_COMPONENT_R_PLANE_2
            s2.r += dot(texVals[i], weights9[layer * N_DIMS + i]);
#endif // end if from line 479
#ifdef USE_COMPONENT_G_PLANE_2
            // gDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix2, vec3(weightsCoords[i], layer)));
            s2.g += dot(texVals[i], weights10[layer * N_DIMS + i]);
#endif // end if from line 482
#ifdef USE_COMPONENT_B_PLANE_2
            // bDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix3, vec3(weightsCoords[i], layer)));
            s2.b += dot(texVals[i], weights11[layer * N_DIMS + i]);
#endif // end if from line 486
#ifdef USE_COMPONENT_A_PLANE_2
            // aDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix4, vec3(weightsCoords[i], layer)));
            s2.a += dot(texVals[i], weights12[layer * N_DIMS + i]);
#endif // end if from line 490
#ifdef USE_COMPONENT_R_PLANE_3
            s3.r += dot(texVals[i], weights13[layer * N_DIMS + i]);
#endif // end if from line 494
#ifdef USE_COMPONENT_G_PLANE_3
            // gDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix2, vec3(weightsCoords[i], layer)));
            s3.g += dot(texVals[i], weights14[layer * N_DIMS + i]);
#endif // end if from line 497
#ifdef USE_COMPONENT_B_PLANE_3
            // bDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix3, vec3(weightsCoords[i], layer)));
            s3.b += dot(texVals[i], weights15[layer * N_DIMS + i]);
#endif // end if from line 501
#ifdef USE_COMPONENT_A_PLANE_3
            // aDot[(i + 3) / 4][i % 4] = dot(texVals[i], TEXTURE(weightMatrix4, vec3(weightsCoords[i], layer)));
            s3.a += dot(texVals[i], weights16[layer * N_DIMS + i]);
#endif // end if from line 505
        }
#endif
    }
#endif // 
#ifdef USE_WEIGHT_CONSTANTS

_PLACEHOLDER_ELEMENT_ACCESS_

_PLACEHOLDER_CALC_

        }
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
//}
