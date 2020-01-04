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

uniform FLOAT_PRECISION vec4 bias;

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

const mediump vec4 weightMatrix1[] = vec4[](
_PLACEHOLDER_WEIGHT1_VEC_CONSTANTS_);

#ifdef USE_COMPONENT_G
const mediump vec4 weightMatrix2[] = vec4[](
_PLACEHOLDER_WEIGHT2_VEC_CONSTANTS_);
#endif

#ifdef USE_COMPONENT_B
const mediump vec4 weightMatrix3[] = vec4[](
_PLACEHOLDER_WEIGHT3_VEC_CONSTANTS_);
#endif

#ifdef USE_COMPONENT_A
const mediump vec4 weightMatrix4[] = vec4[](
_PLACEHOLDER_WEIGHT4_VEC_CONSTANTS_);
#endif

#ifdef USE_BATCH_NORMALIZATION
const mediump vec4 beta = _PLACEHOLDER_BETA_VEC_CONSTANTS_;
const mediump vec4 gamma = _PLACEHOLDER_GAMMA_VEC_CONSTANTS_;
const mediump vec4 movingMean = _PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_;
const mediump vec4 movingVariance = _PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_;
#endif

int checkTexel(highp vec2 coord, highp vec2 maxUV) {
    float stride = float(NUM_STRIDE);
    if (coord.x < 0.0f || coord.x > (maxUV.x + (maxUV.x - 1.0f) * (stride - 1.0f)) ||
        coord.y < 0.0f || coord.y > (maxUV.y + (maxUV.y - 1.0f) * (stride - 1.0f)) ||
        int((coord.x - 0.5)) % NUM_STRIDE != 0 || int((coord.y - 0.5)) % NUM_STRIDE != 0) {
        return 0;
    }
    else {
        return 1;
    }
}

void main()
{
// do boundary check
if (gl_GlobalInvocationID.x >= uint(INPUT_WIDTH * NUM_STRIDE) || gl_GlobalInvocationID.y >= uint(INPUT_HEIGHT * NUM_STRIDE)) return;


// Convolution Process
const int lod = 0;
const vec2 maxUV = vec2(INPUT_WIDTH, INPUT_HEIGHT);//textureSize(inputTextures[0], lod);
const vec2 maxUV_stride = float(NUM_STRIDE) * vec2(maxUV);

// Pre-calculate coordinates here, note that
// textureGather: pixel should be in the center of the four surrounding pixels, use zero offset
// texture: Pixel should be in the center of the pixel since GL_NEAREST, use 0.5 offset
// It's also verified that using texture varying and have vertex shader generate those has same performance.
// for stride 1: offset = vec2(0.0f, 0.0f);
ivec2 baseCoord = ivec2(gl_GlobalInvocationID.xy);

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

FLOAT_PRECISION vec4 s = vec4(0.0f);

for (int i = 0; i < NUM_INPUT_PLANES; i += 4) {
    int layer = i >> 2;
    FLOAT_PRECISION vec4 t0  = checkTexel(texCoord_0,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_0 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t1  = checkTexel(texCoord_1,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_1 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t2  = checkTexel(texCoord_2,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_2 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t3  = checkTexel(texCoord_3,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_3 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t4  = checkTexel(texCoord_4,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_4 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t5  = checkTexel(texCoord_5,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_5 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t6  = checkTexel(texCoord_6,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_6 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t7  = checkTexel(texCoord_7,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_7 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t8  = checkTexel(texCoord_8,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_8 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t9  = checkTexel(texCoord_9,  maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_9 ) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t10 = checkTexel(texCoord_10, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_10) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t11 = checkTexel(texCoord_11, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_11) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t12 = checkTexel(texCoord_12, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_12) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t13 = checkTexel(texCoord_13, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_13) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t14 = checkTexel(texCoord_14, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_14) / maxUV_stride, layer)) : vec4(0.0f);
    FLOAT_PRECISION vec4 t15 = checkTexel(texCoord_15, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_15) / maxUV_stride, layer)) : vec4(0.0f);
    if (i == 0) {
        int index = NUM_KERNEL_SIZE * NUM_KERNEL_SIZE * layer;
        s.r += (dot(t0, weightMatrix1[0]) +
        dot(t1, weightMatrix1[1]) +
        dot(t2, weightMatrix1[2]) +
        dot(t3, weightMatrix1[3]) +
        dot(t4, weightMatrix1[4]) +
        dot(t5, weightMatrix1[5]) +
        dot(t6, weightMatrix1[6]) +
        dot(t7, weightMatrix1[7]) +
        dot(t8, weightMatrix1[8]) +
        dot(t9, weightMatrix1[9]) +
        dot(t10, weightMatrix1[10]) +
        dot(t11, weightMatrix1[11]) +
        dot(t12, weightMatrix1[12]) +
        dot(t13, weightMatrix1[13]) +
        dot(t14, weightMatrix1[14]) +
        dot(t15, weightMatrix1[15]));

        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[0]) +
        dot(t1, weightMatrix2[1]) +
        dot(t2, weightMatrix2[2]) +
        dot(t3, weightMatrix2[3]) +
        dot(t4, weightMatrix2[4]) +
        dot(t5, weightMatrix2[5]) +
        dot(t6, weightMatrix2[6]) +
        dot(t7, weightMatrix2[7]) +
        dot(t8, weightMatrix2[8]) +
        dot(t9, weightMatrix2[9]) +
        dot(t10, weightMatrix2[10]) +
        dot(t11, weightMatrix2[11]) +
        dot(t12, weightMatrix2[12]) +
        dot(t13, weightMatrix2[13]) +
        dot(t14, weightMatrix2[14]) +
        dot(t15, weightMatrix2[15]));
        #endif

        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[0]) +
        dot(t1, weightMatrix3[1]) +
        dot(t2, weightMatrix3[2]) +
        dot(t3, weightMatrix3[3]) +
        dot(t4, weightMatrix3[4]) +
        dot(t5, weightMatrix3[5]) +
        dot(t6, weightMatrix3[6]) +
        dot(t7, weightMatrix3[7]) +
        dot(t8, weightMatrix3[8]) +
        dot(t9, weightMatrix3[9]) +
        dot(t10, weightMatrix3[10]) +
        dot(t11, weightMatrix3[11]) +
        dot(t12, weightMatrix3[12]) +
        dot(t13, weightMatrix3[13]) +
        dot(t14, weightMatrix3[14]) +
        dot(t15, weightMatrix3[15]));
        #endif

        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[0]) +
        dot(t1, weightMatrix4[1]) +
        dot(t2, weightMatrix4[2]) +
        dot(t3, weightMatrix4[3]) +
        dot(t4, weightMatrix4[4]) +
        dot(t5, weightMatrix4[5]) +
        dot(t6, weightMatrix4[6]) +
        dot(t7, weightMatrix4[7]) +
        dot(t8, weightMatrix4[8]) +
        dot(t9, weightMatrix4[9]) +
        dot(t10, weightMatrix4[10]) +
        dot(t11, weightMatrix4[11]) +
        dot(t12, weightMatrix4[12]) +
        dot(t13, weightMatrix4[13]) +
        dot(t14, weightMatrix4[14]) +
        dot(t15, weightMatrix4[15]));
        #endif
    }
        #if NUM_INPUT_PLANES > 4

    else if (i == 4) {
        s.r += (dot(t0, weightMatrix1[16]) +
        dot(t1, weightMatrix1[17]) +
        dot(t2, weightMatrix1[18]) +
        dot(t3, weightMatrix1[19]) +
        dot(t4, weightMatrix1[20]) +
        dot(t5, weightMatrix1[21]) +
        dot(t6, weightMatrix1[22]) +
        dot(t7, weightMatrix1[23]) +
        dot(t8, weightMatrix1[24]) +
        dot(t9, weightMatrix1[25]) +
        dot(t10, weightMatrix1[26]) +
        dot(t11, weightMatrix1[27]) +
        dot(t12, weightMatrix1[28]) +
        dot(t13, weightMatrix1[29]) +
        dot(t14, weightMatrix1[30]) +
        dot(t15, weightMatrix1[31])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[16]) +
        dot(t1, weightMatrix2[17]) +
        dot(t2, weightMatrix2[18]) +
        dot(t3, weightMatrix2[19]) +
        dot(t4, weightMatrix2[20]) +
        dot(t5, weightMatrix2[21]) +
        dot(t6, weightMatrix2[22]) +
        dot(t7, weightMatrix2[23]) +
        dot(t8, weightMatrix2[24]) +
        dot(t9, weightMatrix2[25]) +
        dot(t10, weightMatrix2[26]) +
        dot(t11, weightMatrix2[27]) +
        dot(t12, weightMatrix2[28]) +
        dot(t13, weightMatrix2[29]) +
        dot(t14, weightMatrix2[30]) +
        dot(t15, weightMatrix2[31])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[16]) +
        dot(t1, weightMatrix3[17]) +
        dot(t2, weightMatrix3[18]) +
        dot(t3, weightMatrix3[19]) +
        dot(t4, weightMatrix3[20]) +
        dot(t5, weightMatrix3[21]) +
        dot(t6, weightMatrix3[22]) +
        dot(t7, weightMatrix3[23]) +
        dot(t8, weightMatrix3[24]) +
        dot(t9, weightMatrix3[25]) +
        dot(t10, weightMatrix3[26]) +
        dot(t11, weightMatrix3[27]) +
        dot(t12, weightMatrix3[28]) +
        dot(t13, weightMatrix3[29]) +
        dot(t14, weightMatrix3[30]) +
        dot(t15, weightMatrix3[31])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[16]) +
        dot(t1, weightMatrix4[17]) +
        dot(t2, weightMatrix4[18]) +
        dot(t3, weightMatrix4[19]) +
        dot(t4, weightMatrix4[20]) +
        dot(t5, weightMatrix4[21]) +
        dot(t6, weightMatrix4[22]) +
        dot(t7, weightMatrix4[23]) +
        dot(t8, weightMatrix4[24]) +
        dot(t9, weightMatrix4[25]) +
        dot(t10, weightMatrix4[26]) +
        dot(t11, weightMatrix4[27]) +
        dot(t12, weightMatrix4[28]) +
        dot(t13, weightMatrix4[29]) +
        dot(t14, weightMatrix4[30]) +
        dot(t15, weightMatrix4[31])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 8

    else if (i == 8) {
        s.r += (dot(t0, weightMatrix1[32]) +
        dot(t1, weightMatrix1[33]) +
        dot(t2, weightMatrix1[34]) +
        dot(t3, weightMatrix1[35]) +
        dot(t4, weightMatrix1[36]) +
        dot(t5, weightMatrix1[37]) +
        dot(t6, weightMatrix1[38]) +
        dot(t7, weightMatrix1[39]) +
        dot(t8, weightMatrix1[40]) +
        dot(t9, weightMatrix1[41]) +
        dot(t10, weightMatrix1[42]) +
        dot(t11, weightMatrix1[43]) +
        dot(t12, weightMatrix1[44]) +
        dot(t13, weightMatrix1[45]) +
        dot(t14, weightMatrix1[46]) +
        dot(t15, weightMatrix1[47])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[32]) +
        dot(t1, weightMatrix2[33]) +
        dot(t2, weightMatrix2[34]) +
        dot(t3, weightMatrix2[35]) +
        dot(t4, weightMatrix2[36]) +
        dot(t5, weightMatrix2[37]) +
        dot(t6, weightMatrix2[38]) +
        dot(t7, weightMatrix2[39]) +
        dot(t8, weightMatrix2[40]) +
        dot(t9, weightMatrix2[41]) +
        dot(t10, weightMatrix2[42]) +
        dot(t11, weightMatrix2[43]) +
        dot(t12, weightMatrix2[44]) +
        dot(t13, weightMatrix2[45]) +
        dot(t14, weightMatrix2[46]) +
        dot(t15, weightMatrix2[47])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[32]) +
        dot(t1, weightMatrix3[33]) +
        dot(t2, weightMatrix3[34]) +
        dot(t3, weightMatrix3[35]) +
        dot(t4, weightMatrix3[36]) +
        dot(t5, weightMatrix3[37]) +
        dot(t6, weightMatrix3[38]) +
        dot(t7, weightMatrix3[39]) +
        dot(t8, weightMatrix3[40]) +
        dot(t9, weightMatrix3[41]) +
        dot(t10, weightMatrix3[42]) +
        dot(t11, weightMatrix3[43]) +
        dot(t12, weightMatrix3[44]) +
        dot(t13, weightMatrix3[45]) +
        dot(t14, weightMatrix3[46]) +
        dot(t15, weightMatrix3[47])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[32]) +
        dot(t1, weightMatrix4[33]) +
        dot(t2, weightMatrix4[34]) +
        dot(t3, weightMatrix4[35]) +
        dot(t4, weightMatrix4[36]) +
        dot(t5, weightMatrix4[37]) +
        dot(t6, weightMatrix4[38]) +
        dot(t7, weightMatrix4[39]) +
        dot(t8, weightMatrix4[40]) +
        dot(t9, weightMatrix4[41]) +
        dot(t10, weightMatrix4[42]) +
        dot(t11, weightMatrix4[43]) +
        dot(t12, weightMatrix4[44]) +
        dot(t13, weightMatrix4[45]) +
        dot(t14, weightMatrix4[46]) +
        dot(t15, weightMatrix4[47])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 12

    else if (i == 12) {
        s.r += (dot(t0, weightMatrix1[48]) +
        dot(t1, weightMatrix1[49]) +
        dot(t2, weightMatrix1[50]) +
        dot(t3, weightMatrix1[51]) +
        dot(t4, weightMatrix1[52]) +
        dot(t5, weightMatrix1[53]) +
        dot(t6, weightMatrix1[54]) +
        dot(t7, weightMatrix1[55]) +
        dot(t8, weightMatrix1[56]) +
        dot(t9, weightMatrix1[57]) +
        dot(t10, weightMatrix1[58]) +
        dot(t11, weightMatrix1[59]) +
        dot(t12, weightMatrix1[60]) +
        dot(t13, weightMatrix1[61]) +
        dot(t14, weightMatrix1[62]) +
        dot(t15, weightMatrix1[63])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[48]) +
        dot(t1, weightMatrix2[49]) +
        dot(t2, weightMatrix2[50]) +
        dot(t3, weightMatrix2[51]) +
        dot(t4, weightMatrix2[52]) +
        dot(t5, weightMatrix2[53]) +
        dot(t6, weightMatrix2[54]) +
        dot(t7, weightMatrix2[55]) +
        dot(t8, weightMatrix2[56]) +
        dot(t9, weightMatrix2[57]) +
        dot(t10, weightMatrix2[58]) +
        dot(t11, weightMatrix2[59]) +
        dot(t12, weightMatrix2[60]) +
        dot(t13, weightMatrix2[61]) +
        dot(t14, weightMatrix2[62]) +
        dot(t15, weightMatrix2[63])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[48]) +
        dot(t1, weightMatrix3[49]) +
        dot(t2, weightMatrix3[50]) +
        dot(t3, weightMatrix3[51]) +
        dot(t4, weightMatrix3[52]) +
        dot(t5, weightMatrix3[53]) +
        dot(t6, weightMatrix3[54]) +
        dot(t7, weightMatrix3[55]) +
        dot(t8, weightMatrix3[56]) +
        dot(t9, weightMatrix3[57]) +
        dot(t10, weightMatrix3[58]) +
        dot(t11, weightMatrix3[59]) +
        dot(t12, weightMatrix3[60]) +
        dot(t13, weightMatrix3[61]) +
        dot(t14, weightMatrix3[62]) +
        dot(t15, weightMatrix3[63])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[48]) +
        dot(t1, weightMatrix4[49]) +
        dot(t2, weightMatrix4[50]) +
        dot(t3, weightMatrix4[51]) +
        dot(t4, weightMatrix4[52]) +
        dot(t5, weightMatrix4[53]) +
        dot(t6, weightMatrix4[54]) +
        dot(t7, weightMatrix4[55]) +
        dot(t8, weightMatrix4[56]) +
        dot(t9, weightMatrix4[57]) +
        dot(t10, weightMatrix4[58]) +
        dot(t11, weightMatrix4[59]) +
        dot(t12, weightMatrix4[60]) +
        dot(t13, weightMatrix4[61]) +
        dot(t14, weightMatrix4[62]) +
        dot(t15, weightMatrix4[63])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 16

    else if (i == 16) {
        s.r += (dot(t0, weightMatrix1[64]) +
        dot(t1, weightMatrix1[65]) +
        dot(t2, weightMatrix1[66]) +
        dot(t3, weightMatrix1[67]) +
        dot(t4, weightMatrix1[68]) +
        dot(t5, weightMatrix1[69]) +
        dot(t6, weightMatrix1[70]) +
        dot(t7, weightMatrix1[71]) +
        dot(t8, weightMatrix1[72]) +
        dot(t9, weightMatrix1[73]) +
        dot(t10, weightMatrix1[74]) +
        dot(t11, weightMatrix1[75]) +
        dot(t12, weightMatrix1[76]) +
        dot(t13, weightMatrix1[77]) +
        dot(t14, weightMatrix1[78]) +
        dot(t15, weightMatrix1[79])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[64]) +
        dot(t1, weightMatrix2[65]) +
        dot(t2, weightMatrix2[66]) +
        dot(t3, weightMatrix2[67]) +
        dot(t4, weightMatrix2[68]) +
        dot(t5, weightMatrix2[69]) +
        dot(t6, weightMatrix2[70]) +
        dot(t7, weightMatrix2[71]) +
        dot(t8, weightMatrix2[72]) +
        dot(t9, weightMatrix2[73]) +
        dot(t10, weightMatrix2[74]) +
        dot(t11, weightMatrix2[75]) +
        dot(t12, weightMatrix2[76]) +
        dot(t13, weightMatrix2[77]) +
        dot(t14, weightMatrix2[78]) +
        dot(t15, weightMatrix2[79])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[64]) +
        dot(t1, weightMatrix3[65]) +
        dot(t2, weightMatrix3[66]) +
        dot(t3, weightMatrix3[67]) +
        dot(t4, weightMatrix3[68]) +
        dot(t5, weightMatrix3[69]) +
        dot(t6, weightMatrix3[70]) +
        dot(t7, weightMatrix3[71]) +
        dot(t8, weightMatrix3[72]) +
        dot(t9, weightMatrix3[73]) +
        dot(t10, weightMatrix3[74]) +
        dot(t11, weightMatrix3[75]) +
        dot(t12, weightMatrix3[76]) +
        dot(t13, weightMatrix3[77]) +
        dot(t14, weightMatrix3[78]) +
        dot(t15, weightMatrix3[79])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[64]) +
        dot(t1, weightMatrix4[65]) +
        dot(t2, weightMatrix4[66]) +
        dot(t3, weightMatrix4[67]) +
        dot(t4, weightMatrix4[68]) +
        dot(t5, weightMatrix4[69]) +
        dot(t6, weightMatrix4[70]) +
        dot(t7, weightMatrix4[71]) +
        dot(t8, weightMatrix4[72]) +
        dot(t9, weightMatrix4[73]) +
        dot(t10, weightMatrix4[74]) +
        dot(t11, weightMatrix4[75]) +
        dot(t12, weightMatrix4[76]) +
        dot(t13, weightMatrix4[77]) +
        dot(t14, weightMatrix4[78]) +
        dot(t15, weightMatrix4[79])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 20

    else if (i == 20) {
        s.r += (dot(t0, weightMatrix1[80]) +
        dot(t1, weightMatrix1[81]) +
        dot(t2, weightMatrix1[82]) +
        dot(t3, weightMatrix1[83]) +
        dot(t4, weightMatrix1[84]) +
        dot(t5, weightMatrix1[85]) +
        dot(t6, weightMatrix1[86]) +
        dot(t7, weightMatrix1[87]) +
        dot(t8, weightMatrix1[88]) +
        dot(t9, weightMatrix1[89]) +
        dot(t10, weightMatrix1[90]) +
        dot(t11, weightMatrix1[91]) +
        dot(t12, weightMatrix1[92]) +
        dot(t13, weightMatrix1[93]) +
        dot(t14, weightMatrix1[94]) +
        dot(t15, weightMatrix1[95])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[80]) +
        dot(t1, weightMatrix2[81]) +
        dot(t2, weightMatrix2[82]) +
        dot(t3, weightMatrix2[83]) +
        dot(t4, weightMatrix2[84]) +
        dot(t5, weightMatrix2[85]) +
        dot(t6, weightMatrix2[86]) +
        dot(t7, weightMatrix2[87]) +
        dot(t8, weightMatrix2[88]) +
        dot(t9, weightMatrix2[89]) +
        dot(t10, weightMatrix2[90]) +
        dot(t11, weightMatrix2[91]) +
        dot(t12, weightMatrix2[92]) +
        dot(t13, weightMatrix2[93]) +
        dot(t14, weightMatrix2[94]) +
        dot(t15, weightMatrix2[95])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[80]) +
        dot(t1, weightMatrix3[81]) +
        dot(t2, weightMatrix3[82]) +
        dot(t3, weightMatrix3[83]) +
        dot(t4, weightMatrix3[84]) +
        dot(t5, weightMatrix3[85]) +
        dot(t6, weightMatrix3[86]) +
        dot(t7, weightMatrix3[87]) +
        dot(t8, weightMatrix3[88]) +
        dot(t9, weightMatrix3[89]) +
        dot(t10, weightMatrix3[90]) +
        dot(t11, weightMatrix3[91]) +
        dot(t12, weightMatrix3[92]) +
        dot(t13, weightMatrix3[93]) +
        dot(t14, weightMatrix3[94]) +
        dot(t15, weightMatrix3[95])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[80]) +
        dot(t1, weightMatrix4[81]) +
        dot(t2, weightMatrix4[82]) +
        dot(t3, weightMatrix4[83]) +
        dot(t4, weightMatrix4[84]) +
        dot(t5, weightMatrix4[85]) +
        dot(t6, weightMatrix4[86]) +
        dot(t7, weightMatrix4[87]) +
        dot(t8, weightMatrix4[88]) +
        dot(t9, weightMatrix4[89]) +
        dot(t10, weightMatrix4[90]) +
        dot(t11, weightMatrix4[91]) +
        dot(t12, weightMatrix4[92]) +
        dot(t13, weightMatrix4[93]) +
        dot(t14, weightMatrix4[94]) +
        dot(t15, weightMatrix4[95])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 24

    else if (i == 24) {
        s.r += (dot(t0, weightMatrix1[96]) +
        dot(t1, weightMatrix1[97]) +
        dot(t2, weightMatrix1[98]) +
        dot(t3, weightMatrix1[99]) +
        dot(t4, weightMatrix1[100]) +
        dot(t5, weightMatrix1[101]) +
        dot(t6, weightMatrix1[102]) +
        dot(t7, weightMatrix1[103]) +
        dot(t8, weightMatrix1[104]) +
        dot(t9, weightMatrix1[105]) +
        dot(t10, weightMatrix1[106]) +
        dot(t11, weightMatrix1[107]) +
        dot(t12, weightMatrix1[108]) +
        dot(t13, weightMatrix1[109]) +
        dot(t14, weightMatrix1[110]) +
        dot(t15, weightMatrix1[111])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[96]) +
        dot(t1, weightMatrix2[97]) +
        dot(t2, weightMatrix2[98]) +
        dot(t3, weightMatrix2[99]) +
        dot(t4, weightMatrix2[100]) +
        dot(t5, weightMatrix2[101]) +
        dot(t6, weightMatrix2[102]) +
        dot(t7, weightMatrix2[103]) +
        dot(t8, weightMatrix2[104]) +
        dot(t9, weightMatrix2[105]) +
        dot(t10, weightMatrix2[106]) +
        dot(t11, weightMatrix2[107]) +
        dot(t12, weightMatrix2[108]) +
        dot(t13, weightMatrix2[109]) +
        dot(t14, weightMatrix2[110]) +
        dot(t15, weightMatrix2[111])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[96]) +
        dot(t1, weightMatrix3[97]) +
        dot(t2, weightMatrix3[98]) +
        dot(t3, weightMatrix3[99]) +
        dot(t4, weightMatrix3[100]) +
        dot(t5, weightMatrix3[101]) +
        dot(t6, weightMatrix3[102]) +
        dot(t7, weightMatrix3[103]) +
        dot(t8, weightMatrix3[104]) +
        dot(t9, weightMatrix3[105]) +
        dot(t10, weightMatrix3[106]) +
        dot(t11, weightMatrix3[107]) +
        dot(t12, weightMatrix3[108]) +
        dot(t13, weightMatrix3[109]) +
        dot(t14, weightMatrix3[110]) +
        dot(t15, weightMatrix3[111])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[96]) +
        dot(t1, weightMatrix4[97]) +
        dot(t2, weightMatrix4[98]) +
        dot(t3, weightMatrix4[99]) +
        dot(t4, weightMatrix4[100]) +
        dot(t5, weightMatrix4[101]) +
        dot(t6, weightMatrix4[102]) +
        dot(t7, weightMatrix4[103]) +
        dot(t8, weightMatrix4[104]) +
        dot(t9, weightMatrix4[105]) +
        dot(t10, weightMatrix4[106]) +
        dot(t11, weightMatrix4[107]) +
        dot(t12, weightMatrix4[108]) +
        dot(t13, weightMatrix4[109]) +
        dot(t14, weightMatrix4[110]) +
        dot(t15, weightMatrix4[111])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 28

    else if (i == 28) {
        s.r += (dot(t0, weightMatrix1[112]) +
        dot(t1, weightMatrix1[113]) +
        dot(t2, weightMatrix1[114]) +
        dot(t3, weightMatrix1[115]) +
        dot(t4, weightMatrix1[116]) +
        dot(t5, weightMatrix1[117]) +
        dot(t6, weightMatrix1[118]) +
        dot(t7, weightMatrix1[119]) +
        dot(t8, weightMatrix1[120]) +
        dot(t9, weightMatrix1[121]) +
        dot(t10, weightMatrix1[122]) +
        dot(t11, weightMatrix1[123]) +
        dot(t12, weightMatrix1[124]) +
        dot(t13, weightMatrix1[125]) +
        dot(t14, weightMatrix1[126]) +
        dot(t15, weightMatrix1[127])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[112]) +
        dot(t1, weightMatrix2[113]) +
        dot(t2, weightMatrix2[114]) +
        dot(t3, weightMatrix2[115]) +
        dot(t4, weightMatrix2[116]) +
        dot(t5, weightMatrix2[117]) +
        dot(t6, weightMatrix2[118]) +
        dot(t7, weightMatrix2[119]) +
        dot(t8, weightMatrix2[120]) +
        dot(t9, weightMatrix2[121]) +
        dot(t10, weightMatrix2[122]) +
        dot(t11, weightMatrix2[123]) +
        dot(t12, weightMatrix2[124]) +
        dot(t13, weightMatrix2[125]) +
        dot(t14, weightMatrix2[126]) +
        dot(t15, weightMatrix2[127])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[112]) +
        dot(t1, weightMatrix3[113]) +
        dot(t2, weightMatrix3[114]) +
        dot(t3, weightMatrix3[115]) +
        dot(t4, weightMatrix3[116]) +
        dot(t5, weightMatrix3[117]) +
        dot(t6, weightMatrix3[118]) +
        dot(t7, weightMatrix3[119]) +
        dot(t8, weightMatrix3[120]) +
        dot(t9, weightMatrix3[121]) +
        dot(t10, weightMatrix3[122]) +
        dot(t11, weightMatrix3[123]) +
        dot(t12, weightMatrix3[124]) +
        dot(t13, weightMatrix3[125]) +
        dot(t14, weightMatrix3[126]) +
        dot(t15, weightMatrix3[127])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[112]) +
        dot(t1, weightMatrix4[113]) +
        dot(t2, weightMatrix4[114]) +
        dot(t3, weightMatrix4[115]) +
        dot(t4, weightMatrix4[116]) +
        dot(t5, weightMatrix4[117]) +
        dot(t6, weightMatrix4[118]) +
        dot(t7, weightMatrix4[119]) +
        dot(t8, weightMatrix4[120]) +
        dot(t9, weightMatrix4[121]) +
        dot(t10, weightMatrix4[122]) +
        dot(t11, weightMatrix4[123]) +
        dot(t12, weightMatrix4[124]) +
        dot(t13, weightMatrix4[125]) +
        dot(t14, weightMatrix4[126]) +
        dot(t15, weightMatrix4[127])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 32

    else if (i == 32) {
        s.r += (dot(t0, weightMatrix1[128]) +
        dot(t1, weightMatrix1[129]) +
        dot(t2, weightMatrix1[130]) +
        dot(t3, weightMatrix1[131]) +
        dot(t4, weightMatrix1[132]) +
        dot(t5, weightMatrix1[133]) +
        dot(t6, weightMatrix1[134]) +
        dot(t7, weightMatrix1[135]) +
        dot(t8, weightMatrix1[136]) +
        dot(t9, weightMatrix1[137]) +
        dot(t10, weightMatrix1[138]) +
        dot(t11, weightMatrix1[139]) +
        dot(t12, weightMatrix1[140]) +
        dot(t13, weightMatrix1[141]) +
        dot(t14, weightMatrix1[142]) +
        dot(t15, weightMatrix1[143])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[128]) +
        dot(t1, weightMatrix2[129]) +
        dot(t2, weightMatrix2[130]) +
        dot(t3, weightMatrix2[131]) +
        dot(t4, weightMatrix2[132]) +
        dot(t5, weightMatrix2[133]) +
        dot(t6, weightMatrix2[134]) +
        dot(t7, weightMatrix2[135]) +
        dot(t8, weightMatrix2[136]) +
        dot(t9, weightMatrix2[137]) +
        dot(t10, weightMatrix2[138]) +
        dot(t11, weightMatrix2[139]) +
        dot(t12, weightMatrix2[140]) +
        dot(t13, weightMatrix2[141]) +
        dot(t14, weightMatrix2[142]) +
        dot(t15, weightMatrix2[143])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[128]) +
        dot(t1, weightMatrix3[129]) +
        dot(t2, weightMatrix3[130]) +
        dot(t3, weightMatrix3[131]) +
        dot(t4, weightMatrix3[132]) +
        dot(t5, weightMatrix3[133]) +
        dot(t6, weightMatrix3[134]) +
        dot(t7, weightMatrix3[135]) +
        dot(t8, weightMatrix3[136]) +
        dot(t9, weightMatrix3[137]) +
        dot(t10, weightMatrix3[138]) +
        dot(t11, weightMatrix3[139]) +
        dot(t12, weightMatrix3[140]) +
        dot(t13, weightMatrix3[141]) +
        dot(t14, weightMatrix3[142]) +
        dot(t15, weightMatrix3[143])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[128]) +
        dot(t1, weightMatrix4[129]) +
        dot(t2, weightMatrix4[130]) +
        dot(t3, weightMatrix4[131]) +
        dot(t4, weightMatrix4[132]) +
        dot(t5, weightMatrix4[133]) +
        dot(t6, weightMatrix4[134]) +
        dot(t7, weightMatrix4[135]) +
        dot(t8, weightMatrix4[136]) +
        dot(t9, weightMatrix4[137]) +
        dot(t10, weightMatrix4[138]) +
        dot(t11, weightMatrix4[139]) +
        dot(t12, weightMatrix4[140]) +
        dot(t13, weightMatrix4[141]) +
        dot(t14, weightMatrix4[142]) +
        dot(t15, weightMatrix4[143])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 36

    else if (i == 36) {
        s.r += (dot(t0, weightMatrix1[144]) +
        dot(t1, weightMatrix1[145]) +
        dot(t2, weightMatrix1[146]) +
        dot(t3, weightMatrix1[147]) +
        dot(t4, weightMatrix1[148]) +
        dot(t5, weightMatrix1[149]) +
        dot(t6, weightMatrix1[150]) +
        dot(t7, weightMatrix1[151]) +
        dot(t8, weightMatrix1[152]) +
        dot(t9, weightMatrix1[153]) +
        dot(t10, weightMatrix1[154]) +
        dot(t11, weightMatrix1[155]) +
        dot(t12, weightMatrix1[156]) +
        dot(t13, weightMatrix1[157]) +
        dot(t14, weightMatrix1[158]) +
        dot(t15, weightMatrix1[159])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[144]) +
        dot(t1, weightMatrix2[145]) +
        dot(t2, weightMatrix2[146]) +
        dot(t3, weightMatrix2[147]) +
        dot(t4, weightMatrix2[148]) +
        dot(t5, weightMatrix2[149]) +
        dot(t6, weightMatrix2[150]) +
        dot(t7, weightMatrix2[151]) +
        dot(t8, weightMatrix2[152]) +
        dot(t9, weightMatrix2[153]) +
        dot(t10, weightMatrix2[154]) +
        dot(t11, weightMatrix2[155]) +
        dot(t12, weightMatrix2[156]) +
        dot(t13, weightMatrix2[157]) +
        dot(t14, weightMatrix2[158]) +
        dot(t15, weightMatrix2[159])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[144]) +
        dot(t1, weightMatrix3[145]) +
        dot(t2, weightMatrix3[146]) +
        dot(t3, weightMatrix3[147]) +
        dot(t4, weightMatrix3[148]) +
        dot(t5, weightMatrix3[149]) +
        dot(t6, weightMatrix3[150]) +
        dot(t7, weightMatrix3[151]) +
        dot(t8, weightMatrix3[152]) +
        dot(t9, weightMatrix3[153]) +
        dot(t10, weightMatrix3[154]) +
        dot(t11, weightMatrix3[155]) +
        dot(t12, weightMatrix3[156]) +
        dot(t13, weightMatrix3[157]) +
        dot(t14, weightMatrix3[158]) +
        dot(t15, weightMatrix3[159])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[144]) +
        dot(t1, weightMatrix4[145]) +
        dot(t2, weightMatrix4[146]) +
        dot(t3, weightMatrix4[147]) +
        dot(t4, weightMatrix4[148]) +
        dot(t5, weightMatrix4[149]) +
        dot(t6, weightMatrix4[150]) +
        dot(t7, weightMatrix4[151]) +
        dot(t8, weightMatrix4[152]) +
        dot(t9, weightMatrix4[153]) +
        dot(t10, weightMatrix4[154]) +
        dot(t11, weightMatrix4[155]) +
        dot(t12, weightMatrix4[156]) +
        dot(t13, weightMatrix4[157]) +
        dot(t14, weightMatrix4[158]) +
        dot(t15, weightMatrix4[159])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 40

    else if (i == 40) {
        s.r += (dot(t0, weightMatrix1[160]) +
        dot(t1, weightMatrix1[161]) +
        dot(t2, weightMatrix1[162]) +
        dot(t3, weightMatrix1[163]) +
        dot(t4, weightMatrix1[164]) +
        dot(t5, weightMatrix1[165]) +
        dot(t6, weightMatrix1[166]) +
        dot(t7, weightMatrix1[167]) +
        dot(t8, weightMatrix1[168]) +
        dot(t9, weightMatrix1[169]) +
        dot(t10, weightMatrix1[170]) +
        dot(t11, weightMatrix1[171]) +
        dot(t12, weightMatrix1[172]) +
        dot(t13, weightMatrix1[173]) +
        dot(t14, weightMatrix1[174]) +
        dot(t15, weightMatrix1[175])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[160]) +
        dot(t1, weightMatrix2[161]) +
        dot(t2, weightMatrix2[162]) +
        dot(t3, weightMatrix2[163]) +
        dot(t4, weightMatrix2[164]) +
        dot(t5, weightMatrix2[165]) +
        dot(t6, weightMatrix2[166]) +
        dot(t7, weightMatrix2[167]) +
        dot(t8, weightMatrix2[168]) +
        dot(t9, weightMatrix2[169]) +
        dot(t10, weightMatrix2[170]) +
        dot(t11, weightMatrix2[171]) +
        dot(t12, weightMatrix2[172]) +
        dot(t13, weightMatrix2[173]) +
        dot(t14, weightMatrix2[174]) +
        dot(t15, weightMatrix2[175])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[160]) +
        dot(t1, weightMatrix3[161]) +
        dot(t2, weightMatrix3[162]) +
        dot(t3, weightMatrix3[163]) +
        dot(t4, weightMatrix3[164]) +
        dot(t5, weightMatrix3[165]) +
        dot(t6, weightMatrix3[166]) +
        dot(t7, weightMatrix3[167]) +
        dot(t8, weightMatrix3[168]) +
        dot(t9, weightMatrix3[169]) +
        dot(t10, weightMatrix3[170]) +
        dot(t11, weightMatrix3[171]) +
        dot(t12, weightMatrix3[172]) +
        dot(t13, weightMatrix3[173]) +
        dot(t14, weightMatrix3[174]) +
        dot(t15, weightMatrix3[175])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[160]) +
        dot(t1, weightMatrix4[161]) +
        dot(t2, weightMatrix4[162]) +
        dot(t3, weightMatrix4[163]) +
        dot(t4, weightMatrix4[164]) +
        dot(t5, weightMatrix4[165]) +
        dot(t6, weightMatrix4[166]) +
        dot(t7, weightMatrix4[167]) +
        dot(t8, weightMatrix4[168]) +
        dot(t9, weightMatrix4[169]) +
        dot(t10, weightMatrix4[170]) +
        dot(t11, weightMatrix4[171]) +
        dot(t12, weightMatrix4[172]) +
        dot(t13, weightMatrix4[173]) +
        dot(t14, weightMatrix4[174]) +
        dot(t15, weightMatrix4[175])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 44

    else if (i == 44) {
        s.r += (dot(t0, weightMatrix1[176]) +
        dot(t1, weightMatrix1[177]) +
        dot(t2, weightMatrix1[178]) +
        dot(t3, weightMatrix1[179]) +
        dot(t4, weightMatrix1[180]) +
        dot(t5, weightMatrix1[181]) +
        dot(t6, weightMatrix1[182]) +
        dot(t7, weightMatrix1[183]) +
        dot(t8, weightMatrix1[184]) +
        dot(t9, weightMatrix1[185]) +
        dot(t10, weightMatrix1[186]) +
        dot(t11, weightMatrix1[187]) +
        dot(t12, weightMatrix1[188]) +
        dot(t13, weightMatrix1[189]) +
        dot(t14, weightMatrix1[190]) +
        dot(t15, weightMatrix1[191])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[176]) +
        dot(t1, weightMatrix2[177]) +
        dot(t2, weightMatrix2[178]) +
        dot(t3, weightMatrix2[179]) +
        dot(t4, weightMatrix2[180]) +
        dot(t5, weightMatrix2[181]) +
        dot(t6, weightMatrix2[182]) +
        dot(t7, weightMatrix2[183]) +
        dot(t8, weightMatrix2[184]) +
        dot(t9, weightMatrix2[185]) +
        dot(t10, weightMatrix2[186]) +
        dot(t11, weightMatrix2[187]) +
        dot(t12, weightMatrix2[188]) +
        dot(t13, weightMatrix2[189]) +
        dot(t14, weightMatrix2[190]) +
        dot(t15, weightMatrix2[191])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[176]) +
        dot(t1, weightMatrix3[177]) +
        dot(t2, weightMatrix3[178]) +
        dot(t3, weightMatrix3[179]) +
        dot(t4, weightMatrix3[180]) +
        dot(t5, weightMatrix3[181]) +
        dot(t6, weightMatrix3[182]) +
        dot(t7, weightMatrix3[183]) +
        dot(t8, weightMatrix3[184]) +
        dot(t9, weightMatrix3[185]) +
        dot(t10, weightMatrix3[186]) +
        dot(t11, weightMatrix3[187]) +
        dot(t12, weightMatrix3[188]) +
        dot(t13, weightMatrix3[189]) +
        dot(t14, weightMatrix3[190]) +
        dot(t15, weightMatrix3[191])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[176]) +
        dot(t1, weightMatrix4[177]) +
        dot(t2, weightMatrix4[178]) +
        dot(t3, weightMatrix4[179]) +
        dot(t4, weightMatrix4[180]) +
        dot(t5, weightMatrix4[181]) +
        dot(t6, weightMatrix4[182]) +
        dot(t7, weightMatrix4[183]) +
        dot(t8, weightMatrix4[184]) +
        dot(t9, weightMatrix4[185]) +
        dot(t10, weightMatrix4[186]) +
        dot(t11, weightMatrix4[187]) +
        dot(t12, weightMatrix4[188]) +
        dot(t13, weightMatrix4[189]) +
        dot(t14, weightMatrix4[190]) +
        dot(t15, weightMatrix4[191])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 48

    else if (i == 48) {
        s.r += (dot(t0, weightMatrix1[192]) +
        dot(t1, weightMatrix1[193]) +
        dot(t2, weightMatrix1[194]) +
        dot(t3, weightMatrix1[195]) +
        dot(t4, weightMatrix1[196]) +
        dot(t5, weightMatrix1[197]) +
        dot(t6, weightMatrix1[198]) +
        dot(t7, weightMatrix1[199]) +
        dot(t8, weightMatrix1[200]) +
        dot(t9, weightMatrix1[201]) +
        dot(t10, weightMatrix1[202]) +
        dot(t11, weightMatrix1[203]) +
        dot(t12, weightMatrix1[204]) +
        dot(t13, weightMatrix1[205]) +
        dot(t14, weightMatrix1[206]) +
        dot(t15, weightMatrix1[207])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[192]) +
        dot(t1, weightMatrix2[193]) +
        dot(t2, weightMatrix2[194]) +
        dot(t3, weightMatrix2[195]) +
        dot(t4, weightMatrix2[196]) +
        dot(t5, weightMatrix2[197]) +
        dot(t6, weightMatrix2[198]) +
        dot(t7, weightMatrix2[199]) +
        dot(t8, weightMatrix2[200]) +
        dot(t9, weightMatrix2[201]) +
        dot(t10, weightMatrix2[202]) +
        dot(t11, weightMatrix2[203]) +
        dot(t12, weightMatrix2[204]) +
        dot(t13, weightMatrix2[205]) +
        dot(t14, weightMatrix2[206]) +
        dot(t15, weightMatrix2[207])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[192]) +
        dot(t1, weightMatrix3[193]) +
        dot(t2, weightMatrix3[194]) +
        dot(t3, weightMatrix3[195]) +
        dot(t4, weightMatrix3[196]) +
        dot(t5, weightMatrix3[197]) +
        dot(t6, weightMatrix3[198]) +
        dot(t7, weightMatrix3[199]) +
        dot(t8, weightMatrix3[200]) +
        dot(t9, weightMatrix3[201]) +
        dot(t10, weightMatrix3[202]) +
        dot(t11, weightMatrix3[203]) +
        dot(t12, weightMatrix3[204]) +
        dot(t13, weightMatrix3[205]) +
        dot(t14, weightMatrix3[206]) +
        dot(t15, weightMatrix3[207])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[192]) +
        dot(t1, weightMatrix4[193]) +
        dot(t2, weightMatrix4[194]) +
        dot(t3, weightMatrix4[195]) +
        dot(t4, weightMatrix4[196]) +
        dot(t5, weightMatrix4[197]) +
        dot(t6, weightMatrix4[198]) +
        dot(t7, weightMatrix4[199]) +
        dot(t8, weightMatrix4[200]) +
        dot(t9, weightMatrix4[201]) +
        dot(t10, weightMatrix4[202]) +
        dot(t11, weightMatrix4[203]) +
        dot(t12, weightMatrix4[204]) +
        dot(t13, weightMatrix4[205]) +
        dot(t14, weightMatrix4[206]) +
        dot(t15, weightMatrix4[207])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 52

    else if (i == 52) {
        s.r += (dot(t0, weightMatrix1[208]) +
        dot(t1, weightMatrix1[209]) +
        dot(t2, weightMatrix1[210]) +
        dot(t3, weightMatrix1[211]) +
        dot(t4, weightMatrix1[212]) +
        dot(t5, weightMatrix1[213]) +
        dot(t6, weightMatrix1[214]) +
        dot(t7, weightMatrix1[215]) +
        dot(t8, weightMatrix1[216]) +
        dot(t9, weightMatrix1[217]) +
        dot(t10, weightMatrix1[218]) +
        dot(t11, weightMatrix1[219]) +
        dot(t12, weightMatrix1[220]) +
        dot(t13, weightMatrix1[221]) +
        dot(t14, weightMatrix1[222]) +
        dot(t15, weightMatrix1[223])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[208]) +
        dot(t1, weightMatrix2[209]) +
        dot(t2, weightMatrix2[210]) +
        dot(t3, weightMatrix2[211]) +
        dot(t4, weightMatrix2[212]) +
        dot(t5, weightMatrix2[213]) +
        dot(t6, weightMatrix2[214]) +
        dot(t7, weightMatrix2[215]) +
        dot(t8, weightMatrix2[216]) +
        dot(t9, weightMatrix2[217]) +
        dot(t10, weightMatrix2[218]) +
        dot(t11, weightMatrix2[219]) +
        dot(t12, weightMatrix2[220]) +
        dot(t13, weightMatrix2[221]) +
        dot(t14, weightMatrix2[222]) +
        dot(t15, weightMatrix2[223])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[208]) +
        dot(t1, weightMatrix3[209]) +
        dot(t2, weightMatrix3[210]) +
        dot(t3, weightMatrix3[211]) +
        dot(t4, weightMatrix3[212]) +
        dot(t5, weightMatrix3[213]) +
        dot(t6, weightMatrix3[214]) +
        dot(t7, weightMatrix3[215]) +
        dot(t8, weightMatrix3[216]) +
        dot(t9, weightMatrix3[217]) +
        dot(t10, weightMatrix3[218]) +
        dot(t11, weightMatrix3[219]) +
        dot(t12, weightMatrix3[220]) +
        dot(t13, weightMatrix3[221]) +
        dot(t14, weightMatrix3[222]) +
        dot(t15, weightMatrix3[223])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[208]) +
        dot(t1, weightMatrix4[209]) +
        dot(t2, weightMatrix4[210]) +
        dot(t3, weightMatrix4[211]) +
        dot(t4, weightMatrix4[212]) +
        dot(t5, weightMatrix4[213]) +
        dot(t6, weightMatrix4[214]) +
        dot(t7, weightMatrix4[215]) +
        dot(t8, weightMatrix4[216]) +
        dot(t9, weightMatrix4[217]) +
        dot(t10, weightMatrix4[218]) +
        dot(t11, weightMatrix4[219]) +
        dot(t12, weightMatrix4[220]) +
        dot(t13, weightMatrix4[221]) +
        dot(t14, weightMatrix4[222]) +
        dot(t15, weightMatrix4[223])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 56

    else if (i == 56) {
        s.r += (dot(t0, weightMatrix1[224]) +
        dot(t1, weightMatrix1[225]) +
        dot(t2, weightMatrix1[226]) +
        dot(t3, weightMatrix1[227]) +
        dot(t4, weightMatrix1[228]) +
        dot(t5, weightMatrix1[229]) +
        dot(t6, weightMatrix1[230]) +
        dot(t7, weightMatrix1[231]) +
        dot(t8, weightMatrix1[232]) +
        dot(t9, weightMatrix1[233]) +
        dot(t10, weightMatrix1[234]) +
        dot(t11, weightMatrix1[235]) +
        dot(t12, weightMatrix1[236]) +
        dot(t13, weightMatrix1[237]) +
        dot(t14, weightMatrix1[238]) +
        dot(t15, weightMatrix1[239])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[224]) +
        dot(t1, weightMatrix2[225]) +
        dot(t2, weightMatrix2[226]) +
        dot(t3, weightMatrix2[227]) +
        dot(t4, weightMatrix2[228]) +
        dot(t5, weightMatrix2[229]) +
        dot(t6, weightMatrix2[230]) +
        dot(t7, weightMatrix2[231]) +
        dot(t8, weightMatrix2[232]) +
        dot(t9, weightMatrix2[233]) +
        dot(t10, weightMatrix2[234]) +
        dot(t11, weightMatrix2[235]) +
        dot(t12, weightMatrix2[236]) +
        dot(t13, weightMatrix2[237]) +
        dot(t14, weightMatrix2[238]) +
        dot(t15, weightMatrix2[239])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[224]) +
        dot(t1, weightMatrix3[225]) +
        dot(t2, weightMatrix3[226]) +
        dot(t3, weightMatrix3[227]) +
        dot(t4, weightMatrix3[228]) +
        dot(t5, weightMatrix3[229]) +
        dot(t6, weightMatrix3[230]) +
        dot(t7, weightMatrix3[231]) +
        dot(t8, weightMatrix3[232]) +
        dot(t9, weightMatrix3[233]) +
        dot(t10, weightMatrix3[234]) +
        dot(t11, weightMatrix3[235]) +
        dot(t12, weightMatrix3[236]) +
        dot(t13, weightMatrix3[237]) +
        dot(t14, weightMatrix3[238]) +
        dot(t15, weightMatrix3[239])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[224]) +
        dot(t1, weightMatrix4[225]) +
        dot(t2, weightMatrix4[226]) +
        dot(t3, weightMatrix4[227]) +
        dot(t4, weightMatrix4[228]) +
        dot(t5, weightMatrix4[229]) +
        dot(t6, weightMatrix4[230]) +
        dot(t7, weightMatrix4[231]) +
        dot(t8, weightMatrix4[232]) +
        dot(t9, weightMatrix4[233]) +
        dot(t10, weightMatrix4[234]) +
        dot(t11, weightMatrix4[235]) +
        dot(t12, weightMatrix4[236]) +
        dot(t13, weightMatrix4[237]) +
        dot(t14, weightMatrix4[238]) +
        dot(t15, weightMatrix4[239])
        );
        #endif
    }
        #endif

        #if NUM_INPUT_PLANES > 60

    else {
        s.r += (dot(t0, weightMatrix1[240]) +
        dot(t1, weightMatrix1[241]) +
        dot(t2, weightMatrix1[242]) +
        dot(t3, weightMatrix1[243]) +
        dot(t4, weightMatrix1[244]) +
        dot(t5, weightMatrix1[245]) +
        dot(t6, weightMatrix1[246]) +
        dot(t7, weightMatrix1[247]) +
        dot(t8, weightMatrix1[248]) +
        dot(t9, weightMatrix1[249]) +
        dot(t10, weightMatrix1[250]) +
        dot(t11, weightMatrix1[251]) +
        dot(t12, weightMatrix1[252]) +
        dot(t13, weightMatrix1[253]) +
        dot(t14, weightMatrix1[254]) +
        dot(t15, weightMatrix1[255])
        );
        #ifdef USE_COMPONENT_G
        s.g += (dot(t0, weightMatrix2[240]) +
        dot(t1, weightMatrix2[241]) +
        dot(t2, weightMatrix2[242]) +
        dot(t3, weightMatrix2[243]) +
        dot(t4, weightMatrix2[244]) +
        dot(t5, weightMatrix2[245]) +
        dot(t6, weightMatrix2[246]) +
        dot(t7, weightMatrix2[247]) +
        dot(t8, weightMatrix2[248]) +
        dot(t9, weightMatrix2[249]) +
        dot(t10, weightMatrix2[250]) +
        dot(t11, weightMatrix2[251]) +
        dot(t12, weightMatrix2[252]) +
        dot(t13, weightMatrix2[253]) +
        dot(t14, weightMatrix2[254]) +
        dot(t15, weightMatrix2[255])
        );
        #endif
        #ifdef USE_COMPONENT_B
        s.b += (dot(t0, weightMatrix3[240]) +
        dot(t1, weightMatrix3[241]) +
        dot(t2, weightMatrix3[242]) +
        dot(t3, weightMatrix3[243]) +
        dot(t4, weightMatrix3[244]) +
        dot(t5, weightMatrix3[245]) +
        dot(t6, weightMatrix3[246]) +
        dot(t7, weightMatrix3[247]) +
        dot(t8, weightMatrix3[248]) +
        dot(t9, weightMatrix3[249]) +
        dot(t10, weightMatrix3[250]) +
        dot(t11, weightMatrix3[251]) +
        dot(t12, weightMatrix3[252]) +
        dot(t13, weightMatrix3[253]) +
        dot(t14, weightMatrix3[254]) +
        dot(t15, weightMatrix3[255])
        );
        #endif
        #ifdef USE_COMPONENT_A
        s.a += (dot(t0, weightMatrix4[240]) +
        dot(t1, weightMatrix4[241]) +
        dot(t2, weightMatrix4[242]) +
        dot(t3, weightMatrix4[243]) +
        dot(t4, weightMatrix4[244]) +
        dot(t5, weightMatrix4[245]) +
        dot(t6, weightMatrix4[246]) +
        dot(t7, weightMatrix4[247]) +
        dot(t8, weightMatrix4[248]) +
        dot(t9, weightMatrix4[249]) +
        dot(t10, weightMatrix4[250]) +
        dot(t11, weightMatrix4[251]) +
        dot(t12, weightMatrix4[252]) +
        dot(t13, weightMatrix4[253]) +
        dot(t14, weightMatrix4[254]) +
        dot(t15, weightMatrix4[255])
        );
        #endif
    }
        #endif


}

// Leaky ReLU Process
s += bias;

#ifdef USE_BATCH_NORMALIZATION
s = ((gamma / sqrt(movingVariance + vec4(0.001f))) * (s - movingMean)) + beta;
#endif
//}


