#define FLOAT_PRECISION mediump

precision FLOAT_PRECISION int;
precision FLOAT_PRECISION float;
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
const mediump vec4 movingMean = _PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_ ;
const mediump vec4 movingVariance = _PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_;
#endif

int checkTexel(highp vec2 coord, highp vec2 maxUV) {
	float stride = float(NUM_STRIDE);
	if (coord.x < 0.0f || coord.x > (maxUV.x + (maxUV.x - 1.0f) * (stride - 1.0f)) ||
		coord.y < 0.0f || coord.y > (maxUV.y + (maxUV.y - 1.0f) * (stride - 1.0f)) ||
		!(int((coord.x - 0.5)) % NUM_STRIDE != 0 && int((coord.y - 0.5)) % NUM_STRIDE == 0)) {
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
	const highp vec2 maxUV = vec2(INPUT_WIDTH, INPUT_HEIGHT);
	const vec2 maxUV_stride = float(NUM_STRIDE) * vec2(maxUV);
	FLOAT_PRECISION vec4 s = vec4(0.0f);

	// Pre-calculate coordinates here, note that
	// textureGather: pixel should be in the center of the four surrounding pixels, use zero offset
	// texture: Pixel should be in the center of the pixel since GL_NEAREST, use 0.5 offset
	// It's also verified that using texture varying and have vertex shader generate those has same performance.
	highp ivec2 baseCoord = ivec2(gl_GlobalInvocationID.xy) + ivec2(0.0f, -1.0f);
	highp vec2 texCoord_1 = (vec2(baseCoord) + vec2(-0.5, -0.5));
	highp vec2 texCoord_2 = (vec2(baseCoord) + vec2(0.5, -0.5));
	highp vec2 texCoord_3 = (vec2(baseCoord) + vec2(1.5, -0.5));

	highp vec2 texCoord_4 = (vec2(baseCoord) + vec2(-0.5, 0.5));
	highp vec2 texCoord_5 = (vec2(baseCoord) + vec2(0.5, 0.5));
	highp vec2 texCoord_6 = (vec2(baseCoord) + vec2(1.5, 0.5));

	highp vec2 texCoord_7 = (vec2(baseCoord) + vec2(-0.5, 1.5));
	highp vec2 texCoord_8 = (vec2(baseCoord) + vec2(0.5, 1.5));
	highp vec2 texCoord_9 = (vec2(baseCoord) + vec2(1.5, 1.5));

	for (int i = 0; i < NUM_INPUT_PLANES; i += 4) {
		FLOAT_PRECISION vec4 t0, t1, t2, t3, t4, t5, t6, t7, t8;
		int layer = i >> 2;
		t0 = checkTexel(texCoord_1, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_1) / maxUV_stride, layer)) : vec4(0.0f);
		t1 = checkTexel(texCoord_2, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_2) / maxUV_stride, layer)) : vec4(0.0f);
		t2 = checkTexel(texCoord_3, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_3) / maxUV_stride, layer)) : vec4(0.0f);
		t3 = checkTexel(texCoord_4, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_4) / maxUV_stride, layer)) : vec4(0.0f);
		t4 = checkTexel(texCoord_5, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_5) / maxUV_stride, layer)) : vec4(0.0f);
		t5 = checkTexel(texCoord_6, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_6) / maxUV_stride, layer)) : vec4(0.0f);
		t6 = checkTexel(texCoord_7, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_7) / maxUV_stride, layer)) : vec4(0.0f);
		t7 = checkTexel(texCoord_8, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_8) / maxUV_stride, layer)) : vec4(0.0f);
		t8 = checkTexel(texCoord_9, maxUV) == 1 ? TEXTURE(inputTextures, vec3((texCoord_9) / maxUV_stride, layer)) : vec4(0.0f);

		// Offset needs to be constant to get the perf gain
		if (i == 0) {
			int index = NUM_KERNEL_SIZE * NUM_KERNEL_SIZE * layer;
			// Offset needs to be constant to get the perf gain
			s.r += (dot(t0, weightMatrix1[0]) +
					dot(t1, weightMatrix1[1]) +
					dot(t2, weightMatrix1[2]) +
					dot(t3, weightMatrix1[3]) +
					dot(t4, weightMatrix1[4]) +
					dot(t5, weightMatrix1[5]) +
					dot(t6, weightMatrix1[6]) +
					dot(t7, weightMatrix1[7]) +
					dot(t8, weightMatrix1[8]));

#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[0]) +
			dot(t1, weightMatrix2[1]) +
			dot(t2, weightMatrix2[2]) +
			dot(t3, weightMatrix2[3]) +
			dot(t4, weightMatrix2[4]) +
			dot(t5, weightMatrix2[5]) +
			dot(t6, weightMatrix2[6]) +
			dot(t7, weightMatrix2[7]) +
			dot(t8, weightMatrix2[8]));
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
			dot(t8, weightMatrix3[8]));
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
			dot(t8, weightMatrix4[8]));
#endif
		}
#if NUM_INPUT_PLANES > 4
		else if (i == 4) {
			int index = NUM_KERNEL_SIZE * NUM_KERNEL_SIZE * layer;
			// Offset needs to be constant to get the perf gain
			s.r += (dot(t0, weightMatrix1[9]) +
			dot(t1, weightMatrix1[10]) +
			dot(t2, weightMatrix1[11]) +
			dot(t3, weightMatrix1[12]) +
			dot(t4, weightMatrix1[13]) +
			dot(t5, weightMatrix1[14]) +
			dot(t6, weightMatrix1[15]) +
			dot(t7, weightMatrix1[16]) +
			dot(t8, weightMatrix1[17]));
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[9]) +
			dot(t1, weightMatrix2[10]) +
			dot(t2, weightMatrix2[11]) +
			dot(t3, weightMatrix2[12]) +
			dot(t4, weightMatrix2[13]) +
			dot(t5, weightMatrix2[14]) +
			dot(t6, weightMatrix2[15]) +
			dot(t7, weightMatrix2[16]) +
			dot(t8, weightMatrix2[17]));
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[9]) +
			dot(t1, weightMatrix3[10]) +
			dot(t2, weightMatrix3[11]) +
			dot(t3, weightMatrix3[12]) +
			dot(t4, weightMatrix3[13]) +
			dot(t5, weightMatrix3[14]) +
			dot(t6, weightMatrix3[15]) +
			dot(t7, weightMatrix3[16]) +
			dot(t8, weightMatrix3[17]));
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[9]) +
			dot(t1, weightMatrix4[10]) +
			dot(t2, weightMatrix4[11]) +
			dot(t3, weightMatrix4[12]) +
			dot(t4, weightMatrix4[13]) +
			dot(t5, weightMatrix4[14]) +
			dot(t6, weightMatrix4[15]) +
			dot(t7, weightMatrix4[16]) +
			dot(t8, weightMatrix4[17]));
			#endif

		}
#endif
#if NUM_INPUT_PLANES > 8
		else if (i == 8) {
			int index = NUM_KERNEL_SIZE * NUM_KERNEL_SIZE * layer;
			// Offset needs to be constant to get the perf gain
			s.r += (dot(t0, weightMatrix1[18]) +
			dot(t1, weightMatrix1[19]) +
			dot(t2, weightMatrix1[20]) +
			dot(t3, weightMatrix1[21]) +
			dot(t4, weightMatrix1[22]) +
			dot(t5, weightMatrix1[23]) +
			dot(t6, weightMatrix1[24]) +
			dot(t7, weightMatrix1[25]) +
			dot(t8, weightMatrix1[26]));
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[18]) +
			dot(t1, weightMatrix2[19]) +
			dot(t2, weightMatrix2[20]) +
			dot(t3, weightMatrix2[21]) +
			dot(t4, weightMatrix2[22]) +
			dot(t5, weightMatrix2[23]) +
			dot(t6, weightMatrix2[24]) +
			dot(t7, weightMatrix2[25]) +
			dot(t8, weightMatrix2[26]));
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[18]) +
			dot(t1, weightMatrix3[19]) +
			dot(t2, weightMatrix3[20]) +
			dot(t3, weightMatrix3[21]) +
			dot(t4, weightMatrix3[22]) +
			dot(t5, weightMatrix3[23]) +
			dot(t6, weightMatrix3[24]) +
			dot(t7, weightMatrix3[25]) +
			dot(t8, weightMatrix3[26]));
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[18]) +
			dot(t1, weightMatrix4[19]) +
			dot(t2, weightMatrix4[20]) +
			dot(t3, weightMatrix4[21]) +
			dot(t4, weightMatrix4[22]) +
			dot(t5, weightMatrix4[23]) +
			dot(t6, weightMatrix4[24]) +
			dot(t7, weightMatrix4[25]) +
			dot(t8, weightMatrix4[26]));
			#endif

		}
#endif
#if NUM_INPUT_PLANES > 12
		else if (i == 12) {
			int index = NUM_KERNEL_SIZE * NUM_KERNEL_SIZE * layer;
			// Offset needs to be constant to get the perf gain
			s.r += (dot(t0, weightMatrix1[27]) +
			dot(t1, weightMatrix1[28]) +
			dot(t2, weightMatrix1[29]) +
			dot(t3, weightMatrix1[30]) +
			dot(t4, weightMatrix1[31]) +
			dot(t5, weightMatrix1[32]) +
			dot(t6, weightMatrix1[33]) +
			dot(t7, weightMatrix1[34]) +
			dot(t8, weightMatrix1[35]));
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[27]) +
			dot(t1, weightMatrix2[28]) +
			dot(t2, weightMatrix2[29]) +
			dot(t3, weightMatrix2[30]) +
			dot(t4, weightMatrix2[31]) +
			dot(t5, weightMatrix2[32]) +
			dot(t6, weightMatrix2[33]) +
			dot(t7, weightMatrix2[34]) +
			dot(t8, weightMatrix2[35]));
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[27]) +
			dot(t1, weightMatrix3[28]) +
			dot(t2, weightMatrix3[29]) +
			dot(t3, weightMatrix3[30]) +
			dot(t4, weightMatrix3[31]) +
			dot(t5, weightMatrix3[32]) +
			dot(t6, weightMatrix3[33]) +
			dot(t7, weightMatrix3[34]) +
			dot(t8, weightMatrix3[35]));
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[27]) +
			dot(t1, weightMatrix4[28]) +
			dot(t2, weightMatrix4[29]) +
			dot(t3, weightMatrix4[30]) +
			dot(t4, weightMatrix4[31]) +
			dot(t5, weightMatrix4[32]) +
			dot(t6, weightMatrix4[33]) +
			dot(t7, weightMatrix4[34]) +
			dot(t8, weightMatrix4[35]));
			#endif

		}
#endif
#if NUM_INPUT_PLANES > 16
		else if (i == 16) {
			int index = NUM_KERNEL_SIZE * NUM_KERNEL_SIZE * layer;
			// Offset needs to be constant to get the perf gain
			s.r += (dot(t0, weightMatrix1[36]) +
					dot(t1, weightMatrix1[37]) +
					dot(t2, weightMatrix1[38]) +
					dot(t3, weightMatrix1[39]) +
					dot(t4, weightMatrix1[40]) +
					dot(t5, weightMatrix1[41]) +
					dot(t6, weightMatrix1[42]) +
					dot(t7, weightMatrix1[43]) +
					dot(t8, weightMatrix1[44]));

#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[36]) +
			dot(t1, weightMatrix2[37]) +
			dot(t2, weightMatrix2[38]) +
			dot(t3, weightMatrix2[39]) +
			dot(t4, weightMatrix2[40]) +
			dot(t5, weightMatrix2[41]) +
			dot(t6, weightMatrix2[42]) +
			dot(t7, weightMatrix2[43]) +
			dot(t8, weightMatrix2[44]));
#endif

#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[36]) +
			dot(t1, weightMatrix3[37]) +
			dot(t2, weightMatrix3[38]) +
			dot(t3, weightMatrix3[39]) +
			dot(t4, weightMatrix3[40]) +
			dot(t5, weightMatrix3[41]) +
			dot(t6, weightMatrix3[42]) +
			dot(t7, weightMatrix3[43]) +
			dot(t8, weightMatrix3[44]));
#endif

#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[36]) +
			dot(t1, weightMatrix4[37]) +
			dot(t2, weightMatrix4[38]) +
			dot(t3, weightMatrix4[39]) +
			dot(t4, weightMatrix4[40]) +
			dot(t5, weightMatrix4[41]) +
			dot(t6, weightMatrix4[42]) +
			dot(t7, weightMatrix4[43]) +
			dot(t8, weightMatrix4[44]));
#endif
		}
#endif
#if NUM_INPUT_PLANES > 20
		else if (i == 20) {
			int index = NUM_KERNEL_SIZE * NUM_KERNEL_SIZE * layer;
			// Offset needs to be constant to get the perf gain
			s.r += (dot(t0, weightMatrix1[45]) +
					dot(t1, weightMatrix1[46]) +
					dot(t2, weightMatrix1[47]) +
					dot(t3, weightMatrix1[48]) +
					dot(t4, weightMatrix1[49]) +
					dot(t5, weightMatrix1[50]) +
					dot(t6, weightMatrix1[51]) +
					dot(t7, weightMatrix1[52]) +
					dot(t8, weightMatrix1[53]));

#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[45]) +
			dot(t1, weightMatrix2[46]) +
			dot(t2, weightMatrix2[47]) +
			dot(t3, weightMatrix2[48]) +
			dot(t4, weightMatrix2[49]) +
			dot(t5, weightMatrix2[50]) +
			dot(t6, weightMatrix2[51]) +
			dot(t7, weightMatrix2[52]) +
			dot(t8, weightMatrix2[53]));
#endif

#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[45]) +
			dot(t1, weightMatrix3[46]) +
			dot(t2, weightMatrix3[47]) +
			dot(t3, weightMatrix3[48]) +
			dot(t4, weightMatrix3[49]) +
			dot(t5, weightMatrix3[50]) +
			dot(t6, weightMatrix3[51]) +
			dot(t7, weightMatrix3[52]) +
			dot(t8, weightMatrix3[53]));
#endif

#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[45]) +
			dot(t1, weightMatrix4[46]) +
			dot(t2, weightMatrix4[47]) +
			dot(t3, weightMatrix4[48]) +
			dot(t4, weightMatrix4[49]) +
			dot(t5, weightMatrix4[50]) +
			dot(t6, weightMatrix4[51]) +
			dot(t7, weightMatrix4[52]) +
			dot(t8, weightMatrix4[53]));
#endif
		}
#endif
#if NUM_INPUT_PLANES > 24
		else if (i == 24) {
			int index = NUM_KERNEL_SIZE * NUM_KERNEL_SIZE * layer;
			// Offset needs to be constant to get the perf gain
			s.r += (dot(t0, weightMatrix1[54]) +
					dot(t1, weightMatrix1[55]) +
					dot(t2, weightMatrix1[56]) +
					dot(t3, weightMatrix1[57]) +
					dot(t4, weightMatrix1[58]) +
					dot(t5, weightMatrix1[59]) +
					dot(t6, weightMatrix1[60]) +
					dot(t7, weightMatrix1[61]) +
					dot(t8, weightMatrix1[62]));

#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[54]) +
			dot(t1, weightMatrix2[55]) +
			dot(t2, weightMatrix2[56]) +
			dot(t3, weightMatrix2[57]) +
			dot(t4, weightMatrix2[58]) +
			dot(t5, weightMatrix2[59]) +
			dot(t6, weightMatrix2[60]) +
			dot(t7, weightMatrix2[61]) +
			dot(t8, weightMatrix2[62]));
#endif

#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[54]) +
			dot(t1, weightMatrix3[55]) +
			dot(t2, weightMatrix3[56]) +
			dot(t3, weightMatrix3[57]) +
			dot(t4, weightMatrix3[58]) +
			dot(t5, weightMatrix3[59]) +
			dot(t6, weightMatrix3[60]) +
			dot(t7, weightMatrix3[61]) +
			dot(t8, weightMatrix3[62]));
#endif

#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[54]) +
			dot(t1, weightMatrix4[55]) +
			dot(t2, weightMatrix4[56]) +
			dot(t3, weightMatrix4[57]) +
			dot(t4, weightMatrix4[58]) +
			dot(t5, weightMatrix4[59]) +
			dot(t6, weightMatrix4[60]) +
			dot(t7, weightMatrix4[61]) +
			dot(t8, weightMatrix4[62]));
#endif
		}
#endif
#if NUM_INPUT_PLANES > 28
		else if (i == 28) {
			int index = NUM_KERNEL_SIZE * NUM_KERNEL_SIZE * layer;
			// Offset needs to be constant to get the perf gain
			s.r += (dot(t0, weightMatrix1[63]) +
					dot(t1, weightMatrix1[64]) +
					dot(t2, weightMatrix1[65]) +
					dot(t3, weightMatrix1[66]) +
					dot(t4, weightMatrix1[67]) +
					dot(t5, weightMatrix1[68]) +
					dot(t6, weightMatrix1[69]) +
					dot(t7, weightMatrix1[70]) +
					dot(t8, weightMatrix1[71]));

#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[63]) +
			dot(t1, weightMatrix2[64]) +
			dot(t2, weightMatrix2[65]) +
			dot(t3, weightMatrix2[66]) +
			dot(t4, weightMatrix2[67]) +
			dot(t5, weightMatrix2[68]) +
			dot(t6, weightMatrix2[69]) +
			dot(t7, weightMatrix2[70]) +
			dot(t8, weightMatrix2[71]));
#endif

#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[63]) +
			dot(t1, weightMatrix3[64]) +
			dot(t2, weightMatrix3[65]) +
			dot(t3, weightMatrix3[66]) +
			dot(t4, weightMatrix3[67]) +
			dot(t5, weightMatrix3[68]) +
			dot(t6, weightMatrix3[69]) +
			dot(t7, weightMatrix3[70]) +
			dot(t8, weightMatrix3[71]));
#endif

#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[63]) +
			dot(t1, weightMatrix4[64]) +
			dot(t2, weightMatrix4[65]) +
			dot(t3, weightMatrix4[66]) +
			dot(t4, weightMatrix4[67]) +
			dot(t5, weightMatrix4[68]) +
			dot(t6, weightMatrix4[69]) +
			dot(t7, weightMatrix4[70]) +
			dot(t8, weightMatrix4[71]));
#endif
		}
#endif
	}

	// Leaky ReLU Process
	s += bias;

#ifdef USE_BATCH_NORMALIZATION
    s = ((gamma/sqrt(movingVariance + vec4(0.001f))) * (s - movingMean) ) + beta;
#endif
//}

