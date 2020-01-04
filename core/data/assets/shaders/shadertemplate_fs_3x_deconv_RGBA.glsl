#define FLOAT_PRECISION _PLACEHOLDER_PRECISION_

precision FLOAT_PRECISION float;
precision FLOAT_PRECISION sampler2DArray;

layout(location = 0)out vec4 o_pixel;

uniform FLOAT_PRECISION vec4 bias;
uniform FLOAT_PRECISION uint kernelSize;
uniform FLOAT_PRECISION sampler2DArray inputTextures;

//These should be defined inside the shader loader code
//#define INPUT_WIDTH 540.0
//#define INPUT_HEIGHT 960.0

//#define USE_COMPONENT_G
//#define USE_COMPONENT_B
//#define USE_COMPONENT_A

const FLOAT_PRECISION vec4 weightMatrix1[] = vec4[](
_PLACEHOLDER_WEIGHT1_VEC_CONSTANTS_);

#ifdef USE_COMPONENT_G
const FLOAT_PRECISION vec4 weightMatrix2[] = vec4[](
	_PLACEHOLDER_WEIGHT2_VEC_CONSTANTS_);
#endif

#ifdef USE_COMPONENT_B
const FLOAT_PRECISION vec4 weightMatrix3[] = vec4[](
	_PLACEHOLDER_WEIGHT3_VEC_CONSTANTS_);
#endif

#ifdef USE_COMPONENT_A
const FLOAT_PRECISION vec4 weightMatrix4[] = vec4[](
	_PLACEHOLDER_WEIGHT4_VEC_CONSTANTS_);
#endif

#ifdef USE_BATCH_NORMALIZATION
const FLOAT_PRECISION vec4 beta = _PLACEHOLDER_BETA_VEC_CONSTANTS_;
const FLOAT_PRECISION vec4 gamma = _PLACEHOLDER_GAMMA_VEC_CONSTANTS_;
const FLOAT_PRECISION vec4 movingMean = _PLACEHOLDER_MOVINGMEAN_VEC_CONSTANTS_ ;
const FLOAT_PRECISION vec4 movingVariance = _PLACEHOLDER_MOVINGVARIANCE_VEC_CONSTANTS_;
#endif

int checkTexel(vec2 coord, vec2 maxUV, float padSize) {
	float stride = float(NUM_STRIDE);
	bool xCheck = ((coord.x - padSize) < 0.0f || (coord.x - padSize) > (maxUV.x + (maxUV.x - 1.0f) * (stride - 1.0f)));
	bool yCheck = ((coord.y - padSize) < 0.0f || (coord.y - padSize) > (maxUV.y + (maxUV.y - 1.0f) * (stride - 1.0f)));
	bool padCheck = !((int((coord.x - padSize - 0.5)) % NUM_STRIDE != 0 && int((coord.y - padSize - 0.5)) % NUM_STRIDE != 0));
	return (xCheck || yCheck || padCheck) ? 0 : 1;
}

void main()
{
	// Padding Process
	int k = int(kernelSize);

	// p is for padding. will for future use when having padding
	// int padding_size = int(k - 0 - 1);

	float totalPaddingSize = 0.0f; //  float(int(k / 2)); is for valid padding
	vec2 paddingShift = vec2(totalPaddingSize, totalPaddingSize);

	// Convolution Process
	const int lod = 0;
	FLOAT_PRECISION vec2 maxUV = vec2(INPUT_WIDTH, INPUT_HEIGHT);
	//FLOAT_PRECISION ivec2 maxUV = ivec2(INPUT_WIDTH, INPUT_HEIGHT);//textureSize(inputTextures[0], lod);
	FLOAT_PRECISION vec4 s = vec4(0.0f);

	// Pre-calculate coordinates here, note that
	// textureGather: pixel should be in the center of the four surrounding pixels, use zero offset
	// texture: Pixel should be in the center of the pixel since GL_NEAREST, use 0.5 offset
	// It's also verified that using texture varying and have vertex shader generate those has same performance.
	ivec2 baseCoord = ivec2(gl_FragCoord.xy) + ivec2(0, -1);
	vec2 texCoord_1 = (vec2(baseCoord) + vec2(-0.5, -0.5));
	vec2 texCoord_2 = (vec2(baseCoord) + vec2(0.5, -0.5));
	vec2 texCoord_3 = (vec2(baseCoord) + vec2(1.5, -0.5));

	vec2 texCoord_4 = (vec2(baseCoord) + vec2(-0.5, 0.5));
	vec2 texCoord_5 = (vec2(baseCoord) + vec2(0.5, 0.5));
	vec2 texCoord_6 = (vec2(baseCoord) + vec2(1.5, 0.5));

	vec2 texCoord_7 = (vec2(baseCoord) + vec2(-0.5, 1.5));
	vec2 texCoord_8 = (vec2(baseCoord) + vec2(0.5, 1.5));
	vec2 texCoord_9 = (vec2(baseCoord) + vec2(1.5, 1.5));

	for (int i = 0; i < NUM_INPUT_PLANES; i += 4) {
		FLOAT_PRECISION vec4 t0, t1, t2, t3, t4, t5, t6, t7, t8;
		int layer = i >> 2;

		t0 = checkTexel(texCoord_1, maxUV, totalPaddingSize) == 1 ? texture(inputTextures, vec3((texCoord_1 - paddingShift) / vec2(NUM_STRIDE, NUM_STRIDE) / vec2(maxUV), layer)) : vec4(0.0f);
		t1 = checkTexel(texCoord_2, maxUV, totalPaddingSize) == 1 ? texture(inputTextures, vec3((texCoord_2 - paddingShift) / vec2(NUM_STRIDE, NUM_STRIDE) / vec2(maxUV), layer)) : vec4(0.0f);
		t2 = checkTexel(texCoord_3, maxUV, totalPaddingSize) == 1 ? texture(inputTextures, vec3((texCoord_3 - paddingShift) / vec2(NUM_STRIDE, NUM_STRIDE) / vec2(maxUV), layer)) : vec4(0.0f);
		t3 = checkTexel(texCoord_4, maxUV, totalPaddingSize) == 1 ? texture(inputTextures, vec3((texCoord_4 - paddingShift) / vec2(NUM_STRIDE, NUM_STRIDE) / vec2(maxUV), layer)) : vec4(0.0f);
		t4 = checkTexel(texCoord_5, maxUV, totalPaddingSize) == 1 ? texture(inputTextures, vec3((texCoord_5 - paddingShift) / vec2(NUM_STRIDE, NUM_STRIDE) / vec2(maxUV), layer)) : vec4(0.0f);
		t5 = checkTexel(texCoord_6, maxUV, totalPaddingSize) == 1 ? texture(inputTextures, vec3((texCoord_6 - paddingShift) / vec2(NUM_STRIDE, NUM_STRIDE) / vec2(maxUV), layer)) : vec4(0.0f);
		t6 = checkTexel(texCoord_7, maxUV, totalPaddingSize) == 1 ? texture(inputTextures, vec3((texCoord_7 - paddingShift) / vec2(NUM_STRIDE, NUM_STRIDE) / vec2(maxUV), layer)) : vec4(0.0f);
		t7 = checkTexel(texCoord_8, maxUV, totalPaddingSize) == 1 ? texture(inputTextures, vec3((texCoord_8 - paddingShift) / vec2(NUM_STRIDE, NUM_STRIDE) / vec2(maxUV), layer)) : vec4(0.0f);
		t8 = checkTexel(texCoord_9, maxUV, totalPaddingSize) == 1 ? texture(inputTextures, vec3((texCoord_9 - paddingShift) / vec2(NUM_STRIDE, NUM_STRIDE) / vec2(maxUV), layer)) : vec4(0.0f);

		// Offset needs to be constant to get the perf gain
		if (i == 0) {

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

			#if NUM_INPUT_PLANES > 32

		else if (i == 32) {
			s.r += (dot(t0, weightMatrix1[72]) +
			dot(t1, weightMatrix1[73]) +
			dot(t2, weightMatrix1[74]) +
			dot(t3, weightMatrix1[75]) +
			dot(t4, weightMatrix1[76]) +
			dot(t5, weightMatrix1[77]) +
			dot(t6, weightMatrix1[78]) +
			dot(t7, weightMatrix1[79]) +
			dot(t8, weightMatrix1[80])
			);
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[72]) +
			dot(t1, weightMatrix2[73]) +
			dot(t2, weightMatrix2[74]) +
			dot(t3, weightMatrix2[75]) +
			dot(t4, weightMatrix2[76]) +
			dot(t5, weightMatrix2[77]) +
			dot(t6, weightMatrix2[78]) +
			dot(t7, weightMatrix2[79]) +
			dot(t8, weightMatrix2[80])
			);
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[72]) +
			dot(t1, weightMatrix3[73]) +
			dot(t2, weightMatrix3[74]) +
			dot(t3, weightMatrix3[75]) +
			dot(t4, weightMatrix3[76]) +
			dot(t5, weightMatrix3[77]) +
			dot(t6, weightMatrix3[78]) +
			dot(t7, weightMatrix3[79]) +
			dot(t8, weightMatrix3[80])
			);
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[72]) +
			dot(t1, weightMatrix4[73]) +
			dot(t2, weightMatrix4[74]) +
			dot(t3, weightMatrix4[75]) +
			dot(t4, weightMatrix4[76]) +
			dot(t5, weightMatrix4[77]) +
			dot(t6, weightMatrix4[78]) +
			dot(t7, weightMatrix4[79]) +
			dot(t8, weightMatrix4[80])
			);
			#endif
		}
			#endif

			#if NUM_INPUT_PLANES > 36

		else if (i == 36) {
			s.r += (dot(t0, weightMatrix1[81]) +
			dot(t1, weightMatrix1[82]) +
			dot(t2, weightMatrix1[83]) +
			dot(t3, weightMatrix1[84]) +
			dot(t4, weightMatrix1[85]) +
			dot(t5, weightMatrix1[86]) +
			dot(t6, weightMatrix1[87]) +
			dot(t7, weightMatrix1[88]) +
			dot(t8, weightMatrix1[89])
			);
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[81]) +
			dot(t1, weightMatrix2[82]) +
			dot(t2, weightMatrix2[83]) +
			dot(t3, weightMatrix2[84]) +
			dot(t4, weightMatrix2[85]) +
			dot(t5, weightMatrix2[86]) +
			dot(t6, weightMatrix2[87]) +
			dot(t7, weightMatrix2[88]) +
			dot(t8, weightMatrix2[89])
			);
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[81]) +
			dot(t1, weightMatrix3[82]) +
			dot(t2, weightMatrix3[83]) +
			dot(t3, weightMatrix3[84]) +
			dot(t4, weightMatrix3[85]) +
			dot(t5, weightMatrix3[86]) +
			dot(t6, weightMatrix3[87]) +
			dot(t7, weightMatrix3[88]) +
			dot(t8, weightMatrix3[89])
			);
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[81]) +
			dot(t1, weightMatrix4[82]) +
			dot(t2, weightMatrix4[83]) +
			dot(t3, weightMatrix4[84]) +
			dot(t4, weightMatrix4[85]) +
			dot(t5, weightMatrix4[86]) +
			dot(t6, weightMatrix4[87]) +
			dot(t7, weightMatrix4[88]) +
			dot(t8, weightMatrix4[89])
			);
			#endif
		}
			#endif

			#if NUM_INPUT_PLANES > 40

		else if (i == 40) {
			s.r += (dot(t0, weightMatrix1[90]) +
			dot(t1, weightMatrix1[91]) +
			dot(t2, weightMatrix1[92]) +
			dot(t3, weightMatrix1[93]) +
			dot(t4, weightMatrix1[94]) +
			dot(t5, weightMatrix1[95]) +
			dot(t6, weightMatrix1[96]) +
			dot(t7, weightMatrix1[97]) +
			dot(t8, weightMatrix1[98])
			);
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[90]) +
			dot(t1, weightMatrix2[91]) +
			dot(t2, weightMatrix2[92]) +
			dot(t3, weightMatrix2[93]) +
			dot(t4, weightMatrix2[94]) +
			dot(t5, weightMatrix2[95]) +
			dot(t6, weightMatrix2[96]) +
			dot(t7, weightMatrix2[97]) +
			dot(t8, weightMatrix2[98])
			);
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[90]) +
			dot(t1, weightMatrix3[91]) +
			dot(t2, weightMatrix3[92]) +
			dot(t3, weightMatrix3[93]) +
			dot(t4, weightMatrix3[94]) +
			dot(t5, weightMatrix3[95]) +
			dot(t6, weightMatrix3[96]) +
			dot(t7, weightMatrix3[97]) +
			dot(t8, weightMatrix3[98])
			);
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[90]) +
			dot(t1, weightMatrix4[91]) +
			dot(t2, weightMatrix4[92]) +
			dot(t3, weightMatrix4[93]) +
			dot(t4, weightMatrix4[94]) +
			dot(t5, weightMatrix4[95]) +
			dot(t6, weightMatrix4[96]) +
			dot(t7, weightMatrix4[97]) +
			dot(t8, weightMatrix4[98])
			);
			#endif
		}
			#endif

			#if NUM_INPUT_PLANES > 44

		else if (i == 44) {
			s.r += (dot(t0, weightMatrix1[99]) +
			dot(t1, weightMatrix1[100]) +
			dot(t2, weightMatrix1[101]) +
			dot(t3, weightMatrix1[102]) +
			dot(t4, weightMatrix1[103]) +
			dot(t5, weightMatrix1[104]) +
			dot(t6, weightMatrix1[105]) +
			dot(t7, weightMatrix1[106]) +
			dot(t8, weightMatrix1[107])
			);
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[99]) +
			dot(t1, weightMatrix2[100]) +
			dot(t2, weightMatrix2[101]) +
			dot(t3, weightMatrix2[102]) +
			dot(t4, weightMatrix2[103]) +
			dot(t5, weightMatrix2[104]) +
			dot(t6, weightMatrix2[105]) +
			dot(t7, weightMatrix2[106]) +
			dot(t8, weightMatrix2[107])
			);
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[99]) +
			dot(t1, weightMatrix3[100]) +
			dot(t2, weightMatrix3[101]) +
			dot(t3, weightMatrix3[102]) +
			dot(t4, weightMatrix3[103]) +
			dot(t5, weightMatrix3[104]) +
			dot(t6, weightMatrix3[105]) +
			dot(t7, weightMatrix3[106]) +
			dot(t8, weightMatrix3[107])
			);
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[99]) +
			dot(t1, weightMatrix4[100]) +
			dot(t2, weightMatrix4[101]) +
			dot(t3, weightMatrix4[102]) +
			dot(t4, weightMatrix4[103]) +
			dot(t5, weightMatrix4[104]) +
			dot(t6, weightMatrix4[105]) +
			dot(t7, weightMatrix4[106]) +
			dot(t8, weightMatrix4[107])
			);
			#endif
		}
			#endif

			#if NUM_INPUT_PLANES > 48

		else if (i == 48) {
			s.r += (dot(t0, weightMatrix1[108]) +
			dot(t1, weightMatrix1[109]) +
			dot(t2, weightMatrix1[110]) +
			dot(t3, weightMatrix1[111]) +
			dot(t4, weightMatrix1[112]) +
			dot(t5, weightMatrix1[113]) +
			dot(t6, weightMatrix1[114]) +
			dot(t7, weightMatrix1[115]) +
			dot(t8, weightMatrix1[116])
			);
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[108]) +
			dot(t1, weightMatrix2[109]) +
			dot(t2, weightMatrix2[110]) +
			dot(t3, weightMatrix2[111]) +
			dot(t4, weightMatrix2[112]) +
			dot(t5, weightMatrix2[113]) +
			dot(t6, weightMatrix2[114]) +
			dot(t7, weightMatrix2[115]) +
			dot(t8, weightMatrix2[116])
			);
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[108]) +
			dot(t1, weightMatrix3[109]) +
			dot(t2, weightMatrix3[110]) +
			dot(t3, weightMatrix3[111]) +
			dot(t4, weightMatrix3[112]) +
			dot(t5, weightMatrix3[113]) +
			dot(t6, weightMatrix3[114]) +
			dot(t7, weightMatrix3[115]) +
			dot(t8, weightMatrix3[116])
			);
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[108]) +
			dot(t1, weightMatrix4[109]) +
			dot(t2, weightMatrix4[110]) +
			dot(t3, weightMatrix4[111]) +
			dot(t4, weightMatrix4[112]) +
			dot(t5, weightMatrix4[113]) +
			dot(t6, weightMatrix4[114]) +
			dot(t7, weightMatrix4[115]) +
			dot(t8, weightMatrix4[116])
			);
			#endif
		}
			#endif

			#if NUM_INPUT_PLANES > 52

		else if (i == 52) {
			s.r += (dot(t0, weightMatrix1[117]) +
			dot(t1, weightMatrix1[118]) +
			dot(t2, weightMatrix1[119]) +
			dot(t3, weightMatrix1[120]) +
			dot(t4, weightMatrix1[121]) +
			dot(t5, weightMatrix1[122]) +
			dot(t6, weightMatrix1[123]) +
			dot(t7, weightMatrix1[124]) +
			dot(t8, weightMatrix1[125])
			);
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[117]) +
			dot(t1, weightMatrix2[118]) +
			dot(t2, weightMatrix2[119]) +
			dot(t3, weightMatrix2[120]) +
			dot(t4, weightMatrix2[121]) +
			dot(t5, weightMatrix2[122]) +
			dot(t6, weightMatrix2[123]) +
			dot(t7, weightMatrix2[124]) +
			dot(t8, weightMatrix2[125])
			);
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[117]) +
			dot(t1, weightMatrix3[118]) +
			dot(t2, weightMatrix3[119]) +
			dot(t3, weightMatrix3[120]) +
			dot(t4, weightMatrix3[121]) +
			dot(t5, weightMatrix3[122]) +
			dot(t6, weightMatrix3[123]) +
			dot(t7, weightMatrix3[124]) +
			dot(t8, weightMatrix3[125])
			);
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[117]) +
			dot(t1, weightMatrix4[118]) +
			dot(t2, weightMatrix4[119]) +
			dot(t3, weightMatrix4[120]) +
			dot(t4, weightMatrix4[121]) +
			dot(t5, weightMatrix4[122]) +
			dot(t6, weightMatrix4[123]) +
			dot(t7, weightMatrix4[124]) +
			dot(t8, weightMatrix4[125])
			);
			#endif
		}
			#endif

			#if NUM_INPUT_PLANES > 56

		else if (i == 56) {
			s.r += (dot(t0, weightMatrix1[126]) +
			dot(t1, weightMatrix1[127]) +
			dot(t2, weightMatrix1[128]) +
			dot(t3, weightMatrix1[129]) +
			dot(t4, weightMatrix1[130]) +
			dot(t5, weightMatrix1[131]) +
			dot(t6, weightMatrix1[132]) +
			dot(t7, weightMatrix1[133]) +
			dot(t8, weightMatrix1[134])
			);
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[126]) +
			dot(t1, weightMatrix2[127]) +
			dot(t2, weightMatrix2[128]) +
			dot(t3, weightMatrix2[129]) +
			dot(t4, weightMatrix2[130]) +
			dot(t5, weightMatrix2[131]) +
			dot(t6, weightMatrix2[132]) +
			dot(t7, weightMatrix2[133]) +
			dot(t8, weightMatrix2[134])
			);
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[126]) +
			dot(t1, weightMatrix3[127]) +
			dot(t2, weightMatrix3[128]) +
			dot(t3, weightMatrix3[129]) +
			dot(t4, weightMatrix3[130]) +
			dot(t5, weightMatrix3[131]) +
			dot(t6, weightMatrix3[132]) +
			dot(t7, weightMatrix3[133]) +
			dot(t8, weightMatrix3[134])
			);
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[126]) +
			dot(t1, weightMatrix4[127]) +
			dot(t2, weightMatrix4[128]) +
			dot(t3, weightMatrix4[129]) +
			dot(t4, weightMatrix4[130]) +
			dot(t5, weightMatrix4[131]) +
			dot(t6, weightMatrix4[132]) +
			dot(t7, weightMatrix4[133]) +
			dot(t8, weightMatrix4[134])
			);
			#endif
		}
			#endif

			#if NUM_INPUT_PLANES > 60

		else {
			s.r += (dot(t0, weightMatrix1[135]) +
			dot(t1, weightMatrix1[136]) +
			dot(t2, weightMatrix1[137]) +
			dot(t3, weightMatrix1[138]) +
			dot(t4, weightMatrix1[139]) +
			dot(t5, weightMatrix1[140]) +
			dot(t6, weightMatrix1[141]) +
			dot(t7, weightMatrix1[142]) +
			dot(t8, weightMatrix1[143])
			);
			#ifdef USE_COMPONENT_G
			s.g += (dot(t0, weightMatrix2[135]) +
			dot(t1, weightMatrix2[136]) +
			dot(t2, weightMatrix2[137]) +
			dot(t3, weightMatrix2[138]) +
			dot(t4, weightMatrix2[139]) +
			dot(t5, weightMatrix2[140]) +
			dot(t6, weightMatrix2[141]) +
			dot(t7, weightMatrix2[142]) +
			dot(t8, weightMatrix2[143])
			);
			#endif
			#ifdef USE_COMPONENT_B
			s.b += (dot(t0, weightMatrix3[135]) +
			dot(t1, weightMatrix3[136]) +
			dot(t2, weightMatrix3[137]) +
			dot(t3, weightMatrix3[138]) +
			dot(t4, weightMatrix3[139]) +
			dot(t5, weightMatrix3[140]) +
			dot(t6, weightMatrix3[141]) +
			dot(t7, weightMatrix3[142]) +
			dot(t8, weightMatrix3[143])
			);
			#endif
			#ifdef USE_COMPONENT_A
			s.a += (dot(t0, weightMatrix4[135]) +
			dot(t1, weightMatrix4[136]) +
			dot(t2, weightMatrix4[137]) +
			dot(t3, weightMatrix4[138]) +
			dot(t4, weightMatrix4[139]) +
			dot(t5, weightMatrix4[140]) +
			dot(t6, weightMatrix4[141]) +
			dot(t7, weightMatrix4[142]) +
			dot(t8, weightMatrix4[143])
			);
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

