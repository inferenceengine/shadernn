layout(binding=4) readonly buffer bias{
    vec4 data[];
} uBias;
layout(binding=5) readonly buffer beta{
    vec4 data[];
} uBeta;
layout(binding=6) readonly buffer gamma{
    vec4 data[];
} uGamma;
layout(binding=7) readonly buffer mean{
    vec4 data[];
} uMean;
layout(binding=8) readonly buffer variance{
    vec4 data[];
} uVariance;
layout(location=4) uniform ivec2 uPad;
layout(location=5) uniform ivec2 uKernelSize;
layout(location=6) uniform ivec2 uStride;
layout(location=7) uniform ivec2 uDilate;
layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;
#define UP_DIV(x, y) (((x)+(y)-1)/(y))
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(1, 1, 1);
    ivec3 outputSize = uOutputSize;
    if (all(lessThan(pos, outputSize)))
    {
        int KSIZE_Y = uKernelSize.y;
        int KSIZE_X = uKernelSize.x;
        ivec3 inputSize = uInputSize;
        ivec2 s0 = pos.xy*uStride-uPad;
        int fx, fy, fz;
        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));
        vec4 color = uBias.data[pos.z];
        for (fy=sfxy.y; fy<efxy.y; ++fy)
        {
            int sy = fy*uDilate.y + s0.y;
            for (fx=sfxy.x; fx<efxy.x; ++fx)
            {
                int sx1 = fx*uDilate.x + s0.x;
                vec4 k = texelFetch(uKernel, ivec3(pos.z, fx, fy), 0);
                #ifdef INPUT_TEXTURE_2D
                color  += k*imageLoad(uInput, ivec2(sx1, sy));    
                #else
                color  += k*imageLoad(uInput, ivec3(sx1, sy, pos.z));
                #endif  
            }
        }
        #ifdef USE_BATCH_NORMALIZATION 
        vec4 movingVariance = uVariance.data[pos.z]; 
        vec4 movingMean = uMean.data[pos.z];         
        vec4 gamma = uGamma.data[pos.z]; 
        vec4 beta = uBeta.data[pos.z];                 
        vec4 sqrtVar = sqrt(movingVariance + vec4(0.001f));             
        sqrtVar = max(sqrtVar, vec4(0.0001f)); 
        color = ((gamma/sqrtVar) * (color - movingMean)) + beta;                                                      
        #endif 
        #ifdef RELU
        color = max(color, vec4(0));
        #endif
        #ifdef RELU6
        color = clamp(color, vec4(0), vec4(6));
        #endif
        #ifdef TANH
        color = tanh(color);
        #endif
        #ifdef SIGMOID
        color  = vec4(1.0f)/(vec4(1.0f)+ exp(-color));
        #endif        
        #ifdef LEAKYRELU_VAL
        color   = max(color,  (color * vec4(LEAKYRELU_VAL)));
        #endif 
        #ifdef SILU
        color    = color  * vec4(1.0f)/(vec4(1.0f)+ exp(-color));
        #endif
        #ifdef OUTPUT_TEXTURE_2D
        imageStore(uOutput, ivec2(pos.x+0, pos.y), color);
        #else
        imageStore(uOutput, ivec3(pos.x+0, pos.y, pos.z), color);
        #endif
    }
}
