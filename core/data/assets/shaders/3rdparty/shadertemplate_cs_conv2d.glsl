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
layout(location=8) uniform int uUnroll;
layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;
#define UP_DIV(x, y) (((x)+(y)-1)/(y))
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    if (all(lessThan(ivec3(gl_GlobalInvocationID), uOutputSize)))
    {
        ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(uUnroll, 1, 1);
        int kernelX = uKernelSize.x;
        ivec3 inputSize = uInputSize;
        ivec2 s0 = pos.xy*uStride-uPad;
        int fx, fy, fz;
        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));
        vec4 color = uBias.data[pos.z];
        vec4 color2 = color;
        vec4 color3 = color;
        vec4 color4 = color;
        int kernelY = pos.z;
        int kernelPlane = uInputSize.z * uOutputSize.z;     
        for (fy=0; fy<uKernelSize.y; ++fy)
        {
            int sy = fy*uDilate.y + s0.y;
            #ifdef CONSTANT_PADDING   
            sy = ((sy >= 0) && (sy < inputSize.y)) ? sy: inputSize.y;
            #endif    
            #ifdef REPLICATE_PADDING   
                sy =  min(max(sy, 0), inputSize.y-1); 
            #endif
            #ifdef REFLECT_PADDING           
                sy = (sy < 0) ? -sy:sy;
                sy = (sy >= inputSize.y) ? 2*inputSize.y-2-sy:sy;  
            #endif
            for (fx=0; fx<kernelX; ++fx)
            {
                int kernelZ = fx + fy*kernelX;
                int sx1 = fx*uDilate.x + s0.x;
                int sx2 = sx1 + uStride.x;
                int sx3 = sx2 + uStride.x;
                int sx4 = sx3 + uStride.x;
                #ifdef CONSTANT_PADDING   
                sx1 = ((sx1 >= 0) && (sx1 < inputSize.x)) ? sx1: inputSize.x;
                sx2 = ((sx2 >= 0) && (sx2 < inputSize.x)) ? sx2: inputSize.x;
                sx3 = ((sx3 >= 0) && (sx3 < inputSize.x)) ? sx3: inputSize.x;
                sx4 = ((sx4 >= 0) && (sx4 < inputSize.x)) ? sx4: inputSize.x;
                #endif
                #ifdef REPLICATE_PADDING  
                sx1 = min(max(sx1, 0), inputSize.x-1);
                sx2 = min(max(sx2, 0), inputSize.x-1);
                sx3 = min(max(sx3, 0), inputSize.x-1);
                sx4 = min(max(sx4, 0), inputSize.x-1);      
                #endif
                #ifdef REFLECT_PADDING   
                sx1 = (sx1 < 0) ? -sx1:sx1;
                sx1 = (sx1 >= inputSize.x) ? 2*inputSize.x-2-sx1:sx1;   
                sx2 = (sx2 < 0) ? -sx2:sx2;
                sx2 = (sx2 >= inputSize.x) ? 2*inputSize.x-2-sx2:sx2;      
                sx3 = (sx3 < 0) ? -sx3:sx3;
                sx3 = (sx3 >= inputSize.x) ? 2*inputSize.x-2-sx3:sx3;       
                sx4 = (sx4 < 0) ? -sx4:sx4;
                sx4 = (sx4 >= inputSize.x) ? 2*inputSize.x-2-sx4:sx4; 
                #endif                                   
                fz = 0;
                for (; fz<inputSize.z; ++fz)
                {
                    int kernel4X = uUnroll*fz;
                    vec4 k0 = texelFetch(uKernel, ivec3(kernel4X+0, kernelY, kernelZ), 0);
                    vec4 k1 = texelFetch(uKernel, ivec3(kernel4X+1, kernelY, kernelZ), 0);
                    vec4 k2 = texelFetch(uKernel, ivec3(kernel4X+2, kernelY, kernelZ), 0);
                    vec4 k3 = texelFetch(uKernel, ivec3(kernel4X+3, kernelY, kernelZ), 0);
                    
                    mat4 k = mat4(k0, k1, k2, k3);
                    
                    #ifdef INPUT_TEXTURE_2D
                    color  += k*imageLoad(uInput, ivec2(sx1, sy));
                    color2 += k*imageLoad(uInput, ivec2(sx2, sy));
                    color3 += k*imageLoad(uInput, ivec2(sx3, sy));
                    color4 += k*imageLoad(uInput, ivec2(sx4, sy));       
                    #else
                    color  += k*imageLoad(uInput, ivec3(sx1, sy, fz));
                    color2 += k*imageLoad(uInput, ivec3(sx2, sy, fz));
                    color3 += k*imageLoad(uInput, ivec3(sx3, sy, fz));
                    color4 += k*imageLoad(uInput, ivec3(sx4, sy, fz));
                    #endif       
                }
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
        color2 = ((gamma/sqrtVar) * (color2 - movingMean)) + beta; 
        color3 = ((gamma/sqrtVar) * (color3 - movingMean)) + beta; 
        color4 = ((gamma/sqrtVar) * (color4 - movingMean)) + beta;                                                     
        #endif 
        #ifdef RELU
        color = max(color, vec4(0));
        color2 = max(color2, vec4(0));
        color3 = max(color3, vec4(0));
        color4 = max(color4, vec4(0));
        #endif
        #ifdef RELU6
        color = clamp(color, vec4(0), vec4(6));
        color2 = clamp(color2, vec4(0), vec4(6));
        color3 = clamp(color3, vec4(0), vec4(6));
        color4 = clamp(color4, vec4(0), vec4(6));
        #endif
        #ifdef TANH
        color = tanh(color);
        color2 = tanh(color2);
        color3 = tanh(color3);
        color4 = tanh(color4);
        #endif
        #ifdef SIGMOID
        color  = vec4(1.0f)/(vec4(1.0f)+ exp(-color));
        color2 = vec4(1.0f)/(vec4(1.0f)+ exp(-color2));
        color3 = vec4(1.0f)/(vec4(1.0f)+ exp(-color3));
        color4 = vec4(1.0f)/(vec4(1.0f)+ exp(-color4));
        #endif        
        #ifdef LEAKYRELU_VAL
        color   = max(color,  (color * vec4(LEAKYRELU_VAL)));
        color2  = max(color2, (color2 * vec4(LEAKYRELU_VAL)));
        color3  = max(color3, (color3 * vec4(LEAKYRELU_VAL)));
        color4  = max(color4, (color4 * vec4(LEAKYRELU_VAL)));
        #endif 
        #ifdef SILU
        color    = color  * vec4(1.0f)/(vec4(1.0f)+ exp(-color));
        color2   = color2 * vec4(1.0f)/(vec4(1.0f)+ exp(-color));
        color3   = color3 * vec4(1.0f)/(vec4(1.0f)+ exp(-color));
        color4   = color4 * vec4(1.0f)/(vec4(1.0f)+ exp(-color));
        #endif                           
        #ifdef OUTPUT_TEXTURE_2D
        imageStore(uOutput, ivec2(pos.x+0, pos.y), color);
        imageStore(uOutput, ivec2(pos.x+1, pos.y), color2);
        imageStore(uOutput, ivec2(pos.x+2, pos.y), color3);
        imageStore(uOutput, ivec2(pos.x+3, pos.y), color4);  
        #else
        imageStore(uOutput, ivec3(pos.x+0, pos.y, pos.z), color);
        imageStore(uOutput, ivec3(pos.x+1, pos.y, pos.z), color2);
        imageStore(uOutput, ivec3(pos.x+2, pos.y, pos.z), color3);
        imageStore(uOutput, ivec3(pos.x+3, pos.y, pos.z), color4);
        #endif           
    }
}
