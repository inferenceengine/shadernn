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
layout(location=9) uniform ivec3 uOutputSize;
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    ivec3 outputSize = uOutputSize;
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (all(lessThan(pos, outputSize)))
    {
        #ifdef INPUT_TEXTURE_2D
        vec4 color = imageLoad(uInput, ivec2(pos.x, pos.y));
        #else
        vec4 color = imageLoad(uInput, pos);
        #endif
        vec4 movingVariance = uVariance.data[pos.z]; 
        vec4 movingMean = uMean.data[pos.z];         
        vec4 gamma = uGamma.data[pos.z]; 
        vec4 beta = uBeta.data[pos.z];                 
        vec4 sqrtVar = sqrt(movingVariance + vec4(0.001f)); 
        sqrtVar = max(sqrtVar, vec4(0.0001f)); 
        color = ((gamma/sqrtVar) * (color - movingMean)) + beta;                                                   
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
