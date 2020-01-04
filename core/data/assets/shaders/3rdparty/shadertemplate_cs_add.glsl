layout(location=3) uniform ivec4 imgSize;
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 inSize = imgSize.xyz;
    if (all(lessThan(pos, inSize)))
    {
        #ifdef INPUT_TEXTURE_2D
        vec4 sum = imageLoad(uInput0, ivec2(pos.x, pos.y)) + imageLoad(uInput1, ivec2(pos.x, pos.y));
        #else
        vec4 sum = imageLoad(uInput0, pos) + imageLoad(uInput1, pos);
        #endif
        #ifdef RELU
        sum = max(sum, vec4(0));
        #endif
        #ifdef RELU6
        sum = clamp(sum, vec4(0), vec4(6));
        #endif
        #ifdef TANH
        sum = tanh(sum);
        #endif
        #ifdef SIGMOID
        sum  = vec4(1.0f)/(vec4(1.0f)+ exp(-sum));
        #endif        
        #ifdef LEAKYRELU_VAL
        sum   = max(sum,  (sum * vec4(LEAKYRELU_VAL)));
        #endif 
        #ifdef SILU
        sum    = sum  * vec4(1.0f)/(vec4(1.0f)+ exp(-sum));
        #endif   
        #ifdef OUTPUT_TEXTURE_2D
        imageStore(uOutput, ivec2(pos.x, pos.y), sum);
        #else
        imageStore(uOutput, pos, sum);
        #endif
    }
}
