layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    #ifdef INPUT_TEXTURE_2D
    vec4 value = imageLoad(uInput, ivec2(pos.x, pos.y));
    #else
    vec4 value = imageLoad(uInput, pos);
    #endif
    #ifdef RELU
    value = max(value, vec4(0));
    #endif
    #ifdef RELU6
    value = clamp(value, vec4(0), vec4(6));
    #endif
    #ifdef TANH
    value = tanh(value);
    #endif
    #ifdef SIGMOID
    value  = vec4(1.0f)/(vec4(1.0f)+ exp(-value));
    #endif
    #ifdef LEAKYRELU_VAL
    value   = max(value,  (value * vec4(LEAKYRELU_VAL)));
    #endif
    #ifdef SILU
    value    = value  * vec4(1.0f)/(vec4(1.0f)+ exp(-value));
    #endif
    #ifdef OUTPUT_TEXTURE_2D
    imageStore(uOutput, ivec2(pos.x, pos.y), value);
    #else
    imageStore(uOutput, pos, value);
    #endif
}
