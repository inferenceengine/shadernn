layout(location=1) uniform ivec2 inImgDepths;
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    
    vec4 outValue;
    vec4 outValue1;
    if(pos.z < inImgDepths.x)
    {
        #ifdef INPUT0_TEXTURE_2D
        outValue = imageLoad(uInput0, ivec2(pos.x, pos.y));
        #else
        outValue = imageLoad(uInput0, ivec3(pos.x, pos.y, pos.z));
        #endif
    }
    else
    {
        #ifdef INPUT1_TEXTURE_2D
        outValue = imageLoad(uInput1, ivec2(pos.x, pos.y));
        #else
        outValue = imageLoad(uInput1, ivec3(pos.x, pos.y, pos.z - inImgDepths.x));
        #endif
    }
    
    imageStore(uOutput, pos, outValue);
}
