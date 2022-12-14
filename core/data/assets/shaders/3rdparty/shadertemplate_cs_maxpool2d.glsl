layout(location=2) uniform ivec2 uKernel;
layout(location=3) uniform ivec2 uStride;
layout(location=4) uniform ivec2 uPad;
layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 outputSize = uOutputSize;
    ivec2 spos = pos.xy*uStride-uPad;
    if (all(lessThan(pos, outputSize)))
    {
        ivec3 inputSize = uInputSize;
        ivec2 sfxy = max(ivec2(0), -spos);
        ivec2 efxy = min(uKernel, inputSize.xy-spos);
        vec4 color = vec4(-100000.0);
        for (int fy=sfxy.y; fy<efxy.y; ++fy)
        {
            for (int fx=sfxy.x; fx<efxy.x; ++fx)
            {
                #ifdef INPUT_TEXTURE_2D
                color = max(color, imageLoad(uInput, ivec2(spos.x+fx, spos.y+fy)));
                #else
                color = max(color, imageLoad(uInput, ivec3(spos.x+fx, spos.y+fy, pos.z)));
                #endif  
            }
        }
        #ifdef OUTPUT_TEXTURE_2D
        imageStore(uOutput, ivec2(pos.x, pos.y), color);
        #else
        imageStore(uOutput, pos, color);
        #endif  
    }
}
