layout(location=2) uniform ivec4 inImgSize;
layout(location=3) uniform ivec4 outImgSize;
layout(location=4) uniform vec2 scale;
layout(location=5) uniform vec4 means;
layout(location=6) uniform vec4 norms;                
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    // input output layout is NC4HW4
    
    ivec3 inputImgSize = inImgSize.xyz;
    ivec3 outputImgSize = outImgSize.xyz;
    
    if(pos.x < outputImgSize.x && pos.y < outputImgSize.y)
    {
        float srcX = float(pos.x) * scale.x;
        int x1 = int(floor(srcX));
        int x11 = clamp(x1, 0, inputImgSize.x - 1);
        
        float srcY = float(pos.y) * scale.y;
        int y1 = int(floor(srcY));
        int y11 = clamp(y1, 0, inputImgSize.y - 1);
        #ifdef INPUT_TEXTURE_2D
        vec4 outValue = imageLoad(uInput, ivec2(x11, y11));
        #else
        vec4 outValue = imageLoad(uInput, ivec3(x11, y11, pos.z));
        #endif
        
        outValue = (outValue - means) * norms;
        #ifdef OUTPUT_TEXTURE_2D
        imageStore(uOutput, ivec2(pos.x, pos.y), outValue);
        #else
        imageStore(uOutput, pos, outValue);
        #endif
    }
    
}
