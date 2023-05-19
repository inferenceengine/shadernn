layout(location=2) uniform ivec4 inImgSize;
layout(location=3) uniform ivec4 outImgSize;
layout(location=4) uniform vec2 scale;
layout(location=5) uniform vec4 means;
layout(location=6) uniform vec4 norms;   
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 inputImgSize = inImgSize.xyz;
    ivec3 outputImgSize = outImgSize.xyz;
    
    if(pos.x < outputImgSize.x && pos.y < outputImgSize.y && pos.z < outputImgSize.z)
    {
        float offsetX = 0.5 - 0.5 * scale.x;
        float offsetY = 0.5 - 0.5 * scale.y;

        float srcX = float(pos.x) * scale.x;
        srcX = srcX - offsetX;
        srcX = clamp(srcX, 0.0f, float(inputImgSize.x - 1));
        int x11 = int(floor(srcX));
        int x12 = x11 + 1;

        float srcY = float(pos.y) * scale.y;
        srcY = srcY - offsetY;
        srcY = clamp(srcY, 0.0f, float(inputImgSize.y - 1));
        int y11 = int(floor(srcY));
        int y12 = y11 + 1;
        #ifdef INPUT_TEXTURE_2D
        vec4 res4 = imageLoad(uInput, ivec2(x11, y12));
        vec4 res3 = imageLoad(uInput, ivec2(x12, y12));
        vec4 res1 = imageLoad(uInput, ivec2(x11, y11));
        vec4 res2 = imageLoad(uInput, ivec2(x12, y11));
        #else
        vec4 res4 = imageLoad(uInput, ivec3(x11, y12, pos.z));
        vec4 res3 = imageLoad(uInput, ivec3(x12, y12, pos.z));
        vec4 res1 = imageLoad(uInput, ivec3(x11, y11, pos.z));
        vec4 res2 = imageLoad(uInput, ivec3(x12, y11, pos.z));
        #endif
        vec4 outValue = res1 * vec4((float(x12) - srcX) * (float(y12) - srcY)) +
        res2 * vec4((srcX - float(x11)) * (float(y12) - srcY)) +
        res3 * vec4((srcX - float(x11)) * (srcY - float(y11))) +
        res4 * vec4((float(x12) - srcX) * (srcY - float(y11)));
        outValue = (outValue - means) * norms;
        #ifdef OUTPUT_TEXTURE_2D
        imageStore(uOutput, ivec2(pos.x, pos.y), outValue);
        #else
        imageStore(uOutput, pos, outValue);
        #endif
    }
    
}