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
        float srcX = float(pos.x) * scale.x;
        int x1 = int(floor(srcX));
        int x11 = clamp(x1, 0, inputImgSize.x - 1);
        int x12 = clamp(x1 + 1, 0, inputImgSize.x - 1);
        vec4 factorX = vec4(srcX - float(x1));
        float srcY = float(pos.y) * scale.y;
        int y1 = int(floor(srcY));
        int y11 = clamp(y1, 0, inputImgSize.y - 1);
        int y12 = clamp(y1 + 1, 0, inputImgSize.y - 1);
        vec4 factorY = vec4(srcY - float(y1));
        #ifdef INPUT_TEXTURE_2D
        vec4 res1 = imageLoad(uInput, ivec2(x11, y12));
		vec4 res2 = imageLoad(uInput, ivec2(x12, y12));
		vec4 res3 = imageLoad(uInput, ivec2(x11, y11));
		vec4 res4 = imageLoad(uInput, ivec2(x12, y11));
        #else
        vec4 res1 = imageLoad(uInput, ivec3(x11, y12, pos.z));
        vec4 res2 = imageLoad(uInput, ivec3(x12, y12, pos.z));
        vec4 res3 = imageLoad(uInput, ivec3(x11, y11, pos.z));
        vec4 res4 = imageLoad(uInput, ivec3(x12, y11, pos.z));
        #endif
        vec4 res11 = (vec4(1.0) - factorX) * res1 + factorX * res2;
        vec4 res12 = (vec4(1.0) - factorX) * res3 + factorX * res4;
        vec4 outValue = factorY * res11 + (vec4(1.0) - factorY) * res12;
        outValue = (outValue - means) * norms;    
        #ifdef OUTPUT_TEXTURE_2D
        imageStore(uOutput, ivec2(pos.x, pos.y), outValue);
        #else
        imageStore(uOutput, pos, outValue);
        #endif
    }
    
}
