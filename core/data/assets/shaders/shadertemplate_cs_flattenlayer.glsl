#version 320 es 
#define PRECISION mediump
precision PRECISION float;
layout(rgba32f, binding=0) writeonly uniform PRECISION image2DArray uOutImage;
layout(rgba32f, binding=1) readonly uniform PRECISION image2DArray uInImage;
layout(location = 2) uniform int uWidth;
layout(location = 3) uniform int uHeight;
layout(binding=5) writeonly buffer destBuffer{
    float data[];
} uOutBuffer;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int z = pos.z/(uWidth*uHeight*4);        
    int offset = z*uWidth*uHeight*4;    
    int wh = uWidth*uHeight;" 
    for (int w = 0; w < uWidth; w+=1) 
    {
       for (int h = 0; h < uHeight; h+=1) 
       {
           vec4 color0 = imageLoad(uInImage, ivec3(w, h, z));         
           imageStore(uOutImage,ivec3(offset+wh*0+h*uWidth+w, 0, 0),vec4(color0.r,0,0,0));
           imageStore(uOutImage,ivec3(offset+wh*1+h*uWidth+w, 0, 0),vec4(color0.g,0,0,0));
           imageStore(uOutImage,ivec3(offset+wh*2+h*uWidth+w, 0, 0),vec4(color0.b,0,0,0));
           imageStore(uOutImage,ivec3(offset+wh*3+h*uWidth+w, 0, 0),vec4(color0.a,0,0,0));            
       }
    }
}
