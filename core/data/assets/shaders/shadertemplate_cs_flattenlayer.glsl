/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*/
layout(OUTPUT_FORMAT, binding=0) writeonly uniform PRECISION image2DArray uOutImage;
layout(OUTPUT_FORMAT, binding=1) readonly uniform PRECISION image2DArray uInImage;
layout(location = 2) uniform int uWidth;
layout(location = 3) uniform int uHeight;
//layout(binding=5) writeonly buffer destBuffer{
//    float data[];
//} uOutBuffer;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int z = pos.z/(uWidth*uHeight*4);        
    int offset = z*uWidth*uHeight*4;    
    int wh = uWidth*uHeight;
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
