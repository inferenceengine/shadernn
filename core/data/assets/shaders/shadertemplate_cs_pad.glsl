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
layout(location=4) uniform ivec2 uPad;
layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    if (all(lessThan(ivec3(gl_GlobalInvocationID), uOutputSize)))
    {
        ivec3 pos = ivec3(gl_GlobalInvocationID);
        ivec2 s0 = pos.xy-uPad;
        int sy = s0.y;
        #ifdef CONSTANT_PADDING   
        sy = ((sy >= 0) && (sy < uInputSize.y)) ? sy: uInputSize.y;
        #endif    
        #ifdef REPLICATE_PADDING   
        sy =  min(max(sy, 0), uInputSize.y-1); 
        #endif
        #ifdef REFLECT_PADDING           
        sy = (sy < 0) ? -sy:sy;
        sy = (sy >= uInputSize.y) ? 2*uInputSize.y-2-sy:sy;  
        #endif
        int sx = s0.x;
        #ifdef CONSTANT_PADDING   
        sx = ((sx >= 0) && (sx < uInputSize.x)) ? sx: uInputSize.x;
        #endif
        #ifdef REPLICATE_PADDING  
        sx = min(max(sx, 0), uInputSize.x-1); 
        #endif
        #ifdef REFLECT_PADDING   
        sx = (sx < 0) ? -sx:sx;
        sx = (sx >= uInputSize.x) ? 2*uInputSize.x-2-sx:sx;   
        #endif  
        #ifdef INPUT_TEXTURE_2D
        vec4 sum = imageLoad(uInput, ivec2(sx, sy)); 
        #else
        vec4 sum = imageLoad(uInput, ivec3(sx, sy, pos.z)); 
        #endif
        #ifdef OUTPUT_TEXTURE_2D
        imageStore(uOutput, ivec2(pos.x, pos.y), sum);
        #else
        imageStore(uOutput, pos, sum);
        #endif     
    }
}
