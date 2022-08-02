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
