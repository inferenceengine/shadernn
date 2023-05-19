/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;

layout(location=3) uniform int uConstantUnaryType;
layout(location=4) uniform float uConstantValue;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    #ifdef INPUT_TEXTURE_2D
    ivec3 inSize = ivec3(imageSize(u_Input),1);
    #else
    ivec3 inSize = imageSize(u_Input);
    #endif
    int unaryType = uConstantUnaryType;
    float value = uConstantValue;
    if (all(lessThan(pos, inSize)))
    {
        #ifdef INPUT_TEXTURE_2D
        vec4 color = imageLoad(u_Input, pos.xy);
        #else
        vec4 color = imageLoad(u_Input, pos.xyz);
        #endif
        if (unaryType == 0) {  //No touch
            #ifdef OUTPUT_TEXTURE_2D
            imageStore(u_Output, pos.xy, color);
            #else
            imageStore(u_Output, pos, color);
            #endif
            return;
        } 
        if (unaryType == 1) {  //Fixed value
            color = vec4(value);        
            #ifdef OUTPUT_TEXTURE_2D
            imageStore(u_Output, pos.xy, color);
            #else
            imageStore(u_Output, pos, color);
            #endif
            return;
        } 
        if (unaryType == 2) { //-
            color = vec4(0) - color;
            #ifdef OUTPUT_TEXTURE_2D
            imageStore(u_Output, pos.xy, color);
            #else
            imageStore(u_Output, pos, color);
            #endif
            return;
        } 
        if (unaryType == 3) { //1/x
            color = vec4(1)/color;
            #ifdef OUTPUT_TEXTURE_2D
            imageStore(u_Output, pos.xy, color);
            #else
            imageStore(u_Output, pos, color);
            #endif
            return;
        } 
        if (unaryType == 4) { //x*x
            color  = color*color;
            #ifdef OUTPUT_TEXTURE_2D
            imageStore(u_Output, pos.xy, color);
            #else
            imageStore(u_Output, pos, color);
            #endif
            return;
        }         
        if (unaryType == 5) { //exp(x)
            color   = exp(color);
            #ifdef OUTPUT_TEXTURE_2D
            imageStore(u_Output, pos.xy, color);
            #else
            imageStore(u_Output, pos, color);
            #endif
            return;
        }
        if (unaryType == 6) {  //abs
            color = abs(color);
            #ifdef OUTPUT_TEXTURE_2D
            imageStore(u_Output, pos.xy, color);
            #else
            imageStore(u_Output, pos, color);
            #endif
            return;
        } 
    }
}

