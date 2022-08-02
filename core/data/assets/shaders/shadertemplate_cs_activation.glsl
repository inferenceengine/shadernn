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
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    #ifdef INPUT_TEXTURE_2D
    vec4 value = imageLoad(uInput, ivec2(pos.x, pos.y));
    #else
    vec4 value = imageLoad(uInput, pos);
    #endif
    #ifdef RELU
    value = max(value, vec4(0));
    #endif
    #ifdef RELU6
    value = clamp(value, vec4(0), vec4(6));
    #endif
    #ifdef TANH
    value = tanh(value);
    #endif
    #ifdef SIGMOID
    value  = vec4(1.0f)/(vec4(1.0f)+ exp(-value));
    #endif
    #ifdef LEAKYRELU_VAL
    value   = max(value,  (value * vec4(LEAKYRELU_VAL)));
    #endif
    #ifdef SILU
    value    = value  * vec4(1.0f)/(vec4(1.0f)+ exp(-value));
    #endif
    #ifdef OUTPUT_TEXTURE_2D
    imageStore(uOutput, ivec2(pos.x, pos.y), value);
    #else
    imageStore(uOutput, pos, value);
    #endif
}
