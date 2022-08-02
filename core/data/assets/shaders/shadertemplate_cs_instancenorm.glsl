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
layout(binding=5) readonly buffer beta{
    vec4 data[];
} uBeta;
layout(binding=6) readonly buffer gamma{
    vec4 data[];
} uGamma;
layout(location=7) uniform ivec3 uOutputSize;
layout(location=8) uniform ivec3 uInputSize;
void retirePhase() { memoryBarrierShared(); barrier(); }
layout (local_size_x = WORK_X, local_size_y = WORK_Y, local_size_z = WORK_Z) in;
shared vec4 shared_mem[256]; 
void main()
{
    ivec3 gid = ivec3(gl_GlobalInvocationID); 
    ivec3 outputSize = uOutputSize;
    int tid = int(gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x);
    if (all(lessThan(gid, outputSize)))
    {
        int width  = uInputSize.x; 
        int height = uInputSize.y; 
        int thread_count = WORK_X * WORK_Y; 
        ivec2 tg_size = ivec2(WORK_X, WORK_Y); 
        vec4 sum = vec4(0.0f); 
        for(int xIndex = gid.x; xIndex < width; xIndex += tg_size.x) { 
            for(int yIndex = gid.y; yIndex < height; yIndex += tg_size.y) { 
                #ifdef INPUT_TEXTURE_2D
                vec4 val = imageLoad(uInput, ivec2(xIndex, yIndex));     
                #else
                vec4 val = imageLoad(uInput, ivec3(xIndex, yIndex, gid.z));                 
                #endif  
                sum += val; 
            } 
        } 
        shared_mem[tid] = sum; 
        retirePhase(); 
        sum = vec4(0.0f);
        if (tid < 32) { 
            for (int i = tid + 32; i < thread_count; i += 32) { 
                sum += shared_mem[i]; 
            } 
        } 
        shared_mem[tid] += sum; 
        retirePhase(); 
        // Calculate mean 
        sum = vec4(0.0f);; 
        if (tid == 0) { 
            int top = min(int(32), thread_count); 
            for (int i = 0; i < top; i += 1) { 
                sum += shared_mem[i]; 
            } 
            shared_mem[0] = sum / float(width * height); 
        } 
        retirePhase(); 
        vec4 mean = shared_mem[0]; 
        retirePhase(); 
        // Variance     
        sum = vec4(0.0f); 
        for(int xIndex = gid.x; xIndex < width; xIndex += tg_size.x) { 
            for(int yIndex = gid.y; yIndex < height; yIndex += tg_size.y) { 
                #ifdef INPUT_TEXTURE_2D
                vec4 val = imageLoad(uInput, ivec2(xIndex, yIndex));     
                #else
                vec4 val = imageLoad(uInput, ivec3(xIndex, yIndex, gid.z));                 
                #endif  
                sum += (val-mean) * (val-mean); 
            } 
        } 
        shared_mem[tid] = sum; 
        retirePhase(); 
        // Reduce to 32 values  
        sum = vec4(0.0f); 
        if (tid < 32) { 
            for (int i = tid + 32; i < thread_count; i += 32) { 
                sum += shared_mem[i]; 
            } 
        } 
        shared_mem[tid] += sum; 
        retirePhase(); 
        // Calculate variance   
        sum = vec4(0.0f); 
        if (tid == 0) { 
            int top = min(int(32), thread_count); 
            for (int i = 0; i < top; i += 1) { 
                sum += shared_mem[i]; 
            } 
            shared_mem[0] = sum / float(width * height); 
        } 
        retirePhase(); 
        vec4 sigma = sqrt(shared_mem[0] + vec4(0.00001f)); 
        vec4 multiplier = uGamma.data[gid.z] / sigma; 
        for(int xIndex = gid.x; xIndex < width; xIndex += tg_size.x) { 
            for(int yIndex = gid.y; yIndex < height; yIndex += tg_size.y) { 
                #ifdef INPUT_TEXTURE_2D
                vec4 val = imageLoad(uInput, ivec2(xIndex, yIndex));     
                #else
                vec4 val = imageLoad(uInput, ivec3(xIndex, yIndex, gid.z));                 
                #endif    
                vec4 color = (val - mean) * multiplier + uBeta.data[gid.z];                   
                #ifdef RELU
                color = max(color, vec4(0));
                #endif
                #ifdef RELU6
                color = clamp(color, vec4(0), vec4(6));
                #endif
                #ifdef TANH
                color = tanh(color);
                #endif
                #ifdef SIGMOID
                color  = vec4(1.0f)/(vec4(1.0f)+ exp(-color));
                #endif        
                #ifdef LEAKYRELU_VAL
                color   = max(color,  (color * vec4(LEAKYRELU_VAL)));
                #endif 
                #ifdef SILU
                color    = color  * vec4(1.0f)/(vec4(1.0f)+ exp(-color));
                #endif
                #ifdef OUTPUT_TEXTURE_2D
                imageStore(uOutput, ivec2(xIndex, yIndex), color);
                #else
                imageStore(uOutput, ivec3(xIndex, yIndex, gid.z), color);
                #endif
            } 
        }            
    }
}
