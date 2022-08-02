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
#define FLOAT_PRECISION _PLACEHOLDER_PRECISION_

precision FLOAT_PRECISION float;
precision FLOAT_PRECISION sampler2DArray;

#ifdef OUTPUT_Y2RG
out FLOAT_PRECISION vec2 o_pixel;
#else
out FLOAT_PRECISION float o_pixel;
#endif

uniform float bias;
uniform uint kernelSize;
uniform sampler2DArray inputTextures;

// Following code requires #version 150
//layout(pixel_center_integer) in vec4 gl_FragCoord;
// Since above code not working, we need to manually adjust pixel center from the default (x+0.5, y+0.5) to (x,y)

void main()
{
    // Subpixel Merging
    FLOAT_PRECISION vec4 pixel = vec4(0.0f);
float s = 0.0f;
    int lod = 0;
    float fkernelSize = float(kernelSize);
    int component = int(round(mod(gl_FragCoord.x-0.5, fkernelSize) + fkernelSize*mod(gl_FragCoord.y-0.5, fkernelSize)));

#ifdef KERNEL_LARGER_THAN_2
    int layer = component / 4;
    component = component % 4;
    ivec3 uvt = ivec3(gl_FragCoord.x/ fkernelSize, gl_FragCoord.y / fkernelSize, layer);
#else
    ivec3 uvt = ivec3(gl_FragCoord.x/ fkernelSize, gl_FragCoord.y / fkernelSize, 0);
#endif
    pixel = texelFetch(inputTextures, uvt, lod);

    if (component == 0) {
        s = pixel.r;
    }
    else if (component == 1) {
        s = pixel.g;
    }
    else if (component == 2) {
        s = pixel.b;
    }
    else if (component == 3) {
        s = pixel.a;
    }

#ifdef OUTPUT_Y2RG
    #ifdef OUTPUT_Y2RG_HALF
        if (s < 0.0) {
            o_pixel = vec2(0.0, -s/2.0);
        }
        else {
            o_pixel = vec2(s/2.0, 0.0);
        }
    #else
        if (s < 0.0) {
            o_pixel = vec2(0.0, -s);
        }
        else {
            o_pixel = vec2(s, 0.0);
        }
    #endif
//#else
    //#ifdef OUTPUT_TEXID
    //    o_pixel = (s + 1.0)/2.0;
    //#else
    //    o_pixel = s;
    //#endif
#endif

