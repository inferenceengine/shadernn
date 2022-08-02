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
precision FLOAT_PRECISION sampler2D;
precision FLOAT_PRECISION sampler2DArray;

layout(location = 0) out vec4 o_pixel;

_PLACEHOLDER_UNIFORMS_DECLARATION_

void main()
{
    int lod = 0;
    FLOAT_PRECISION vec4 s = vec4(0.0f);

    ivec3 uvt = ivec3(gl_FragCoord.xy, 0);
    ivec2 uv = ivec2(gl_FragCoord.xy);

    _PLACEHOLDER_CALCULATION_

    o_pixel = s;
}
