
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
