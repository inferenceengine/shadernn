
#define FLOAT_PRECISION _PLACEHOLDER_PRECISION_

precision FLOAT_PRECISION float;
precision FLOAT_PRECISION sampler2D;
precision FLOAT_PRECISION sampler2DArray;

uniform sampler2DArray inputTextures;

layout(location = 0) out vec4 o_pixel;

void main()
{
    FLOAT_PRECISION vec4 s = vec4(0.0f);

    ivec2 uv = ivec2(gl_FragCoord.xy);

    // The layer setting is for moonwellbox 4/3/2020 version.
    // Layer 0: first input High image(RGBA); Layer 1:second input Low image(RGBA);
    // Layer 2: illumination_High, illumination_Low, 0, 0 (RGBA)
    // Result: s = High_image (RGB) / illumination_High
    float illumination_h = texelFetch(inputTextures, ivec3(uv, 2), 0).r;
    s = texelFetch(inputTextures, ivec3(uv, 0), 0).rgba;

    s = vec4(s.r/illumination_h, s.g/illumination_h, s.b/illumination_h, 0.0f);
    o_pixel = s;
}
