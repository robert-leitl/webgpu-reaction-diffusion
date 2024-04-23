// language=C
export const CompositeShader = `
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f
}

struct AnimationUniforms {
   pulse: f32,
};

@group(0) @binding(0) var<uniform> animationUniforms: AnimationUniforms;

@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    const pos : array<vec2f, 3> = array(
        vec2f(-1, 3),
        vec2f(-1, -1),
        vec2f(3, -1)
    );
    const uv : array<vec2f, 3> = array(
        vec2f(0, 2),
        vec2f(0, 0),
        vec2f(2, 0)
    );
    var output : VertexOutput;
    output.position = vec4f(pos[vertexIndex], 0., 1.);
    output.uv = uv[vertexIndex];
    return output;
}

@group(0) @binding(1) var inputTexSampler : sampler;
@group(0) @binding(2) var inputTex : texture_2d<f32>;

// https://iquilezles.org/articles/palettes/
fn pal(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.28318 * (c * t + d));
}

// Bulge distortion based on: https://www.shadertoy.com/view/ldBfRV
fn distort(r: vec2<f32>, alpha: f32) -> vec2<f32> {
    return r + r * -alpha * (1. - dot(r, r) * 1.25);
}

fn emboss(
    p: vec2f,
    channel: vec4f,
    center: vec4f,
    tex: texture_2d<f32>, 
    texSampler: sampler,
    texelSize: vec2f,
    scale: f32,
    shift: f32) -> vec4f
{
    let tlColor: vec4f = textureSample(tex, texSampler, p + vec2(-texelSize.x,  texelSize.y) * scale);
    let brColor: vec4f = textureSample(tex, texSampler, p + vec2(texelSize.x,  -texelSize.y) * scale);
    let c: f32 = smoothstep(0., shift, dot(center, channel));
    let tl: f32 = smoothstep(0., shift, dot(tlColor, channel));
    let br: f32 = smoothstep(0., shift, dot(brColor, channel));
    return vec4f(tl, c, br, clamp(2.0 * br - c - tl, 0.0, 1.0));
}

@fragment
fn frag_main(@location(0) uv : vec2f) -> @location(0) vec4f {

    // add bulge distortion to the uv coords
    let p = distort(uv * 2. - 1., -0.15) * .5 + .5;
    
    // get input data
    let inputTexSize : vec2f = vec2f(textureDimensions(inputTex));
    let inputTexelSize = 1. / inputTexSize;
    let input: vec4f = textureSample(inputTex, inputTexSampler, p);
    
    // use the chemical B distribution as the base color value
    let value = smoothstep(0.225, .8, input.g);
    var base: vec3f = pal(value * .4 + 0.4, vec3(.5,0.5,0.5), vec3(0.5,0.5,.5), vec3(1.,1.0,1.0), vec3(0.05,0.1,0.2));
    base *= 1.5 * ((animationUniforms.pulse) * .15 + .85);
    
    // get centered uv coords and distance to center
    let st = uv * 2. - 1.;
    let dist = length(st);
    
    // inner emboss effect
    var emboss1: vec4f = emboss(p, vec4(1., 0., 0., 0.), input, inputTex, inputTexSampler, inputTexelSize, .5, .4 + dist * .3);
    emboss1.w = emboss1.w * .3 * (animationUniforms.pulse * .2 + .8);
    
    // inner specular from emboss data
    let specular = smoothstep(0.2, 0.3, 2.0 * emboss1.x - emboss1.y - emboss1.z) * .5 * (1. - dist) * ((1. - animationUniforms.pulse) * .15 + .85);
    
    // outer emboss for iridescence
    var emboss2: vec4f = emboss(p, vec4(0., 1., 0., 0.), input, inputTex, inputTexSampler, inputTexelSize, .8, .1);
    var iridescence = pal(input.r * 5. + .2, vec3(.5,0.5,0.5), vec3(0.5,0.5,.5), vec3(1.,1.0,1.0),vec3(0.0,0.33,0.67));
    iridescence = mix(iridescence, vec3f(0.), smoothstep(0., .4, max(input.g, emboss2.w)));
    iridescence *= .07 * ((1. - animationUniforms.pulse) * .2 + .8);
    
    // simple centered vignette
    let vignette = dist * .075;
    
    var color: vec4f = vec4f(base + vec3(emboss1.w) + specular + iridescence - vignette, 1.);
    
    return color;
}
`;
