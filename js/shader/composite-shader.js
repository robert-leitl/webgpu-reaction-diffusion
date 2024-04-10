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

fn pal(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.28318 * (c * t + d));
}

@fragment
fn frag_main(@location(0) uv : vec2f) -> @location(0) vec4f {
    let inputTexSize : vec2f = vec2f(textureDimensions(inputTex));
    let inputTexelSize = 1. / inputTexSize;
    let input: vec4f = textureSample(inputTex, inputTexSampler, uv);
    
    _ = animationUniforms;
    
    let value = smoothstep(0.2, .8, input.g);
    
    var base: vec3f = pal(value * .4 + 0.4, vec3(.5,0.5,0.5),vec3(0.5,0.5,.5),vec3(1.,1.0,1.0),vec3(0.05,0.1,0.2));
    base *= 1.5 * (animationUniforms.pulse * .2 + .8);
    
    let st = uv * 2. - 1.;
    let dist = length(st);
    
    // emboss effect
    let embossScale = .5;
    let tlColor: vec4f = textureSample(inputTex, inputTexSampler, uv + vec2(-inputTexelSize.x,  inputTexelSize.y) * embossScale);
    let brColor: vec4f = textureSample(inputTex, inputTexSampler, uv + vec2(inputTexelSize.x,  -inputTexelSize.y) * embossScale);
    let c: f32 = smoothstep(0., .4 + dist * .3, input.r);
    let tl: f32 = smoothstep(0., .4 + dist * .3, tlColor.r);
    let br: f32 = smoothstep(0., .4 + dist * .3, brColor.r);
    var emboss: vec3f = vec3f(2.0 * br - c - tl);
    let luminance: f32 = clamp(0.299 * emboss.r + 0.587 * emboss.g + 0.114 * emboss.b, 0.0, 1.0);
    emboss = vec3f(luminance) * .3 * (animationUniforms.pulse * .2 + .8);
    let specular = smoothstep(0.2, 0.3, 2.0 * tl - c - br) * .5 * (1. - dist) * (animationUniforms.pulse * .2 + .8);
    
    let embossScale2 = 2.;
    let tlColor2: vec4f = textureSample(inputTex, inputTexSampler, uv + vec2(-inputTexelSize.x,  inputTexelSize.y) * embossScale2);
    let brColor2: vec4f = textureSample(inputTex, inputTexSampler, uv + vec2(inputTexelSize.x,  -inputTexelSize.y) * embossScale2);
    let c2: f32 = smoothstep(0.0, 1., input.g);
    let tl2: f32 = smoothstep(0.0, 1., tlColor2.g);
    let br2: f32 = smoothstep(0.0, 1., brColor2.g);
    var emboss2: vec3f = vec3f(2.0 * br2 - c2 - tl2);
    let luminance2: f32 = clamp(0.299 * emboss2.r + 0.587 * emboss2.g + 0.114 * emboss2.b, 0.0, 1.0);
    var ext: vec3f = pal(luminance2 * 1., vec3(.5,0.5,0.5),vec3(0.5,0.5,.5),vec3(1.,1.0,1.0),vec3(0.0,0.33,0.67));
    ext = mix(vec3f(0.), ext, smoothstep(0., .05, luminance2));
    ext *= .09 * (animationUniforms.pulse * .2 + .8);
    
    let vignette = dist * .05;
    
    
    //var color: vec4f = vec4f(base + emboss + vignette + ext, 1.);
    var color: vec4f = vec4f(base + emboss - vignette + specular + ext, 1.);
    
    return color;
}
`;
