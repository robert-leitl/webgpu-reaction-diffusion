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
    base *= 1.2 * (animationUniforms.pulse * .3 + .7);
    
    // emboss effect
    let embossScale = .5;
    let tlColor: vec4f = textureSample(inputTex, inputTexSampler, uv + vec2(-inputTexelSize.x,  inputTexelSize.y) * embossScale);
    let brColor: vec4f = textureSample(inputTex, inputTexSampler, uv + vec2(inputTexelSize.x,  -inputTexelSize.y) * embossScale);
    let c: f32 = smoothstep(0.0, 0.4, input.r);
    let tl: f32 = smoothstep(0.0, 0.4, tlColor.r);
    let br: f32 = smoothstep(0.0, 0.4, brColor.r);
    var emboss: vec3f = vec3f(2.0 * tl - c - br);
    let luminance: f32 = clamp(0.299 * emboss.r + 0.587 * emboss.g + 0.114 * emboss.b, 0.0, 1.0);
    emboss = vec3f(luminance) * 1.5 * (animationUniforms.pulse * .2 + .8);
    var ext: vec3f = 1. - vec3f(br + c + tl) / 3.;
    
    let vignette = length(uv * 2. - 1.) * .05;
    
    
    var color: vec4f = vec4f(base + emboss + vignette, 1.);
    
    return color;
}
`;
