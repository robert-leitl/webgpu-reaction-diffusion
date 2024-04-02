// language=C
export const CompositeShader = `
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f
}

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

@group(0) @binding(0) var colorTexSampler : sampler;
@group(0) @binding(1) var colorTex : texture_2d<f32>;

@fragment
fn frag_main(@location(0) uv : vec2f) -> @location(0) vec4f {
    let colorTexSize : vec2f = vec2f(textureDimensions(colorTex));
    let color : vec4f = textureSample(colorTex, colorTexSampler, uv);
    return color;
}
`;
