// language=C
export const ReactionDiffusionFragmentShader = `
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

@group(0) @binding(0) var inputTex : texture_2d<f32>;

@fragment
fn frag_main(@location(0) uv : vec2f) -> @location(0) vec4f {
    let texSize : vec2f = vec2f(textureDimensions(inputTex));
    let sample: vec2u = vec2u(floor(uv * texSize));
    
    let laplacian: array<f32, 9> = array(
        0.05, 0.20, 0.05,
        0.20, -1.0, 0.20,
        0.05, 0.20, 0.05,
    );
    
    // convolution with laplacian kernel
    var lap = vec2f(0);
    for (var x = 0; x < 3; x++) {
        for (var y = 0; y < 3; y++) {
            let i = vec2i(sample) + vec2i(x - 1, y - 1);
            lap += textureLoad(inputTex, i, 0).xy * laplacian[y * 3 + x];
        }
    }
    
    // reaction diffusion calculation
    let rd0 = textureLoad(inputTex, sample, 0);
    let dA = .5;
    let dB = .2;
    let feed = 0.063;
    let kill = 0.062;
    // calculate result
    let A = rd0.x;
    let B = rd0.y;
    let reaction = A * B * B;
    let rd = vec2f(
        A + (dA * lap.x - reaction + feed * (1. - A)),
        B + (dB * lap.y + reaction - (kill + feed) * B),
    );
    
    let color = vec4(rd, 0., 1.0);
    
    return color;
}
`;
