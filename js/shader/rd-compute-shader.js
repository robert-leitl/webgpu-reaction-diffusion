// reaction diffusion requires a 3x3 kernel
const kernelSize = 3;

// keep the number of threads to the recommended threshold (64)
const workgroupSize = [8, 8];

// each thread handles a tile of pixels
const tileSize = [2, 2];

// holds all the pixels needed for one workgroup
const cacheSize = [
    tileSize[0] * workgroupSize[0], // 16
    tileSize[1] * workgroupSize[1]  // 16
];

// the cache has to include the boundary pixels needed for a
// valid evaluation of the kernel within the dispatch area
const dispatchSize = [
    cacheSize[0] - (kernelSize - 1), // 14
    cacheSize[1] - (kernelSize - 1), // 14
];

export const ReactionDiffusionShaderDispatchSize = dispatchSize;

// language=C
export const ReactionDiffusionComputeShader = `

const kernelSize = ${kernelSize};
const dispatchSize = vec2u(${dispatchSize[0]},${dispatchSize[1]});
const tileSize = vec2u(${tileSize[0]},${tileSize[1]});
const laplacian: array<f32, 9> = array(
    0.05, 0.20, 0.05,
    0.20, -1.0, 0.20,
    0.05, 0.20, 0.05,
);

struct AnimationUniforms {
   pulse: f32,
   pointerVelocity: vec2f,
   pointerPos: vec2f
};

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var seedTex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> animationUniforms: AnimationUniforms;

// based on: https://community.khronos.org/t/manual-bilinear-filter/58504
fn texture2D_bilinear(t: texture_2d<f32>, coord: vec2f, dims: vec2u) -> vec4f {
    let f: vec2f = fract(coord);
    let sample: vec2u = vec2u(coord + (0.5 - f));
    let tl: vec4f = textureLoad(t, clamp(sample, vec2u(1, 1), dims), 0);
    let tr: vec4f = textureLoad(t, clamp(sample + vec2u(1, 0), vec2u(1, 1), dims), 0);
    let bl: vec4f = textureLoad(t, clamp(sample + vec2u(0, 1), vec2u(1, 1), dims), 0);
    let br: vec4f = textureLoad(t, clamp(sample + vec2u(1, 1), vec2u(1, 1), dims), 0);
    let tA: vec4f = mix(tl, tr, f.x);
    let tB: vec4f = mix(bl, br, f.x);
    return mix(tA, tB, f.y);
}

// the cache for the texture lookups (tileSize * workgroupSize).
// each thread adds a tile of pixels to the workgroups shared memory
var<workgroup> cache: array<array<vec4f, ${cacheSize[0]}>, ${cacheSize[1]}>;

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, 1)
fn compute_main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationID : vec3<u32>,
  @builtin(global_invocation_id) globalInvocationID : vec3<u32>
) {
  // the kernel offset (number of pixels next to the center of the kernel) defines
  // the border area next to the dispatched (=work) area that has to be included
  // within the pixel cache
  let kernelOffset: vec2u = vec2((kernelSize - 1) / 2);

  // the local pixel offset of this threads tile (the tile inside the workgroup)
  let tileOffset: vec2u = localInvocationID.xy * tileSize;

  // the global pixel offset of the workgroup
  let dispatchOffset: vec2u = workGroupID.xy * dispatchSize;

  // get texture dimensions
  let dims: vec2u = vec2<u32>(textureDimensions(inputTex, 0));
  let aspectFactor: vec2f = vec2f(dims) / f32(max(dims.x, dims.y));

  // add the pixels of this thread's tiles to the cache
  for (var c=0u; c<tileSize.x; c++) {
    for (var r=0u; r<tileSize.y; r++) {
      let local: vec2u = vec2u(c, r) + tileOffset;

      // subtract the kernel offset to include the border pixels needed
      // for the convolution of the kernel within the dispatch (work) area
      var sample: vec2u = dispatchOffset + local - kernelOffset;
      
      // clamp to edges
      sample.x = clamp(sample.x, 1, u32(dims.x));
      sample.y = clamp(sample.y, 1, u32(dims.y));
      
      // convert to uv space
      var sampleCoord: vec2f = vec2f(sample);
      var sampleUv: vec2f = sampleCoord / vec2f(dims);
      
      // move the sample away from the center (center pulse motion)
      sampleCoord -= (sampleUv * 2. - 1.) * 0.01 * (2. * animationUniforms.pulse + 2. + 1.5);
      
      // move the sample by the pointer velocity
      let st = ((sampleUv * 2. - 1.) * aspectFactor) * .5 + .5;
      let pointerPos = (animationUniforms.pointerPos * aspectFactor) * .5 + .5;
      var pointerMask = smoothstep(0.6, 1., 1. - min(1., (distance(st, pointerPos))));
      sampleCoord -= animationUniforms.pointerVelocity * 2.5 * pointerMask;
      
      // perform manual bilinear sampling of the input texture
      let input: vec4f = texture2D_bilinear(inputTex, sampleCoord, dims);
      
      // create the combined value: R = chemical A, G = chemical B, B = seed value (clock text)
      var value: vec4f = vec4f(input.rg, vec2f(0.));
      let seed: vec4f = textureLoad(seedTex, sample, 0);
      value.b = seed.r;
      
      cache[local.y][local.x] = value;
    }
  }

  workgroupBarrier();

  // global pixel bounds within the application of the kernel is valid
  let bounds: vec4u = vec4u(
    dispatchOffset,
    min(dims, dispatchOffset + dispatchSize)
  );

  // perform reaction diffusion for every pixel of this tile
  for (var c=0u; c<tileSize.x; c++) {
    for (var r=0u; r<tileSize.y; r++) {
      let local: vec2u = vec2u(c, r) + tileOffset;
      let sample: vec2u = dispatchOffset + local - kernelOffset;

      // only apply the kernel to pixels for which we have all
      // necessary pixels in the cache
      if (all(sample >= bounds.xy) && all(sample < bounds.zw)) {
      
        // get centered uv coords
        let uv: vec2f = (2. * vec2f(sample) / vec2f(dims)) - 1.;

        // convolution with laplacian kernel
        var lap = vec2f(0);
        let ks: i32 = i32(kernelSize);
        for (var x = 0; x < ks; x++) {
          for (var y = 0; y < ks; y++) {
            var i = vec2i(local) + vec2(x, y) - vec2i(kernelOffset);
            lap += cache[i.y][i.x].xy * laplacian[y * ks + x];
          }
        }
        
        // create a pointer mask value to influence the dB value
        var st = (uv * aspectFactor) * .5 + .5;
        let pointerPos = (animationUniforms.pointerPos * aspectFactor) * .5 + .5;
        var pointerMask = smoothstep(0.6, 1., 1. - min(1., (distance(st, pointerPos))));
        pointerMask = pointerMask * length(animationUniforms.pointerVelocity) * 30.;
        
        // the dA and dB values are also influenced by the horizontal or vertical distance from the center
        let dist = mix(dot(uv.xx, uv.xx), dot(uv.yy, uv.yy), step(1.4, f32(dims.x) / f32(dims.y)));

        // reaction diffusion parameters
        let cacheValue: vec4f = cache[local.y][local.x];
        let dA = 1. - dist * .15;
        var dB = .25 + dist * 0.1;
        dB = dB + 0.1 * (animationUniforms.pulse * .5 + .5); // apply pulse motion
        dB = dB - min(0.15, 0.2 * pointerMask); // apply pointer mask
        dB = max(0.1, dB); // prevent instable values
        let feed = 0.065;
        var kill = 0.06;
        // increase the kill param in the areas covered by the clock letters
        kill = kill + cacheValue.b * .05 * (animationUniforms.pulse * .3 + .7);
        
        // reaction diffusion calculation
        let rd0 = cacheValue.xy;
        let A = rd0.x;
        let B = rd0.y;
        let reaction = A * B * B;
        let rd = vec2f(
          A + (dA * lap.x - reaction + feed * (1. - A)),
          B + (dB * lap.y + reaction - (kill + feed) * B),
        );

        textureStore(outputTex, sample, vec4(rd, 0., 1.0));
      }
    }
  }
}
`;

