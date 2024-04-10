const kernelSize = 3;
const workgroupSize = [8, 8];
// each thread handles a tile of pixels
const tileSize = [2, 2];
// holds all the pixels needed for one workgroup
const cacheSize = [
    tileSize[0] * workgroupSize[0],
    tileSize[1] * workgroupSize[1]
];
// the cache has to include the boundary pixels needed for a
// valid evaluation of the kernel within the dispatch area
const dispatchSize = [
    cacheSize[0] - Math.max(0, (kernelSize - 1)),
    cacheSize[1] - Math.max(0, (kernelSize - 1)),
];

export const ReactionDiffusionShaderDispatchSize = dispatchSize;

// language=C
export const ReactionDiffusionComputeShader = `

const kernelSize = ${kernelSize};
const dispatchSize = vec2u(${dispatchSize[0]},${dispatchSize[1]});
const tileSize = vec2u(${tileSize[0]},${tileSize[1]});

struct AnimationUniforms {
   pulse: f32,
   pointerVelocity: vec2f,
   pointerPos: vec2f
};

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var seedTex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> animationUniforms: AnimationUniforms;

fn texture2D_bilinear(t: texture_2d<f32>, coord: vec2f, dims: vec2u) -> vec4f {
    let sample: vec2u = vec2u(coord);
    let tl: vec4f = textureLoad(t, clamp(sample, vec2u(1, 1), dims), 0);
    let tr: vec4f = textureLoad(t, clamp(sample + vec2u(1, 0), vec2u(1, 1), dims), 0);
    let bl: vec4f = textureLoad(t, clamp(sample + vec2u(0, 1), vec2u(1, 1), dims), 0);
    let br: vec4f = textureLoad(t, clamp(sample + vec2u(1, 1), vec2u(1, 1), dims), 0);
    let f: vec2f = fract(coord);
    let tA: vec4f = mix(tl, tr, f.x);
    let tB: vec4f = mix(bl, br, f.x);
    return mix(tA, tB, f.y);
}

// the cache for the texture lookups (tileSize * workgroupSize)
var<workgroup> cache: array<array<vec4f, ${cacheSize[0]}>, ${cacheSize[1]}>;

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, 1)
fn compute_main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationID : vec3<u32>,
  @builtin(global_invocation_id) globalInvocationID : vec3<u32>
) {
  // each thread adds a tile of pixels to the workgroups shared memory

  let kernelArea: u32 = kernelSize * kernelSize;

  // the kernel offset (number of pixels next to the center of the kernel) defines
  // the border area next to the dispatched (=work) area that has to be included
  // within the pixel cache
  let kernelOffset: vec2u = vec2((kernelSize - 1) / 2);

  // the local pixel offset of this threads tile
  let tileOffset: vec2u = localInvocationID.xy * tileSize;

  // the global pixel offset of the workgroup
  let dispatchOffset: vec2u = workGroupID.xy * dispatchSize;

  // get texture dimensions
  let dims: vec2u = vec2<u32>(textureDimensions(inputTex, 0));
  
  let aspectFactor: vec2f = vec2f(dims) / f32(max(dims.x, dims.y));

  // add this threads tiles pixels to the cache
  for (var c=0u; c<tileSize.x; c++) {
    for (var r=0u; r<tileSize.y; r++) {
      let local: vec2u = vec2u(c, r) + tileOffset;

      // subtract the kernel offset to include the border pixels needed
      // for the convolution of the kernel within the dispatch (work) area
      var sample: vec2u = dispatchOffset + local - kernelOffset;
      
      sample.x = clamp(sample.x, 1, u32(dims.x));
      sample.y = clamp(sample.y, 1, u32(dims.y));
      
      var sampleCoord: vec2f = vec2f(sample);
      var sampleUv: vec2f = sampleCoord / vec2f(dims);
      sampleCoord -= (sampleUv * 2. - 1.) * 0.01 * (2. * animationUniforms.pulse + 2. + 1.5);
      
      let st = ((sampleUv * 2. - 1.) * aspectFactor) * .5 + .5;
      let pointerPos = (animationUniforms.pointerPos * aspectFactor) * .5 + .5;
      var pointerMask = smoothstep(0.6, 1., 1. - min(1., (distance(st, pointerPos))));
      sampleCoord -= animationUniforms.pointerVelocity * 2.5 * pointerMask;
      
      // perform manual bilinear sampling of the input texture
      let input: vec4f = texture2D_bilinear(inputTex, sampleCoord, dims);
      
      //let input: vec4f = textureLoad(inputTex, sample, 0);
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

  let laplacian: array<f32, 9> = array(
    0.05, 0.20, 0.05,
    0.20, -1.0, 0.20,
    0.05, 0.20, 0.05,
  );

  // run through the whole cache area
  for (var c=0u; c<tileSize.x; c++) {
    for (var r=0u; r<tileSize.y; r++) {
      let local: vec2u = vec2u(c, r) + tileOffset;
      let sample: vec2u = dispatchOffset + local - kernelOffset;

      // only apply the kernel to pixels for which we have all
      // necessary pixels in the cache
      if (all(sample >= bounds.xy) && all(sample < bounds.zw)) {
      
        let uv: vec2f = vec2f(sample) / vec2f(dims);
        
        var st = ((uv * 2. - 1.) * aspectFactor) * .5 + .5;
        let pointerPos = (animationUniforms.pointerPos * aspectFactor) * .5 + .5;
        var pointerMask = smoothstep(0.6, 1., 1. - min(1., (distance(st, pointerPos))));
        pointerMask = pointerMask * length(animationUniforms.pointerVelocity) * 30.;

        // convolution with laplacian kernel
        var lap = vec2f(0);
        let ks: i32 = i32(kernelSize);
        for (var x = 0; x < ks; x++) {
          for (var y = 0; y < ks; y++) {
            var i = vec2i(local) + vec2(x, y) - vec2i(kernelOffset);
            lap += cache[i.y][i.x].xy * laplacian[y * ks + x];
          }
        }
        
        st = (uv * 2. - 1.);
        let dist = mix(dot(st.xx, st.xx), dot(st.yy, st.yy), step(1.3, f32(dims.x) / f32(dims.y)));

        // reaction diffusion calculation
        let cacheValue: vec4f = cache[local.y][local.x];
        let rd0 = cacheValue.xy;
        let dA = 1. - dist * .15;
        var dB = .25 + dist * 0.1 + 0.1 * (animationUniforms.pulse * .5 + .5) - min(0.15, 0.2 * pointerMask);
        dB = max(0.1, dB);
        let feed = 0.065;// * max(0.3, (1. - dist * .7));
        let kill = 0.06 + cacheValue.b * .05 * (animationUniforms.pulse * .3 + .7);
        // calculate result
        let A = rd0.x;
        let B = rd0.y;
        let reaction = A * B * B;
        let rd = vec2f(
          A + (dA * lap.x - reaction + feed * (1. - A)),
          B + (dB * lap.y + reaction - (kill + feed) * B),
        );

        textureStore(outputTex, sample, vec4(rd, 0., 1.0));

        // debug code
        //textureStore(outputTex, sample, vec4(cache[local.y][local.x], 1.0));
        //textureStore(outputTex, sample, vec4(uv, 0., 1.0));
      }
    }
  }
}

`;

