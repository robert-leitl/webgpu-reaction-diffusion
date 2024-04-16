import {ReactionDiffusionCompute} from './rd-compute.js';
import {Composite} from './composite.js';

const hasWebGPU = !!navigator.gpu;
let canvas, device, webGPUContext, viewportSize, timeMS = 0, reactionDiffusion, composite;

async function init() {
    if (!hasWebGPU) {
        const noWebGPUMessage = document.querySelector('#no-webgpu');
        noWebGPUMessage.style.display = '';
        return;
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    const adapter = await navigator.gpu?.requestAdapter();
    device = await adapter?.requestDevice();

    canvas = document.getElementById('viewport');
    webGPUContext = canvas.getContext('webgpu');
    webGPUContext.configure({ device, format });

    updateViewportSize();

    reactionDiffusion = new ReactionDiffusionCompute(device, viewportSize);
    composite = new Composite(device, reactionDiffusion);

    const observer = new ResizeObserver(() => resize());
    observer.observe(canvas);

    run(0);
}

function updateViewportSize() {
    const pixelRatio = Math.min(2, window.devicePixelRatio);
    viewportSize = [
        Math.max(1, Math.min(window.innerWidth * pixelRatio, device.limits.maxTextureDimension2D)),
        Math.max(1, Math.min(window.innerHeight * pixelRatio, device.limits.maxTextureDimension2D))
    ];
}

function resize() {
    updateViewportSize();

    canvas.width = viewportSize[0];
    canvas.height = viewportSize[1];

    reactionDiffusion.resize(viewportSize[0], viewportSize[1]);
    composite.resize();
}

function run(t) {
    timeMS += Math.min(32, t);

    // create the global pulse animation value: sin with one cycle per second
    const dateTimeMS = new Date().getTime() + 800;
    const pulse = Math.sin(2 * Math.PI * dateTimeMS * .001);

    const commandEncoder = device.createCommandEncoder();

    // update the reaction diffusion compute
    const computePassEncoder = commandEncoder.beginComputePass(commandEncoder);
    reactionDiffusion.compute(computePassEncoder, pulse);
    computePassEncoder.end();

    // render the composite result
    const compositePassEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: webGPUContext.getCurrentTexture().createView(),
            clearValue: { r: 1, g: 1, b: 1, a: 1},
            loadOp: 'clear',
            storeOp: 'store'
        }],
    });
    composite.render(compositePassEncoder, pulse);
    compositePassEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(t => run(t));
}

// wait until the webfont is loaded
const font = new FontFace('Syne Mono', 'url(assets/syne-mono-subset.woff), url(assets/syne-mono-subset.ttf)');
document.fonts.add(font);
font.load().then(() => document.fonts.ready.then(async () => await init()));

