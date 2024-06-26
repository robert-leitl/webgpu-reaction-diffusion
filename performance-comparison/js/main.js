import {ReactionDiffusionCompute} from './rd-compute.js';
import {ReactionDiffusionFragment} from './rd-fragment.js';
import {Composite} from './composite.js';
import {TimingHelper} from './utils/timing-helper.js';
import {RollingAverage} from './utils/rolling-average.js';

const canvas = document.getElementById('viewport');
const adapter = await navigator.gpu?.requestAdapter();
const canTimestamp = adapter.features.has('timestamp-query');
const device = await adapter?.requestDevice({
    requiredFeatures: [ ...(canTimestamp ? ['timestamp-query'] : [])],
});
const context = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'premultiplied' });
const viewportSize = [
    Math.max(1, Math.min(window.innerWidth, device.limits.maxTextureDimension2D)),
    Math.max(1, Math.min(window.innerHeight, device.limits.maxTextureDimension2D))
];
canvas.width = viewportSize[0];
canvas.height = viewportSize[1];

const useCompute = false;

const timingHelper = useCompute ? new TimingHelper(device) : new TimingHelper(device, ReactionDiffusionFragment.ITERATIONS * 2);
const rdTime = new RollingAverage(500);
const timeDisplay = document.createElement('span');
timeDisplay.style.position = 'absolute';
timeDisplay.style.top = '0';
timeDisplay.style.left = '0';
timeDisplay.style.fontSize = '1.4em';
timeDisplay.style.color = '#fff';
document.body.appendChild(timeDisplay);

const reactionDiffusion = useCompute ? new ReactionDiffusionCompute(device, viewportSize) : new ReactionDiffusionFragment(device, viewportSize);
const composite = new Composite(device, reactionDiffusion);

const run = t => {

    const commandEncoder = device.createCommandEncoder();

    if (useCompute) {
        const computePassEncoder = timingHelper.beginComputePass(commandEncoder);
        reactionDiffusion.compute(computePassEncoder);
        computePassEncoder.end();
    } else {
        reactionDiffusion.render(commandEncoder, timingHelper);
    }

    const compositePassEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 1, g: 1, b: 1, a: 1},
            loadOp: 'clear',
            storeOp: 'store'
        }],
    });
    composite.render(compositePassEncoder);
    compositePassEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    timingHelper.getResult().then(gpuTime => rdTime.addSample(gpuTime / 1000));
    timeDisplay.innerText = `GPU time ${useCompute ? 'compute shader' : 'fragment shader'}: ${Math.round(rdTime.value)} µs`;

    requestAnimationFrame(t => run(t));
}

run(0);
