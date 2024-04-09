import {ReactionDiffusionCompute} from '/js/rd-compute.js';
import {ReactionDiffusionFragment} from '/js/rd-fragment.js';
import {Composite} from '/js/composite.js';
import {TimingHelper} from '/js/utils/timing-helper.js';
import {RollingAverage} from '/js/utils/rolling-average.js';

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

const useCompute = true;

const timingHelper = useCompute ? new TimingHelper(device) : new TimingHelper(device, ReactionDiffusionFragment.ITERATIONS * 2);
const rdTime = new RollingAverage(500);

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

    console.log(rdTime.value)

    requestAnimationFrame(t => run(t));
}

run(0);
