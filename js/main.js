import {ReactionDiffusionCompute} from './rd-compute.js';
import {ReactionDiffusionFragment} from './rd-fragment.js';
import {Composite} from './composite.js';
import {TimingHelper} from './utils/timing-helper.js';
import {RollingAverage} from './utils/rolling-average.js';


const canvas = document.getElementById('viewport');

// check compatibility
const hasWebGPU = !!navigator.gpu;
if (!hasWebGPU) {
    const noWebGPUMessage = document.querySelector('#no-webgpu');
    noWebGPUMessage.style.display = '';
    canvas.style.display = 'none';
} else {
    const adapter = await navigator.gpu?.requestAdapter();
    const canTimestamp = adapter.features.has('timestamp-query');
    const device = await adapter?.requestDevice({
        requiredFeatures: [ ...(canTimestamp ? ['timestamp-query'] : [])],
    });
    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'premultiplied' });
    const pixelRatio = Math.min(2, window.devicePixelRatio);
    const viewportSize = [
        Math.max(1, Math.min(pixelRatio * window.innerWidth, device.limits.maxTextureDimension2D)),
        Math.max(1, Math.min(pixelRatio * window.innerHeight, device.limits.maxTextureDimension2D))
    ];
    canvas.width = viewportSize[0];
    canvas.height = viewportSize[1];
    let timeMS = 0;

    const useCompute = true;

    const timingHelper = useCompute ? new TimingHelper(device) : new TimingHelper(device, ReactionDiffusionFragment.ITERATIONS * 2);
    const rdTime = new RollingAverage(500);

    const reactionDiffusion = useCompute ? new ReactionDiffusionCompute(device, viewportSize) : new ReactionDiffusionFragment(device, viewportSize);
    const composite = new Composite(device, reactionDiffusion);

    const run = t => {

        timeMS += Math.min(32, t);

        const dateTimeMS = new Date().getTime() + 800;
        const pulse = Math.sin(2 * Math.PI * dateTimeMS * .001);

        const commandEncoder = device.createCommandEncoder();

        if (useCompute) {
            const computePassEncoder = timingHelper.beginComputePass(commandEncoder);
            reactionDiffusion.compute(computePassEncoder, pulse);
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
        composite.render(compositePassEncoder, pulse);
        compositePassEncoder.end();

        device.queue.submit([commandEncoder.finish()]);

        timingHelper.getResult().then(gpuTime => rdTime.addSample(gpuTime / 1000));

        requestAnimationFrame(t => run(t));
    }

    run(0);
}


