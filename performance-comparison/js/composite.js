import {CompositeShader} from '/performance-comparison/js/shader/composite-shader.js';
import * as wgh from '/performance-comparison/js/libs/webgpu-utils.module.js';

export class Composite {

    constructor(device, reactionDiffusion) {
        this.device = device;
        this.reactionDiffusion = reactionDiffusion;

        const module = device.createShaderModule({ code: CompositeShader });
        const defs = wgh.makeShaderDataDefinitions(CompositeShader);
        const pipelineLayout = {
            vertex: {
                module: module,
                entryPoint: 'vertex_main',
            },
            fragment: {
                module,
                entryPoint:'frag_main',
                targets: [
                    { format: navigator.gpu.getPreferredCanvasFormat() }
                ]
            },
            primitive: {
                topology: 'triangle-list',
            },
        }
        const descriptors = wgh.makeBindGroupLayoutDescriptors(defs, pipelineLayout);
        const bindGroupLayout = device.createBindGroupLayout(descriptors[0]);

        this.sampler = device.createSampler({
            minFilter: 'linear',
            magFilter: 'linear'
        });
        this.bindGroupLayouts = [bindGroupLayout];

        this.pipeline = device.createRenderPipeline({
            label: 'composite pipeline',
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            ...pipelineLayout
        });

        this.createBindGroups();
    }

    render(renderPassEncoder) {
        renderPassEncoder.setPipeline(this.pipeline);
        renderPassEncoder.setBindGroup(0, this.bindGroup);
        renderPassEncoder.draw(3);
    }

    createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: this.sampler },
                { binding: 1, resource: this.reactionDiffusion.resultStorageTexture.createView() },
            ]
        });
    }
}
