import {CompositeShader} from './shader/composite-shader.js';
import * as wgh from './libs/webgpu-utils.module.js';

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

        // create the animation uniforms view and buffer
        const animationUniformView = wgh.makeStructuredView(defs.uniforms.animationUniforms);
        this.animationUniform = {
            view: animationUniformView,
            buffer: this.device.createBuffer({
                size: animationUniformView.arrayBuffer.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            })
        };

        // the sampler for the input texture
        this.sampler = device.createSampler({
            minFilter: 'linear',
            magFilter: 'linear'
        });

        // create the pipeline
        this.bindGroupLayouts = [bindGroupLayout];
        this.pipeline = device.createRenderPipeline({
            label: 'composite pipeline',
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            ...pipelineLayout
        });

        // initial resize
        this.resize();
    }

    resize() {
        this.createBindGroups();
    }

    render(renderPassEncoder, pulse) {
        this.animationUniform.view.set({ pulse });
        this.device.queue.writeBuffer(this.animationUniform.buffer, 0, this.animationUniform.view.arrayBuffer);

        renderPassEncoder.setPipeline(this.pipeline);
        renderPassEncoder.setBindGroup(0, this.bindGroup);
        renderPassEncoder.draw(3);
    }

    createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: this.animationUniform.buffer } },
                { binding: 1, resource: this.sampler },
                { binding: 2, resource: this.reactionDiffusion.resultStorageTexture.createView() },
            ]
        });
    }
}
