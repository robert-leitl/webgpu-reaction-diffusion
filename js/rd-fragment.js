import * as wgh from '/js/libs/webgpu-utils.module.js';
import { Float16Array } from '/js/libs/float16.js';
import { ReactionDiffusionFragmentShader } from './shader/rd-fragment-shader.js';

export class ReactionDiffusionFragment {

    static ITERATIONS = 15;

    SCALE = .5;

    constructor(device, viewportSize) {
        this.device = device;

        // create pipeline and bind group layouts
        const module = this.device.createShaderModule({ code: ReactionDiffusionFragmentShader });
        const defs = wgh.makeShaderDataDefinitions(ReactionDiffusionFragmentShader);
        const pipelineDescriptor = {
            vertex: {
                module: module,
                entryPoint: 'vertex_main',
            },
            fragment: {
                module,
                entryPoint:'frag_main',
                targets: [{ format: 'rgba16float' }]
            },
            primitive: {
                topology: 'triangle-list',
            },
        };
        const descriptors = wgh.makeBindGroupLayoutDescriptors(defs, pipelineDescriptor);
        this.bindGroupLayout = this.device.createBindGroupLayout(descriptors[0]);

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });
        this.pipeline = this.device.createRenderPipeline({
            label: 'reaction diffusion fragment pipeline',
            layout: pipelineLayout,
            ...pipelineDescriptor
        });

        this.init(viewportSize[0] * this.SCALE, viewportSize[1] * this.SCALE);
    }

    init(width, height) {
        this.createTextures(Math.round(width), Math.round(height));
        this.createBindGroups();
    }

    get resultStorageTexture() {
        return this.swapTextures[0];
    }

    createTextures(width, height) {
        if (this.swapTextures) {
            this.swapTextures.forEach(texture => texture.destroy());
        }

        this.swapTextures = new Array(2).fill(null).map((v, ndx) => {
            const texture = this.device.createTexture({
                size: { width, height },
                format: 'rgba16float',
                usage:
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            });

            const w = width;
            const h = height;
            let data;
            const rgba = new Array(w * h * 4).fill(0);
            const s = 20;
            const bx = [w / 2 - s, w / 2 + s];
            const by = [h / 2 - s, h / 2 + s];
            for(let x=0; x<w; x++) {
                for(let y=0; y<h; y++) {
                    const v = x > bx[0] && x < bx[1] && y > by[0] && y < by[1];
                    rgba[(x + y * w) * 4 + 0] = ndx === 0 && !v ? 1 : 0;
                    rgba[(x + y * w) * 4 + 1] = ndx === 0 && v ? 1 : 0;
                    rgba[(x + y * w) * 4 + 2] = 0;
                    rgba[(x + y * w) * 4 + 3] = 1;
                }
            }
            data = new Float16Array(rgba);

            this.device.queue.writeTexture({ texture }, data.buffer, { bytesPerRow: width * 8 }, { width, height });

            return texture;
        });

        this.swapTextureViews = this.swapTextures.map(texture => texture.createView());
    }

    createBindGroups() {
        this.swapBindGroups = [
            this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.swapTextures[0].createView() },
                ]
            }),
            this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.swapTextures[1].createView() },
                ]
            })
        ];
    }

    render(commandEncoder, timingHelper) {
        for(let i = 0; i < ReactionDiffusionFragment.ITERATIONS; i++) {
            let renderPassEncoder = timingHelper.beginRenderPass(commandEncoder, {
                colorAttachments: [{
                    view: this.swapTextureViews[1],
                    loadOp: 'load',
                    storeOp: 'store'
                }],
            }, i * 2);
            renderPassEncoder.setPipeline(this.pipeline);
            renderPassEncoder.setBindGroup(0, this.swapBindGroups[0]);
            renderPassEncoder.draw(3);
            renderPassEncoder.end();

            renderPassEncoder = timingHelper.beginRenderPass(commandEncoder, {
                colorAttachments: [{
                    view: this.swapTextureViews[0],
                    loadOp: 'load',
                    storeOp: 'store'
                }],
            }, i * 2 + 1);
            renderPassEncoder.setPipeline(this.pipeline);
            renderPassEncoder.setBindGroup(0, this.swapBindGroups[1]);
            renderPassEncoder.draw(3);
            renderPassEncoder.end();
        }
    }
}
