import * as wgh from '/js/libs/webgpu-utils.module.js';
import { Float16Array } from '/js/libs/float16.js';
import { ReactionDiffusionComputeShader, ReactionDiffusionShaderDispatchSize } from './shader/rd-compute-shader.js';

export class ReactionDiffusionCompute {

    ITERATIONS = 15;

    SCALE = .5;

    constructor(device, viewportSize) {
        this.device = device;

        // create pipeline and bind group layouts
        const module = this.device.createShaderModule({ code: ReactionDiffusionComputeShader });
        const defs = wgh.makeShaderDataDefinitions(ReactionDiffusionComputeShader);
        const pipelineDescriptor = {
            compute: {
                module,
                entryPoint: 'compute_main',
            }
        };
        const descriptors = wgh.makeBindGroupLayoutDescriptors(defs, pipelineDescriptor);
        descriptors[0].entries.push({
            binding: 1,
            storageTexture: { access: 'write-only', format: 'rgba16float' },
            visibility: GPUShaderStage.COMPUTE
        });
        this.bindGroupLayout = this.device.createBindGroupLayout(descriptors[0]);

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });
        this.pipeline = this.device.createComputePipeline({
            label: 'reaction diffusion compute pipeline',
            layout: pipelineLayout,
            ...pipelineDescriptor
        });

        this.inputCanvas = document.createElement('canvas');
        this.inputCanvas.width = viewportSize[0] * this.SCALE;
        this.inputCanvas.height = viewportSize[1] * this.SCALE;
        this.fontSize = Math.min(this.inputCanvas.width, this.inputCanvas.height) / 4;
        this.inputContext = this.inputCanvas.getContext("2d");
        this.inputContext.font = `${this.fontSize}px sans-serif`;
        document.body.appendChild(this.inputCanvas);

        this.drawTime();

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
                    GPUTextureUsage.STORAGE_BINDING |
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

            if (ndx === 0) {
                const imgData = this.inputContext.getImageData(0, 0, width, height);
                const imgNormRGBA = Array.from(imgData.data).map(v => v / 255);
                data = new Float16Array(imgNormRGBA);
                this.device.queue.writeTexture({ texture }, data.buffer, { bytesPerRow: width * 8 }, { width, height });
            } else {
                this.device.queue.writeTexture({ texture }, data.buffer, { bytesPerRow: width * 8 }, { width, height });
            }

            return texture;
        });

        this.dispatches = [
            Math.ceil(width / ReactionDiffusionShaderDispatchSize[0]),
            Math.ceil(height / ReactionDiffusionShaderDispatchSize[1])
        ];
    }

    drawTime() {
        this.inputContext.translate(0, 0);
        this.inputContext.scale(1, 1);
        this.inputContext.rect(0, 0, this.inputCanvas.width, this.inputCanvas.height);
        this.inputContext.fillStyle = '#f00';
        this.inputContext.fill();
        this.inputContext.translate(this.inputCanvas.width / 2, this.inputCanvas.height / 2);
        this.inputContext.scale(1, -1);
        this.inputContext.fillStyle = '#0f0';
        const now = new Date();
        this.inputContext.fillText(`${now.getHours().toString(10).padStart(2, '0')}:${now.getMinutes().toString(10).padStart(2, '0')}:${now.getSeconds().toString(10).padStart(2, '0')}`, - this.fontSize * 2, + this.fontSize * .25);
    }

    createBindGroups() {
        this.swapBindGroups = [
            this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.swapTextures[0].createView() },
                    { binding: 1, resource: this.swapTextures[1].createView() },
                ]
            }),
            this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.swapTextures[1].createView() },
                    { binding: 1, resource: this.swapTextures[0].createView() },
                ]
            })
        ];
    }

    compute(computePassEncoder) {
        computePassEncoder.setPipeline(this.pipeline);

        for(let i = 0; i < this.ITERATIONS; i++) {
            computePassEncoder.setBindGroup(0, this.swapBindGroups[0]);
            computePassEncoder.dispatchWorkgroups(this.dispatches[0], this.dispatches[1]);

            computePassEncoder.setBindGroup(0, this.swapBindGroups[1]);
            computePassEncoder.dispatchWorkgroups(this.dispatches[0], this.dispatches[1]);
        }
    }
}
