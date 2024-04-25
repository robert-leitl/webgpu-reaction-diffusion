import * as wgh from './libs/webgpu-utils.module.js';
import { Float16Array } from './libs/float16.js';
import { ReactionDiffusionComputeShader, ReactionDiffusionShaderDispatchSize } from './shader/rd-compute-shader.js';

export class ReactionDiffusionCompute {

    // these are the iterations of the simulation during one frame (the more iterations, the faster the simulation)
    ITERATIONS = 10;

    // this is the scaling factor for the simulation textures (one quarter the size of the canvas)
    SCALE = .25;

    // interaction data
    pointer = {
        position: [0, 0],
        followerPosition: undefined,
        followerVelocity: [0, 0]
    };

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
        // the storage texture descriptor has to be created manually
        descriptors[0].entries.push({
            binding: 1,
            storageTexture: { access: 'write-only', format: 'rgba16float' },
            visibility: GPUShaderStage.COMPUTE
        });
        this.bindGroupLayout = this.device.createBindGroupLayout(descriptors[0]);

        // create the compute pipeline
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });
        this.pipeline = this.device.createComputePipeline({
            label: 'reaction diffusion compute pipeline',
            layout: pipelineLayout,
            ...pipelineDescriptor
        });

        // create the animation uniform view and buffer
        const animationUniformView = wgh.makeStructuredView(defs.uniforms.animationUniforms);
        this.animationUniform = {
            view: animationUniformView,
            buffer: this.device.createBuffer({
                size: animationUniformView.arrayBuffer.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            })
        };

        this.initEvents();

        // initial resize (this also creates the textures and bind groups)
        this.resize(viewportSize[0], viewportSize[1]);
    }

    initEvents() {
        document.body.addEventListener('pointerdown', e => {
            this.pointer.position = this.getNormalizedPointerCoords(e.clientX, e.clientY);
            this.pointer.followerPosition = [...this.pointer.position];
        });

        document.body.addEventListener('pointermove', e => {
            this.pointer.position = this.getNormalizedPointerCoords(e.clientX, e.clientY);
            if (!this.pointer.followerPosition) this.pointer.followerPosition = [...this.pointer.position];
        });
    }

    resize(width, height) {
        this.width = Math.round(width * this.SCALE);
        this.height = Math.round(height * this.SCALE);
        this.aspect = this.width / this.height;

        this.inputCanvas = document.createElement('canvas');
        this.inputCanvas.width = this.width;
        this.inputCanvas.height = this.height;
        this.fontSize = Math.max(80, Math.min(this.inputCanvas.width, this.inputCanvas.height) / 3);
        this.inputContext = this.inputCanvas.getContext("2d", { willReadFrequently: true });
        this.inputContext.font = `${this.fontSize}px "Syne Mono"`;
        this.letterWidth = this.inputContext.measureText('0').width;
        this.letterHeight = this.inputContext.measureText('0').actualBoundingBoxAscent;

        this.createTextures(this.width, this.height);
        this.createBindGroups();
    }

    get resultStorageTexture() {
        return this.swapTextures[0];
    }

    createTextures(width, height) {
        // dispose existing textures
        if (this.swapTextures) {
            this.swapTextures.forEach(texture => texture.destroy());
            this.seedTexture.destroy();
        }

        // the seed texture to copy the canvas image to
        this.seedTexture = this.device.createTexture({
            size: { width, height },
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.TEXTURE_BINDING
        });

        // the textures for the actual reaction diffusion ping-pong computation
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

            if (ndx === 0) {
                // initially add the clock letters as chemical A
                const imgData = this.drawTime(true);
                const imgNormRGBA = Array.from(imgData.data).map(v => v / 255);
                const data = new Float16Array(imgNormRGBA);
                this.device.queue.writeTexture({ texture }, data.buffer, { bytesPerRow: width * 8 }, { width, height });
            } else {
                const data = new Float16Array(new Array(width * height * 4).fill(0));
                this.device.queue.writeTexture({ texture }, data.buffer, { bytesPerRow: width * 8 }, { width, height });
            }

            return texture;
        });

        this.dispatches = [
            Math.ceil(width / ReactionDiffusionShaderDispatchSize[0]),
            Math.ceil(height / ReactionDiffusionShaderDispatchSize[1])
        ];
    }

    drawTime(init = false) {
        const ctx = this.inputContext;
        ctx.resetTransform();
        ctx.clearRect(0, 0, this.width, this.height);

        ctx.scale(1, 1);
        ctx.rect(0, 0, this.width, this.height);
        ctx.fillStyle = init ? '#0f0' : 'rgba(0, 0, 0, 0)';
        ctx.fill();

        ctx.translate(this.width / 2, this.height / 2);
        ctx.scale(1, -1);
        ctx.fillStyle = '#f00';
        const now = new Date();

        if (this.aspect > 1.3) {
            ctx.fillText(`${now.getHours().toString(10).padStart(2, '0')}:${now.getMinutes().toString(10).padStart(2, '0')}:${now.getSeconds().toString(10).padStart(2, '0')}`,
                - this.letterWidth * 4,
                + this.letterHeight * .5);
        } else {
            const lineHeight = 1.25;
            const x = - this.letterWidth;
            const y = - this.letterHeight * .5 + (this.letterHeight * (1 - lineHeight));
            const rowHeight = this.letterHeight * lineHeight;
            ctx.fillText(`${now.getHours().toString(10).padStart(2, '0')}`,
                x,
                y);
            ctx.fillText(`${now.getMinutes().toString(10).padStart(2, '0')}`,
                x,
                y + rowHeight);
            ctx.fillText(`${now.getSeconds().toString(10).padStart(2, '0')}`,
                x,
                y + rowHeight * 2);
        }

        this.lastTime = init ? undefined : now;
        return this.inputContext.getImageData(0, 0, this.width, this.height);
    }

    createBindGroups() {
        this.swapBindGroups = [
            this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.swapTextures[0].createView() },
                    { binding: 1, resource: this.swapTextures[1].createView() },
                    { binding: 2, resource: this.seedTexture.createView() },
                    { binding: 3, resource: { buffer: this.animationUniform.buffer }},
                ]
            }),
            this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.swapTextures[1].createView() },
                    { binding: 1, resource: this.swapTextures[0].createView() },
                    { binding: 2, resource: this.seedTexture.createView() },
                    { binding: 3, resource: { buffer: this.animationUniform.buffer }},
                ]
            })
        ];
    }

    compute(computePassEncoder, pulse) {
        this.animatePointer();

        // update animation uniforms
        this.animationUniform.view.set({ pulse });
        this.animationUniform.view.set({ pointerVelocity: this.pointer.followerVelocity });
        this.animationUniform.view.set({ pointerPos: this.pointer.followerPosition ? this.pointer.followerPosition : [0, 0] });
        this.device.queue.writeBuffer(this.animationUniform.buffer, 0, this.animationUniform.view.arrayBuffer);

        computePassEncoder.setPipeline(this.pipeline);

        // redraw the clock only if needed
        if (!this.lastTime || this.lastTime.getSeconds() !== new Date().getSeconds()) {
            const imgData = this.drawTime();
            const seedData = new Uint8Array(Array.from(imgData.data));
            this.device.queue.writeTexture({ texture: this.seedTexture }, seedData.buffer, { bytesPerRow: this.width * 4 }, { width: this.width, height: this.height });
        }

        // reaction diffusion ping-pong loop ;)
        for(let i = 0; i < this.ITERATIONS; i++) {
            computePassEncoder.setBindGroup(0, this.swapBindGroups[0]);
            computePassEncoder.dispatchWorkgroups(this.dispatches[0], this.dispatches[1]);

            computePassEncoder.setBindGroup(0, this.swapBindGroups[1]);
            computePassEncoder.dispatchWorkgroups(this.dispatches[0], this.dispatches[1]);
        }
    }

    animatePointer() {
        if (!this.pointer.followerPosition) return;

        const prevPointerFollower = [...this.pointer.followerPosition];
        this.pointer.followerPosition[0] += (this.pointer.position[0] - this.pointer.followerPosition[0]) / 18;
        this.pointer.followerPosition[1] += (this.pointer.position[1] - this.pointer.followerPosition[1]) / 18;
        this.pointer.followerVelocity[0] = (this.pointer.followerPosition[0] - prevPointerFollower[0]);
        this.pointer.followerVelocity[1] = (this.pointer.followerPosition[1] - prevPointerFollower[1]);
    }

    getNormalizedPointerCoords(clientX, clientY) {
        return [
            (clientX / window.innerWidth) * 2 - 1,
            (1 - clientY / window.innerHeight) * 2 - 1
        ];
    }
}
