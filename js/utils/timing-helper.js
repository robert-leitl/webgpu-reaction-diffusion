// Credits: https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html
function assert(cond, msg = '') {
    if (!cond) {
        throw new Error(msg);
    }
}

export class TimingHelper {
    #canTimestamp;
    #device;
    #querySet;
    #resolveBuffer;
    #resultBuffer;
    #resultBuffers = [];
    // state can be 'free', 'need resolve', 'wait for result'
    #state = 'free';
    #passCount = 1;

    constructor(device, passCount = 1) {
        this.#device = device;
        this.#passCount = passCount;
        this.#canTimestamp = device.features.has('timestamp-query');
        if (!this.#canTimestamp) return;

        this.#querySet = device.createQuerySet({
            type: 'timestamp',
            count: this.#passCount * 2,
        });
        this.#resolveBuffer = device.createBuffer({
            size: this.#querySet.count * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
    }

    #beginTimestampPass(encoder, fnName, descriptor, passIndex) {
        if (this.#canTimestamp) {
            assert(this.#state === 'free', 'state not free');
            this.#state = passIndex !== this.#passCount - 1 ? 'free' : 'need resolve';

            const pass = encoder[fnName]({
                ...descriptor,
                ...{
                    timestampWrites: {
                        querySet: this.#querySet,
                        beginningOfPassWriteIndex: passIndex * 2,
                        endOfPassWriteIndex: passIndex * 2 + 1,
                    },
                },
            });

            const resolve = () => this.#resolveTiming(encoder, passIndex);
            pass.end = (function(origFn) {
                return function() {
                    origFn.call(this);
                    resolve();
                };
            })(pass.end);

            return pass;
        } else {
            return encoder[fnName](descriptor);
        }
    }

    beginRenderPass(encoder, descriptor = {}, passIndex = 0) {
        return this.#beginTimestampPass(encoder, 'beginRenderPass', descriptor, passIndex);
    }

    beginComputePass(encoder, descriptor = {}, passIndex = 0) {
        return this.#beginTimestampPass(encoder, 'beginComputePass', descriptor, passIndex);
    }

    #resolveTiming(encoder, passIndex) {
        if (!this.#canTimestamp || passIndex !== this.#passCount - 1) {
            return;
        }
        assert(this.#state === 'need resolve', 'must call addTimestampToPass');
        this.#state = 'wait for result';

        this.#resultBuffer = this.#resultBuffers.pop() || this.#device.createBuffer({
            size: this.#resolveBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        encoder.resolveQuerySet(this.#querySet, 0, this.#querySet.count, this.#resolveBuffer, 0);
        encoder.copyBufferToBuffer(this.#resolveBuffer, 0, this.#resultBuffer, 0, this.#resultBuffer.size);
    }

    async getResult() {
        if (!this.#canTimestamp) {
            return 0;
        }
        assert(this.#state === 'wait for result', 'must call resolveTiming');
        this.#state = 'free';

        const resultBuffer = this.#resultBuffer;
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const times = new BigInt64Array(resultBuffer.getMappedRange());

        let durationSum = 0;
        for(let i=0; i<this.#passCount; i++) {
            durationSum += Number(times[i * 2 + 1] - times[i * 2]);
        }

        resultBuffer.unmap();
        this.#resultBuffers.push(resultBuffer);
        return durationSum;
    }
}
