/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;
let isLoading = false;

addEventListener('message', async ({ data }) => {
    const { type, payload } = data;

    switch (type) {
        case 'LOAD_MODEL':
            if (isLoading) return;

            // Release existing session if reloading
            if (session) {
                try {
                    await session.release();
                } catch (e) {
                    console.warn('Failed to release session:', e);
                }
                session = null;
            }

            isLoading = true;
            try {
                // Set wasm paths - must be absolute path for module resolution
                ort.env.wasm.wasmPaths = '/assets/wasm/';

                // CPU-only: WASM is the most reliable and consistent across all devices
                const options: ort.InferenceSession.SessionOptions = {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                };

                console.log('Loading EdgeFace with CPU (WASM) backend');

                session = await ort.InferenceSession.create('/assets/models/edgeface_xxs.onnx', options);
                postMessage({ type: 'MODEL_LOADED', payload: true });
            } catch (e: any) {
                console.error('Failed to load EdgeFace:', e);
                postMessage({ type: 'ERROR', payload: e.message });
            } finally {
                isLoading = false;
            }
            break;

        case 'RECOGNIZE':
            if (!session) {
                const { id } = payload;
                postMessage({ type: 'ERROR', payload: 'Model not loaded', id });
                return;
            }
            try {
                const { imageBuffer, id, useBGR } = payload;
                // imageBuffer is now a Uint8ClampedArray (RGBA) transferred from main thread
                const embedding = await runInference(imageBuffer, useBGR);
                postMessage({ type: 'RESULT', payload: embedding, id });
            } catch (e: any) {
                const { id } = payload;
                postMessage({ type: 'ERROR', payload: e.message, id });
            }
            break;
    }
});

async function runInference(inputData: Uint8ClampedArray, useBGR: boolean = true): Promise<number[]> {
    if (!session) throw new Error('Session null');

    // Expected input: 112x112
    const width = 112;
    const height = 112;
    const totalPixels = width * height;
    const tensorData = new Float32Array(1 * 3 * width * height);

    // --- PRE-PROCESSING: Standard Normalize & CHW Layout ---
    // Convert from Uint8 RGBA to normalized Float32 NCHW in one pass
    // (x - 127.5) / 128.0 (Standard InsightFace Normalization)

    for (let i = 0; i < totalPixels; i++) {
        let r = inputData[i * 4];
        let g = inputData[i * 4 + 1];
        let b = inputData[i * 4 + 2];

        if (useBGR) {
            // Swap for BGR
            const temp = r;
            r = b;
            b = temp;
        }

        // NCHW Layout: R, G, B (Planar)
        // Normalize: (Val - 127.5) / 128.0
        tensorData[i] = (r - 127.5) / 128.0;
        tensorData[i + totalPixels] = (g - 127.5) / 128.0;
        tensorData[i + 2 * totalPixels] = (b - 127.5) / 128.0;
    }

    const feeds: Record<string, ort.Tensor> = {};
    feeds['input'] = new ort.Tensor('float32', tensorData, [1, 3, 112, 112]);

    const results = await session.run(feeds);

    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];

    return Array.from(outputTensor.data as Float32Array);
}
