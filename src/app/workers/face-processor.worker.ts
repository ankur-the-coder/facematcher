/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;
let isLoading = false;

addEventListener('message', async ({ data }) => {
    const { type, payload } = data;

    switch (type) {
        case 'LOAD_MODEL':
            if (!session && !isLoading) {
                isLoading = true;
                try {
                    // Set wasm paths - must be absolute path for module resolution
                    ort.env.wasm.wasmPaths = '/assets/wasm/';

                    session = await ort.InferenceSession.create('/assets/models/edgeface_xxs.onnx', {
                        executionProviders: ['wasm'],
                        graphOptimizationLevel: 'all'
                    });
                    postMessage({ type: 'MODEL_LOADED', payload: true });
                } catch (e: any) {
                    console.error('Failed to load EdgeFace:', e);
                    postMessage({ type: 'ERROR', payload: e.message });
                } finally {
                    isLoading = false;
                }
            }
            break;

        case 'RECOGNIZE':
            if (!session) {
                // Return error with ID if possible, but here we just post to generic error if ID missing
                // Actually payload should have id now.
                const { id } = payload;
                postMessage({ type: 'ERROR', payload: 'Model not loaded', id });
                return;
            }
            try {
                const { imageBuffer, id, useBGR } = payload;
                // payload: { imageBuffer, width, height, id, useBGR }

                const embedding = await runInference(imageBuffer, useBGR);
                postMessage({ type: 'RESULT', payload: embedding, id });
            } catch (e: any) {
                const { id } = payload;
                postMessage({ type: 'ERROR', payload: e.message, id });
            }
            break;
    }
});

async function runInference(inputData: Float32Array, useBGR: boolean = true): Promise<number[]> {
    if (!session) throw new Error('Session null');

    // Input processing: Convert RGBA/RGB to NCHW Plane format
    // Expected input: 112x112
    const width = 112;
    const height = 112;
    const totalPixels = width * height;
    const tensorData = new Float32Array(1 * 3 * width * height);

    // --- PRE-PROCESSING: Histogram Equalization (Y Channel only) ---
    // 1. Compute Histogram
    const histogram = new Uint32Array(256).fill(0);
    for (let i = 0; i < totalPixels; i++) {
        const r = inputData[i * 4];
        const g = inputData[i * 4 + 1];
        const b = inputData[i * 4 + 2];
        const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        histogram[lum]++;
    }

    // 2. Compute CDF (Cumulative Distribution Function)
    const cdf = new Uint32Array(256);
    let sum = 0;
    for (let i = 0; i < 256; i++) {
        sum += histogram[i];
        cdf[i] = sum;
    }

    // 3. Normalize CDF
    const cdfMin = cdf.find(val => val > 0) || 0;
    const map = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
        map[i] = Math.round(((cdf[i] - cdfMin) / (totalPixels - cdfMin)) * 255);
    }

    // 4. Apply Equalization & Normalize for Model
    for (let i = 0; i < totalPixels; i++) {
        let r = inputData[i * 4];
        let g = inputData[i * 4 + 1];
        let b = inputData[i * 4 + 2];

        // Convert to YCbCr approx to apply offset to RGB
        const y = 0.299 * r + 0.587 * g + 0.114 * b;
        const newY = map[Math.round(y)];
        const ratio = (y > 0) ? (newY / y) : 1;

        r = Math.min(255, Math.max(0, r * ratio));
        g = Math.min(255, Math.max(0, g * ratio));
        b = Math.min(255, Math.max(0, b * ratio));

        if (useBGR) {
            // Swap for BGR
            const temp = r;
            r = b;
            b = temp;
        }

        // NCHW Layout: R, G, B (After optional swap)
        // If BGR, then R-plane holds B, etc.
        tensorData[i] = (r - 127.5) / 127.5;
        tensorData[i + totalPixels] = (g - 127.5) / 127.5;
        tensorData[i + 2 * totalPixels] = (b - 127.5) / 127.5;
    }

    const feeds: Record<string, ort.Tensor> = {};
    feeds['input'] = new ort.Tensor('float32', tensorData, [1, 3, 112, 112]);

    const results = await session.run(feeds);

    // Output usually named 'embedding', 'output', 'features' etc.
    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];

    return Array.from(outputTensor.data as Float32Array);
}
