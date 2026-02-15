import { Injectable, signal } from '@angular/core';

export interface KnownFace {
    name: string;
    embedding: number[];
    imageUrl?: string;
    timestamp: number;
}

@Injectable({
    providedIn: 'root'
})
export class FaceRecognitionService {
    private worker: Worker | null = null;
    isModelLoaded = signal<boolean>(false);
    knownFaces = signal<KnownFace[]>([]);
    useBGR = signal<boolean>(false); // Default to RGB for better results

    private pendingRequests = new Map<string, { resolve: (v: any) => void, reject: (e: any) => void }>();

    constructor() {
        this.initWorker();
        this.loadKnownFaces();
    }

    private initWorker() {
        if (typeof Worker !== 'undefined') {
            this.worker = new Worker(new URL('../../workers/face-processor.worker', import.meta.url));

            this.worker.onmessage = ({ data }) => {
                const { type, payload, id } = data;
                switch (type) {
                    case 'MODEL_LOADED':
                        this.isModelLoaded.set(payload);
                        console.log('EdgeFace Model Loaded via Worker (CPU/WASM)');
                        break;
                    case 'RESULT':
                        if (id && this.pendingRequests.has(id)) {
                            this.pendingRequests.get(id)!.resolve(payload);
                            this.pendingRequests.delete(id);
                        }
                        break;
                    case 'ERROR':
                        console.error('Worker Error:', payload);
                        if (id && this.pendingRequests.has(id)) {
                            this.pendingRequests.get(id)!.reject(payload);
                            this.pendingRequests.delete(id);
                        }
                        break;
                }
            };

            // Load Model (CPU-only)
            this.worker.postMessage({ type: 'LOAD_MODEL' });
        } else {
            console.error('Web Workers are not supported in this environment.');
        }
    }

    /**
     * Send raw RGBA pixel data to the worker for embedding extraction.
     * Uses Transferable to zero-copy the buffer to the worker thread.
     */
    public getEmbedding(imageData: ImageData): Promise<number[]> {
        if (!this.worker) return Promise.reject('No Worker');

        return new Promise((resolve, reject) => {
            const id = crypto.randomUUID();
            this.pendingRequests.set(id, { resolve, reject });

            // Transfer the underlying ArrayBuffer (zero-copy)
            const buffer = imageData.data.buffer;
            this.worker!.postMessage({
                type: 'RECOGNIZE',
                payload: { imageBuffer: imageData.data, id, useBGR: this.useBGR() }
            }, [buffer]);
        });
    }

    public addKnownFace(name: string, embedding: number[], imageUrl?: string) {
        const face: KnownFace = {
            name,
            embedding,
            imageUrl,
            timestamp: Date.now()
        };

        const updated = [...this.knownFaces(), face];
        this.knownFaces.set(updated);
        this.saveKnownFaces(updated);
    }

    public matchFace(embedding: number[]): { name: string, score: number } | null {
        const faces = this.knownFaces();
        if (faces.length === 0) return { name: 'Unknown', score: 0 };

        let bestScore = -1;
        let bestMatch = 'Unknown';

        for (const face of faces) {
            const score = this.cosineSimilarity(embedding, face.embedding);
            if (score > bestScore) {
                bestScore = score;
                bestMatch = face.name;
            }
        }

        // Threshold usually 0.4 - 0.6 for ArcFace/EdgeFace
        if (bestScore > 0.4) {
            // Map 0.4 - 1.0 to 0.8 - 1.0
            const mappedScore = 0.8 + ((bestScore - 0.4) / 0.6) * 0.2;
            return { name: bestMatch, score: mappedScore };
        }
        return { name: 'Unknown', score: bestScore };
    }

    public deleteKnownFace(face: KnownFace) {
        const updated = this.knownFaces().filter(f => f.timestamp !== face.timestamp);
        this.knownFaces.set(updated);
        this.saveKnownFaces(updated);
    }

    private cosineSimilarity(a: number[], b: number[]): number {
        let dot = 0;
        let magA = 0;
        let magB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-6);
    }

    private saveKnownFaces(faces: KnownFace[]) {
        // localStorage.setItem('known_faces', JSON.stringify(faces));
    }

    private loadKnownFaces() {
        const stored = localStorage.getItem('known_faces');
        if (stored) {
            try {
                this.knownFaces.set(JSON.parse(stored));
            } catch {
                this.knownFaces.set([]);
            }
        }
    }
}
