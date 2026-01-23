import { Injectable, signal } from '@angular/core';
import { FaceLandmarker, FilesetResolver, FaceLandmarkerResult } from '@mediapipe/tasks-vision';

@Injectable({
    providedIn: 'root'
})
export class FaceDetectionService {
    faceLandmarker: FaceLandmarker | null = null;
    isLoaded = signal<boolean>(false);

    // Running mode: 'IMAGE', 'VIDEO', or 'LIVE_STREAM'
    // We use VIDEO for frames or generic webcam feed
    runningMode: 'IMAGE' | 'VIDEO' = 'VIDEO';

    constructor() {
        this.initialize();
    }

    async initialize() {
        try {
            // Load WASM from CDN or local if migrated
            const vision = await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
                // If we want local: 'assets/wasm' (need to copy files there first)
            );

            this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                    // or local: 'assets/models/face_landmarker.task'
                    delegate: 'GPU'
                },
                outputFaceBlendshapes: false,
                outputFacialTransformationMatrixes: false,
                runningMode: this.runningMode,
                numFaces: 5
            });

            this.isLoaded.set(true);
            console.log('FaceLandmarker loaded!');
        } catch (err) {
            console.error('Failed to init FaceLandmarker:', err);
        }
    }

    async detect(videoElement: HTMLVideoElement, startTimeMs: number): Promise<FaceLandmarkerResult | null> {
        if (!this.faceLandmarker || !this.isLoaded()) return null;

        try {
            const results = this.faceLandmarker.detectForVideo(videoElement, startTimeMs);
            return results;
        } catch (e) {
            console.error('Detection error:', e);
            return null;
        }
    }

    async detectImage(image: HTMLImageElement | HTMLCanvasElement): Promise<FaceLandmarkerResult | null> {
        if (!this.faceLandmarker || !this.isLoaded()) return null;

        // Switch to IMAGE mode if currently in VIDEO mode
        if (this.runningMode !== 'IMAGE') {
            this.runningMode = 'IMAGE';
            await this.faceLandmarker.setOptions({ runningMode: 'IMAGE' });
        }

        try {
            return this.faceLandmarker.detect(image);
        } catch (e) {
            console.error('Image detection error:', e);
            return null;
        }
    }

    async switchToVideoMode() {
        if (!this.faceLandmarker) return;
        if (this.runningMode !== 'VIDEO') {
            this.runningMode = 'VIDEO';
            await this.faceLandmarker.setOptions({ runningMode: 'VIDEO' });
        }
    }
}
