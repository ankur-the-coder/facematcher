import { Component, ElementRef, ViewChild, signal, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CameraService } from './core/services/camera.service';
import { FaceDetectionService } from './core/services/face-detection.service';
import { FaceRecognitionService, KnownFace } from './core/services/face-recognition.service';
import { estimateAffineTransform, warpFace } from './core/utils/image-utils';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasElement') canvasElement!: ElementRef<HTMLCanvasElement>;

  // UI State
  isLoading = signal(true);
  isCameraActive = signal(false);
  fps = signal(0);
  currentMatch = signal<string>("Unknown");
  showSidebar = signal(false); // For mobile drawer

  private lastFrameTime = 0;
  private captureNextFrame = false;

  // Reusable OffscreenCanvas for face cropping (avoids GC pressure)
  private cropCanvas: OffscreenCanvas | null = null;
  private cropCtx: OffscreenCanvasRenderingContext2D | null = null;

  // FPS throttle: update display only every 500ms
  private fpsFrameCount = 0;
  private fpsLastUpdate = 0;

  knownFaces = signal<KnownFace[]>([]);

  constructor(
    public cameraService: CameraService,
    private detectionService: FaceDetectionService,
    public recognitionService: FaceRecognitionService
  ) {
    this.knownFaces = this.recognitionService.knownFaces;

    // Determine overall loading state based on models
    effect(() => {
      const detLoaded = this.detectionService.isLoaded();
      const recLoaded = this.recognitionService.isModelLoaded();
      this.isLoading.set(!detLoaded || !recLoaded);
    });
  }

  async startApp() {
    await this.cameraService.startCamera();
    this.isCameraActive.set(true);

    const video = this.videoElement.nativeElement;
    video.srcObject = this.cameraService.stream();
    video.play();

    video.onloadedmetadata = () => {
      this.processFrame();
    };
  }

  async processFrame() {
    if (!this.isCameraActive()) return;

    const video = this.videoElement.nativeElement;
    const canvas = this.canvasElement.nativeElement;
    const ctx = canvas.getContext('2d');

    if (!ctx || video.videoWidth === 0) {
      requestAnimationFrame(() => this.processFrame());
      return;
    }

    // FPS Calculation (throttled — update every 500ms)
    const now = performance.now();
    this.fpsFrameCount++;
    if (now - this.fpsLastUpdate >= 500) {
      const elapsed = now - this.fpsLastUpdate;
      this.fps.set(Math.round((this.fpsFrameCount / elapsed) * 1000));
      this.fpsFrameCount = 0;
      this.fpsLastUpdate = now;
    }

    // Match canvas size to video
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // 1. Detect
    const results = await this.detectionService.detect(video, performance.now());

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results && results.faceLandmarks.length > 0) {
      // Initialize reusable crop canvas once
      if (!this.cropCanvas) {
        this.cropCanvas = new OffscreenCanvas(112, 112);
        this.cropCtx = this.cropCanvas.getContext('2d')!;
      }

      for (let i = 0; i < results.faceLandmarks.length; i++) {
        const landmarks = results.faceLandmarks[i];

        // 2. Draw Bounding Box
        const box = this.getBoundingBox(landmarks, canvas.width, canvas.height);

        // MIRROR BOX due to CSS mirror on video
        const mirroredBox = {
          ...box,
          x: canvas.width - box.x - box.w
        };

        this.drawBox(ctx, mirroredBox);

        // 3. Align & Crop
        const p33 = landmarks[33];
        const p133 = landmarks[133];
        const p362 = landmarks[362];
        const p263 = landmarks[263];
        const pNose = landmarks[4];
        const pMouthL = landmarks[61];
        const pMouthR = landmarks[291];

        const leftEye = { x: (p33.x + p133.x) / 2, y: (p33.y + p133.y) / 2 };
        const rightEye = { x: (p362.x + p263.x) / 2, y: (p362.y + p263.y) / 2 };

        const keyPoints = [
          leftEye, rightEye, pNose, pMouthL, pMouthR
        ].map(p => ({ x: p.x * canvas.width, y: p.y * canvas.height }));

        // 4. Transform
        const matrix = estimateAffineTransform(keyPoints);

        // 5. Warp
        try {
          const warpedBitmap = await warpFace(video, matrix);

          // Reuse the crop canvas instead of allocating new one each frame
          const wCtx = this.cropCtx!;
          wCtx.drawImage(warpedBitmap, 0, 0);
          const imgData = wCtx.getImageData(0, 0, 112, 112);

          // 6. Recognize — sends ImageData directly, buffer transferred (zero-copy)
          const embedding = await this.recognitionService.getEmbedding(imgData);

          if (this.captureNextFrame) {
            this.captureNextFrame = false;
            const blob = await this.cropCanvas!.convertToBlob();
            const url = URL.createObjectURL(blob);
            const name = `Live_${new Date().toLocaleTimeString().replace(/:/g, '')}`;
            this.recognitionService.addKnownFace(name, embedding, url);
            alert(`Face '${name}' Captured!`);
          }

          // 7. Match
          const match = this.recognitionService.matchFace(embedding);

          if (match) {
            this.currentMatch.set(`${match.name} (${(match.score * 100).toFixed(0)}%)`);
          } else {
            this.currentMatch.set("Unknown");
          }

          this.drawLabel(ctx, mirroredBox, match?.name || "Unknown", match?.score || 0);

        } catch (e) {
          console.error('ProcessFrame Error:', e);
        }
      }
    }

    requestAnimationFrame(() => this.processFrame());
  }

  // --- Helpers ---

  getBoundingBox(landmarks: any[], w: number, h: number) {
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    landmarks.forEach(p => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });
    return {
      x: minX * w,
      y: minY * h,
      w: (maxX - minX) * w,
      h: (maxY - minY) * h
    };
  }

  drawBox(ctx: CanvasRenderingContext2D, box: any) {
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
  }

  drawLabel(ctx: CanvasRenderingContext2D, box: any, text: string, score: number) {
    const label = `${text} (${(score * 100).toFixed(0)}%)`;
    ctx.font = 'bold 18px Arial';
    const textWidth = ctx.measureText(label).width;
    const padding = 6;

    let y = box.y - 10;
    if (y < 25) y = box.y + box.h + 25;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(box.x, y - 18, textWidth + padding * 2, 24);

    ctx.fillStyle = '#00ff00';
    ctx.fillText(label, box.x + padding, y);
  }

  async handleUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const files = Array.from(input.files);

      for (const file of files) {
        await this.processFile(file);
      }
    }
  }

  async processFile(file: File) {
    const name = file.name.split('.')[0];
    const img = new Image();
    img.src = URL.createObjectURL(file);

    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
    });

    const results = await this.detectionService.detectImage(img);
    await this.detectionService.switchToVideoMode();

    if (results && results.faceLandmarks.length > 0) {
      const landmarks = results.faceLandmarks[0];
      const p33 = landmarks[33];
      const p133 = landmarks[133];
      const p362 = landmarks[362];
      const p263 = landmarks[263];
      const pNose = landmarks[4];
      const pMouthL = landmarks[61];
      const pMouthR = landmarks[291];

      const leftEye = { x: (p33.x + p133.x) / 2, y: (p33.y + p133.y) / 2 };
      const rightEye = { x: (p362.x + p263.x) / 2, y: (p362.y + p263.y) / 2 };

      const keyPoints = [
        leftEye, rightEye, pNose, pMouthL, pMouthR
      ].map(p => ({ x: p.x * img.width, y: p.y * img.height }));

      const matrix = estimateAffineTransform(keyPoints);
      const warpedBitmap = await warpFace(img, matrix);

      const warpedCanvas = new OffscreenCanvas(112, 112);
      const wCtx = warpedCanvas.getContext('2d')!;
      wCtx.drawImage(warpedBitmap, 0, 0);
      const imgData = wCtx.getImageData(0, 0, 112, 112);

      const embedding = await this.recognitionService.getEmbedding(imgData);

      const thumbBlob = await warpedCanvas.convertToBlob();
      const thumbUrl = URL.createObjectURL(thumbBlob);

      this.recognitionService.addKnownFace(name, embedding, thumbUrl);
    } else {
      alert(`No face detected in ${file.name}!`);
    }
  }

  deleteFace(face: KnownFace) {
    if (confirm(`Delete face: ${face.name}?`)) {
      this.recognitionService.deleteKnownFace(face);
    }
  }

  triggerSnapshot() {
    this.captureNextFrame = true;
  }

  toggleColorMode() {
    this.recognitionService.useBGR.update(v => !v);
    this.recognitionService.knownFaces.set([]);
    this.currentMatch.set("Unknown");
  }

  toggleSidebar() {
    this.showSidebar.update(v => !v);
  }
}
