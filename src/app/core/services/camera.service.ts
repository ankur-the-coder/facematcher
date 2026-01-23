import { Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class CameraService {
  stream = signal<MediaStream | null>(null);
  error = signal<string | null>(null);
  videoWidth = signal<number>(0);
  videoHeight = signal<number>(0);

  constructor() { }

  async startCamera(constraints: MediaStreamConstraints = {
    video: {
      facingMode: 'user',
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  }): Promise<void> {
    try {
      this.error.set(null);
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.stream.set(stream);

      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      this.videoWidth.set(settings.width ?? 0);
      this.videoHeight.set(settings.height ?? 0);

    } catch (err: any) {
      console.error('Error accessing camera:', err);
      this.error.set('Could not access camera. Please allow permissions.');
    }
  }

  stopCamera() {
    this.stream()?.getTracks().forEach(track => track.stop());
    this.stream.set(null);
  }
}
