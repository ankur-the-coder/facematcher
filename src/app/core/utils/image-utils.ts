
// Standard ArcFace/EdgeFace 112x112 Reference Landmarks
const REFERENCE_LANDMARKS = [
    [38.2946, 51.6963], // Left Eye
    [73.5318, 51.5014], // Right Eye
    [56.0252, 71.7366], // Nose
    [41.5493, 92.3655], // Left Mouth Corner
    [70.7299, 92.2041]  // Right Mouth Corner
];

export interface Point {
    x: number;
    y: number;
}

/**
 * Estimates a Similarity Transform matrix (scale, rotation, translation) 
 * to align srcPoints to dstPoints (Reference).
 * Solves using Least Squares.
 * Returns a 2x3 Affine Matrix [a, b, tx, c, d, ty]
 */
export function estimateAffineTransform(srcPoints: Point[]): number[] {
    // Simple non-reflective similarity transform implementation
    // Based on standard Umeyama or similar fitting

    if (srcPoints.length !== 5) {
        throw new Error("Alignment requires exactly 5 landmarks");
    }

    // Calculate centroids
    let srcCx = 0, srcCy = 0, dstCx = 0, dstCy = 0;
    for (let i = 0; i < 5; i++) {
        srcCx += srcPoints[i].x;
        srcCy += srcPoints[i].y;
        dstCx += REFERENCE_LANDMARKS[i][0];
        dstCy += REFERENCE_LANDMARKS[i][1];
    }
    srcCx /= 5; srcCy /= 5; dstCx /= 5; dstCy /= 5;

    // Subtract centroids
    let srcDemean: number[][] = [];
    let dstDemean: number[][] = [];
    for (let i = 0; i < 5; i++) {
        srcDemean.push([srcPoints[i].x - srcCx, srcPoints[i].y - srcCy]);
        dstDemean.push([REFERENCE_LANDMARKS[i][0] - dstCx, REFERENCE_LANDMARKS[i][1] - dstCy]);
    }

    // Eq: Dst = Scale * Rot * Src
    // Numerator and Denominator for Least Squares
    let A = 0, B = 0, D = 0;
    for (let i = 0; i < 5; i++) {
        A += srcDemean[i][0] * dstDemean[i][0] + srcDemean[i][1] * dstDemean[i][1];
        B += srcDemean[i][0] * dstDemean[i][1] - srcDemean[i][1] * dstDemean[i][0];
        D += srcDemean[i][0] ** 2 + srcDemean[i][1] ** 2;
    }

    const scale = Math.sqrt(A * A + B * B) / D;
    const angle = Math.atan2(B, A);

    const a = scale * Math.cos(angle);
    const b = scale * Math.sin(angle); // usually -sin for std math but canvas matrix logic
    // Rotation matrix R = [[cos, -sin], [sin, cos]]
    // Affine: [s*cos, -s*sin, tx
    //          s*sin, s*cos,  ty]

    // Wait, standard canvas matrix is: a, b, c, d, e, f -> [a c e], [b d f]
    // Let's stick to calculating coefficients for: x' = a*x + b*y + tx, y' = c*x + d*y + ty
    // With Similarity: a=s*cos, b=-s*sin, c=s*sin, d=s*cos

    // But wait, Math.atan2(B, A) gives rotation from Src to Dst
    // B/A = tan(theta)

    const cos = A / Math.sqrt(A * A + B * B);
    const sin = B / Math.sqrt(A * A + B * B);

    const T_a = scale * cos;
    const T_b = -scale * sin; // Note sign for standard coord system
    const T_c = scale * sin;
    const T_d = scale * cos;

    const tx = dstCx - (T_a * srcCx + T_b * srcCy);
    const ty = dstCy - (T_c * srcCx + T_d * srcCy);

    // Return specific format for Canvas setTransform or manual warping
    return [T_a, T_b, tx, T_c, T_d, ty];
}

/**
 * Warps an image using the affine matrix to 112x112.
 * This is computationally expensive in JS, should be done in Worker or using Canvas.
 * Using OffscreenCanvas in Worker is best.
 */
// Shared canvas for warping to avoid re-creation pressure
let sharedCanvas: OffscreenCanvas | null = null;
let sharedCtx: OffscreenCanvasRenderingContext2D | null = null;

export function warpFace(
    source: CanvasImageSource,
    matrix: number[]
): Promise<ImageBitmap> {
    const [a, b, tx, c, d, ty] = matrix;

    // Create or reuse OffscreenCanvas
    if (!sharedCanvas) {
        sharedCanvas = new OffscreenCanvas(112, 112);
        sharedCtx = sharedCanvas.getContext('2d')!;
    }

    const ctx = sharedCtx!;

    // Reset transform to identity before setting new one (though setTransform overrides, good practice)
    // ctx.resetTransform(); 
    // Actually setTransform wipes previous, so just set it.

    // Canvas transform params: (a, b, c, d, e, f)
    // x' = ax + cy + e
    // y' = bx + dy + f
    // Our matrix: x' = a*x + b*y + tx
    // Map T_a -> a, T_b -> c, tx -> e
    //     T_c -> b, T_d -> d, ty -> f

    ctx.setTransform(a, c, b, d, tx, ty);
    ctx.drawImage(source as any, 0, 0);

    return createImageBitmap(sharedCanvas);
}
