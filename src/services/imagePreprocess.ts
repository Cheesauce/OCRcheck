
// Advanced image pre-processing pipeline for faint/low-contrast OCR.
// Runs CPU-intensive filters off the main thread (OffscreenCanvas) when possible.
// Produces multiple enhanced variants so Tesseract can "vote" on the best read.

export interface PreprocessResult {
  dataUrl: string;
  variant: string;
  width: number;
  height: number;
}

export interface PreprocessOptions {
  // Target minimum DPI-equivalent dimension. Faint text benefits from upscaling.
  minLongEdge?: number;
  // Which variants to produce. More = more accurate but slower.
  variants?: PreprocessVariant[];
}

export type PreprocessVariant =
  | 'original'
  | 'grayscale-clahe'       // grayscale + adaptive histogram equalization (best for faded text)
  | 'binarized-sauvola'     // adaptive Sauvola binarization (best for uneven lighting)
  | 'high-contrast'         // strong global contrast + gamma (best for very pale text)
  | 'inverted-bright'       // for text that is lighter than background
  | 'sharpened';            // unsharp mask for slightly blurry scans

const DEFAULT_VARIANTS: PreprocessVariant[] = [
  'grayscale-clahe',
  'binarized-sauvola',
  'high-contrast',
  'sharpened',
];

const hasOffscreen = typeof OffscreenCanvas !== 'undefined';

/**
 * Generate multiple enhanced versions of an image. Each is tuned to recover
 * different kinds of faint/degraded text.
 */
export async function preprocessForOCR(
  source: string | Blob | ImageBitmap | HTMLImageElement,
  opts: PreprocessOptions = {}
): Promise<PreprocessResult[]> {
  const variants = opts.variants ?? DEFAULT_VARIANTS;
  const minLongEdge = opts.minLongEdge ?? 1800;

  // Load once
  const bitmap = await toBitmap(source);
  const { width: ow, height: oh } = bitmap;

  // Upscale if too small — faint text needs resolution
  const longEdge = Math.max(ow, oh);
  const scale = longEdge < minLongEdge ? minLongEdge / longEdge : 1;
  const W = Math.round(ow * scale);
  const H = Math.round(oh * scale);

  // Draw base (upscaled, bicubic-ish via browser) into a shared canvas
  const baseCanvas = makeCanvas(W, H);
  const baseCtx = baseCanvas.getContext('2d') as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D;
  (baseCtx as any).imageSmoothingEnabled = true;
  (baseCtx as any).imageSmoothingQuality = 'high';
  baseCtx.fillStyle = '#ffffff';
  baseCtx.fillRect(0, 0, W, H);
  baseCtx.drawImage(bitmap as any, 0, 0, W, H);

  // Get base pixels once
  const baseImg = baseCtx.getImageData(0, 0, W, H);

  const results: PreprocessResult[] = [];

  for (const variant of variants) {
    try {
      const processed = await applyVariant(baseImg, variant, W, H);
      const canvas = makeCanvas(W, H);
      const ctx = canvas.getContext('2d') as any;
      ctx.putImageData(processed, 0, 0);
      const dataUrl = await canvasToDataUrl(canvas);
      results.push({ dataUrl, variant, width: W, height: H });
    } catch (e) {
      console.warn(`Preprocess variant "${variant}" failed:`, e);
    }
  }

  // Also include lightly-upscaled original as a safety fallback
  if (variants.includes('original') || results.length === 0) {
    const canvas = makeCanvas(W, H);
    const ctx = canvas.getContext('2d') as any;
    ctx.putImageData(baseImg, 0, 0);
    results.push({
      dataUrl: await canvasToDataUrl(canvas),
      variant: 'original',
      width: W,
      height: H,
    });
  }

  // Release bitmap
  if ('close' in bitmap) {
    try { (bitmap as ImageBitmap).close(); } catch { /* noop */ }
  }

  return results;
}

async function applyVariant(
  src: ImageData,
  variant: PreprocessVariant,
  W: number,
  H: number
): Promise<ImageData> {
  switch (variant) {
    case 'grayscale-clahe':
      return claheGrayscale(src, W, H);
    case 'binarized-sauvola':
      return sauvolaBinarize(src, W, H);
    case 'high-contrast':
      return highContrastGamma(src, W, H);
    case 'inverted-bright':
      return invertIfBrightText(src, W, H);
    case 'sharpened':
      return unsharpMask(src, W, H);
    case 'original':
    default:
      return src;
  }
}

/* ============================================================
 * Variant 1: CLAHE-like grayscale
 * Contrast-Limited Adaptive Histogram Equalization (tile-based).
 * Excellent for faded photocopies, carbon copies, sun-damaged documents.
 * ============================================================ */

function claheGrayscale(src: ImageData, W: number, H: number): ImageData {
  const out = new ImageData(W, H);
  const d = src.data;
  const o = out.data;

  // Convert to grayscale
  const gray = new Uint8ClampedArray(W * H);
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    gray[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  }

  // CLAHE parameters
  const TILE = 64;                 // tile size in px
  const clipLimit = 4.0;           // contrast clip (histogram bin cap)
  const tilesX = Math.max(1, Math.ceil(W / TILE));
  const tilesY = Math.max(1, Math.ceil(H / TILE));

  // Compute LUT for each tile
  const luts: Uint8Array[] = [];
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = tx * TILE;
      const y0 = ty * TILE;
      const x1 = Math.min(W, x0 + TILE);
      const y1 = Math.min(H, y0 + TILE);

      const hist = new Uint32Array(256);
      for (let y = y0; y < y1; y++) {
        const row = y * W;
        for (let x = x0; x < x1; x++) {
          hist[gray[row + x]]++;
        }
      }

      // Clip histogram
      const n = (x1 - x0) * (y1 - y0);
      const clip = Math.max(1, Math.floor((clipLimit * n) / 256));
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > clip) {
          excess += hist[i] - clip;
          hist[i] = clip;
        }
      }
      // Redistribute excess uniformly
      const bonus = Math.floor(excess / 256);
      let leftover = excess - bonus * 256;
      for (let i = 0; i < 256; i++) {
        hist[i] += bonus;
        if (leftover > 0) {
          hist[i]++;
          leftover--;
        }
      }

      // CDF → LUT
      const lut = new Uint8Array(256);
      let cdf = 0;
      const scale = 255 / Math.max(1, n);
      for (let i = 0; i < 256; i++) {
        cdf += hist[i];
        lut[i] = Math.max(0, Math.min(255, Math.round(cdf * scale)));
      }
      luts.push(lut);
    }
  }

  // Bilinear-interpolate between tile LUTs
  for (let y = 0; y < H; y++) {
    const fy = (y - TILE / 2) / TILE;
    let ty0 = Math.floor(fy);
    let ay = fy - ty0;
    if (ty0 < 0) { ty0 = 0; ay = 0; }
    if (ty0 >= tilesY - 1) { ty0 = tilesY - 1; ay = 1; }
    const ty1 = Math.min(tilesY - 1, ty0 + 1);

    for (let x = 0; x < W; x++) {
      const fx = (x - TILE / 2) / TILE;
      let tx0 = Math.floor(fx);
      let ax = fx - tx0;
      if (tx0 < 0) { tx0 = 0; ax = 0; }
      if (tx0 >= tilesX - 1) { tx0 = tilesX - 1; ax = 1; }
      const tx1 = Math.min(tilesX - 1, tx0 + 1);

      const g = gray[y * W + x];
      const v00 = luts[ty0 * tilesX + tx0][g];
      const v01 = luts[ty0 * tilesX + tx1][g];
      const v10 = luts[ty1 * tilesX + tx0][g];
      const v11 = luts[ty1 * tilesX + tx1][g];

      const top = v00 * (1 - ax) + v01 * ax;
      const bot = v10 * (1 - ax) + v11 * ax;
      const v = Math.round(top * (1 - ay) + bot * ay);

      const idx = (y * W + x) * 4;
      o[idx] = v;
      o[idx + 1] = v;
      o[idx + 2] = v;
      o[idx + 3] = 255;
    }
  }

  return out;
}

/* ============================================================
 * Variant 2: Sauvola adaptive binarization
 * Best for scans with uneven lighting, shadows, show-through.
 * Uses integral images for O(1) local mean/variance.
 * ============================================================ */

function sauvolaBinarize(src: ImageData, W: number, H: number): ImageData {
  const d = src.data;
  const out = new ImageData(W, H);
  const o = out.data;

  // Grayscale
  const gray = new Float32Array(W * H);
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    gray[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  }

  // Integral images for mean & squared mean
  const intSum = new Float64Array((W + 1) * (H + 1));
  const intSqSum = new Float64Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let rowSum = 0;
    let rowSqSum = 0;
    for (let x = 0; x < W; x++) {
      const v = gray[y * W + x];
      rowSum += v;
      rowSqSum += v * v;
      const idx = (y + 1) * (W + 1) + (x + 1);
      intSum[idx] = intSum[y * (W + 1) + (x + 1)] + rowSum;
      intSqSum[idx] = intSqSum[y * (W + 1) + (x + 1)] + rowSqSum;
    }
  }

  // Sauvola params
  const window = Math.max(15, Math.round(Math.min(W, H) * 0.025));
  const half = Math.floor(window / 2);
  const k = 0.34;     // sensitivity
  const R = 128;      // dynamic range of standard deviation

  for (let y = 0; y < H; y++) {
    const y0 = Math.max(0, y - half);
    const y1 = Math.min(H - 1, y + half);
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - half);
      const x1 = Math.min(W - 1, x + half);
      const area = (y1 - y0 + 1) * (x1 - x0 + 1);

      const A = y0 * (W + 1) + x0;
      const B = y0 * (W + 1) + (x1 + 1);
      const C = (y1 + 1) * (W + 1) + x0;
      const D = (y1 + 1) * (W + 1) + (x1 + 1);

      const sum = intSum[D] - intSum[B] - intSum[C] + intSum[A];
      const sqSum = intSqSum[D] - intSqSum[B] - intSqSum[C] + intSqSum[A];
      const mean = sum / area;
      const variance = Math.max(0, sqSum / area - mean * mean);
      const std = Math.sqrt(variance);

      const threshold = mean * (1 + k * (std / R - 1));
      const v = gray[y * W + x];
      const bin = v > threshold ? 255 : 0;

      const idx = (y * W + x) * 4;
      o[idx] = bin;
      o[idx + 1] = bin;
      o[idx + 2] = bin;
      o[idx + 3] = 255;
    }
  }

  return out;
}

/* ============================================================
 * Variant 3: High contrast + gamma
 * Simple but very effective for slightly faded print.
 * ============================================================ */

function highContrastGamma(src: ImageData, W: number, H: number): ImageData {
  const out = new ImageData(W, H);
  const d = src.data;
  const o = out.data;

  // Pass 1: measure histogram to find black/white points
  const hist = new Uint32Array(256);
  for (let i = 0; i < d.length; i += 4) {
    const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    hist[Math.max(0, Math.min(255, Math.round(g)))]++;
  }
  const total = W * H;
  const lowFrac = 0.02;
  const highFrac = 0.995;
  let cum = 0;
  let blackPoint = 0;
  let whitePoint = 255;
  for (let i = 0; i < 256; i++) {
    cum += hist[i];
    if (cum >= total * lowFrac) { blackPoint = i; break; }
  }
  cum = 0;
  for (let i = 0; i < 256; i++) {
    cum += hist[i];
    if (cum >= total * highFrac) { whitePoint = i; break; }
  }
  const range = Math.max(1, whitePoint - blackPoint);
  const gamma = 0.85; // boost midtones slightly

  // LUT
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    let v = (i - blackPoint) / range;
    v = Math.max(0, Math.min(1, v));
    v = Math.pow(v, gamma);
    lut[i] = Math.round(v * 255);
  }

  for (let i = 0; i < d.length; i += 4) {
    const g = Math.max(0, Math.min(255, Math.round(
      0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
    )));
    const v = lut[g];
    o[i] = v;
    o[i + 1] = v;
    o[i + 2] = v;
    o[i + 3] = 255;
  }
  return out;
}

/* ============================================================
 * Variant 4: Invert if text is brighter than background
 * Detects "light on dark" documents and flips them.
 * ============================================================ */

function invertIfBrightText(src: ImageData, W: number, H: number): ImageData {
  const out = new ImageData(W, H);
  const d = src.data;
  const o = out.data;

  // Sample mean
  let sum = 0;
  let count = 0;
  for (let i = 0; i < d.length; i += 16) {
    sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    count++;
  }
  const mean = sum / count;
  const shouldInvert = mean < 110;

  for (let i = 0; i < d.length; i += 4) {
    let g = Math.max(0, Math.min(255, Math.round(
      0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
    )));
    if (shouldInvert) g = 255 - g;
    o[i] = g;
    o[i + 1] = g;
    o[i + 2] = g;
    o[i + 3] = 255;
  }
  return highContrastGamma(out, W, H);
}

/* ============================================================
 * Variant 5: Unsharp mask
 * For slightly blurred or JPEG-compressed scans.
 * ============================================================ */

function unsharpMask(src: ImageData, W: number, H: number): ImageData {
  const d = src.data;
  // Grayscale first
  const gray = new Uint8ClampedArray(W * H);
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    gray[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  }

  // Box blur approximation (separable) with radius 2
  const blurred = boxBlur(gray, W, H, 2);

  const amount = 1.4;
  const threshold = 3;
  const out = new ImageData(W, H);
  const o = out.data;
  for (let i = 0; i < gray.length; i++) {
    const orig = gray[i];
    const blur = blurred[i];
    let diff = orig - blur;
    if (Math.abs(diff) < threshold) diff = 0;
    const v = Math.max(0, Math.min(255, Math.round(orig + amount * diff)));
    const idx = i * 4;
    o[idx] = v;
    o[idx + 1] = v;
    o[idx + 2] = v;
    o[idx + 3] = 255;
  }

  // Follow up with auto-contrast for final punch
  return highContrastGamma(out, W, H);
}

function boxBlur(src: Uint8ClampedArray, W: number, H: number, r: number): Uint8ClampedArray {
  const tmp = new Uint8ClampedArray(W * H);
  const out = new Uint8ClampedArray(W * H);
  const size = 2 * r + 1;

  // Horizontal
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let sum = 0;
    for (let x = -r; x <= r; x++) {
      sum += src[row + Math.max(0, Math.min(W - 1, x))];
    }
    for (let x = 0; x < W; x++) {
      tmp[row + x] = sum / size;
      const xAdd = Math.min(W - 1, x + r + 1);
      const xSub = Math.max(0, x - r);
      sum += src[row + xAdd] - src[row + xSub];
    }
  }
  // Vertical
  for (let x = 0; x < W; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) {
      sum += tmp[Math.max(0, Math.min(H - 1, y)) * W + x];
    }
    for (let y = 0; y < H; y++) {
      out[y * W + x] = sum / size;
      const yAdd = Math.min(H - 1, y + r + 1);
      const ySub = Math.max(0, y - r);
      sum += tmp[yAdd * W + x] - tmp[ySub * W + x];
    }
  }
  return out;
}

/* ============================================================
 * Utilities
 * ============================================================ */

function makeCanvas(W: number, H: number): HTMLCanvasElement | OffscreenCanvas {
  if (hasOffscreen) return new OffscreenCanvas(W, H);
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  return c;
}

async function canvasToDataUrl(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<string> {
  if (hasOffscreen && canvas instanceof OffscreenCanvas) {
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    return await blobToDataUrl(blob);
  }
  return (canvas as HTMLCanvasElement).toDataURL('image/png');
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

async function toBitmap(
  source: string | Blob | ImageBitmap | HTMLImageElement
): Promise<ImageBitmap | HTMLImageElement> {
  if (typeof source === 'string') {
    if (typeof createImageBitmap === 'function') {
      const res = await fetch(source);
      const blob = await res.blob();
      return await createImageBitmap(blob);
    }
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = source;
    });
  }
  if (source instanceof Blob) {
    if (typeof createImageBitmap === 'function') {
      return await createImageBitmap(source);
    }
    const url = URL.createObjectURL(source);
    return await toBitmap(url);
  }
  return source as any;
}
