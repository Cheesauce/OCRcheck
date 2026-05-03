
/**
 * Composite Image Builder
 * =======================
 * Blends multiple training samples of the same subject (e.g. all "Crizil" stamps)
 * into a single **idealized composite image** that represents the "average"
 * appearance of that subject.
 *
 * Techniques used:
 *   1. Auto-align each sample to a reference (via center-of-mass + scale fit)
 *   2. Blend pixel values with alpha = 1/N  →  mean image
 *   3. Light contrast/clarity pass so faint features stay visible
 *
 * This composite is purely visual — the AI already builds its own internal
 * mean embedding. But showing users the composite is powerful: they see
 * exactly what the AI "thinks" the subject looks like.
 */

export interface CompositeOptions {
  size?: number;           // output size (square), default 320
  autoAlign?: boolean;     // center each sample by its content bounds
  backgroundColor?: string;
}

export async function buildCompositeImage(
  dataUrls: string[],
  opts: CompositeOptions = {}
): Promise<string | null> {
  if (dataUrls.length === 0) return null;
  if (dataUrls.length === 1) return dataUrls[0];

  const size = opts.size ?? 320;
  const autoAlign = opts.autoAlign ?? true;
  const bg = opts.backgroundColor ?? '#ffffff';

  // Load all images
  const imgs = await Promise.all(dataUrls.map(loadImg));

  // Output canvas
  const out = document.createElement('canvas');
  out.width = size;
  out.height = size;
  const octx = out.getContext('2d', { willReadFrequently: true })!;
  octx.fillStyle = bg;
  octx.fillRect(0, 0, size, size);

  // Accumulator (float32) for averaging RGB
  const acc = new Float32Array(size * size * 3);
  const counts = new Float32Array(size * size); // per-pixel weight (some pixels may be "bg" in one sample)

  for (const img of imgs) {
    const aligned = autoAlign ? alignAndFit(img, size, bg) : fitTo(img, size, bg);
    const data = aligned.getContext('2d', { willReadFrequently: true })!
      .getImageData(0, 0, size, size).data;

    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      // Ignore near-white background pixels so they don't wash out the composite
      const isBg = r > 245 && g > 245 && b > 245;
      if (!isBg) {
        acc[p * 3] += r;
        acc[p * 3 + 1] += g;
        acc[p * 3 + 2] += b;
        counts[p]++;
      }
    }
  }

  // Build averaged image
  const outData = octx.createImageData(size, size);
  for (let p = 0; p < size * size; p++) {
    if (counts[p] > 0) {
      outData.data[p * 4] = acc[p * 3] / counts[p];
      outData.data[p * 4 + 1] = acc[p * 3 + 1] / counts[p];
      outData.data[p * 4 + 2] = acc[p * 3 + 2] / counts[p];
      outData.data[p * 4 + 3] = 255;
    } else {
      // No sample had content here — use background
      outData.data[p * 4] = 255;
      outData.data[p * 4 + 1] = 255;
      outData.data[p * 4 + 2] = 255;
      outData.data[p * 4 + 3] = 255;
    }
  }
  octx.putImageData(outData, 0, 0);

  // Clarity pass: gentle contrast boost so faint averaged edges stay visible
  applyContrast(octx, size, size, 1.15);

  return out.toDataURL('image/jpeg', 0.9);
}

function fitTo(img: HTMLImageElement, size: number, bg: string): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, size, size);
  const ratio = Math.min(size / img.width, size / img.height);
  const w = img.width * ratio;
  const h = img.height * ratio;
  ctx.drawImage(img, (size - w) / 2, (size - h) / 2, w, h);
  return c;
}

function alignAndFit(img: HTMLImageElement, size: number, bg: string): HTMLCanvasElement {
  // Step 1: draw original to a working canvas
  const work = document.createElement('canvas');
  work.width = img.width;
  work.height = img.height;
  const wctx = work.getContext('2d', { willReadFrequently: true })!;
  wctx.fillStyle = bg;
  wctx.fillRect(0, 0, img.width, img.height);
  wctx.drawImage(img, 0, 0);

  // Step 2: find content bounding box (non-background pixels)
  const id = wctx.getImageData(0, 0, img.width, img.height).data;
  let minX = img.width, minY = img.height, maxX = 0, maxY = 0;
  let found = false;
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      const i = (y * img.width + x) * 4;
      const r = id[i], g = id[i + 1], b = id[i + 2];
      if (!(r > 245 && g > 245 && b > 245)) {
        found = true;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (!found) return fitTo(img, size, bg);

  const bw = Math.max(1, maxX - minX);
  const bh = Math.max(1, maxY - minY);

  // Step 3: draw cropped content centered and scaled into output
  const out = document.createElement('canvas');
  out.width = size;
  out.height = size;
  const octx = out.getContext('2d')!;
  octx.fillStyle = bg;
  octx.fillRect(0, 0, size, size);

  const padding = 0.9; // leave 10% margin
  const ratio = Math.min((size * padding) / bw, (size * padding) / bh);
  const dw = bw * ratio;
  const dh = bh * ratio;
  octx.drawImage(
    work,
    minX, minY, bw, bh,
    (size - dw) / 2, (size - dh) / 2, dw, dh
  );
  return out;
}

function applyContrast(ctx: CanvasRenderingContext2D, w: number, h: number, factor: number) {
  const id = ctx.getImageData(0, 0, w, h);
  const d = id.data;
  const intercept = 128 * (1 - factor);
  for (let i = 0; i < d.length; i += 4) {
    d[i] = Math.max(0, Math.min(255, d[i] * factor + intercept));
    d[i + 1] = Math.max(0, Math.min(255, d[i + 1] * factor + intercept));
    d[i + 2] = Math.max(0, Math.min(255, d[i + 2] * factor + intercept));
  }
  ctx.putImageData(id, 0, 0);
}

function loadImg(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
