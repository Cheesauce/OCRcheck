// Image compression to save space in IndexedDB.
// Converts an image source (File, Blob, or dataURL) to a compressed JPEG dataURL.
//
// OPTIMIZED: uses createImageBitmap when available (2-3x faster than <img>),
// and OffscreenCanvas when available (doesn't block main thread).

export interface CompressionResult {
  dataUrl: string;
  originalSize: number;
  compressedSize: number;
  width: number;
  height: number;
}

const hasImageBitmap = typeof createImageBitmap === 'function';
const hasOffscreenCanvas = typeof OffscreenCanvas !== 'undefined';

export async function compressImage(
  source: File | Blob | string,
  opts: { maxDim?: number; quality?: number } = {}
): Promise<CompressionResult> {
  // Removing limits: Preserving original size and setting default quality to 100%
  const maxDim = opts.maxDim ?? 99999;
  const quality = opts.quality ?? 1.0;

  const originalSize =
    typeof source === 'string'
      ? Math.ceil((source.length * 3) / 4)
      : (source as Blob).size;

  // Fast path: createImageBitmap is significantly faster than <img> + URL.createObjectURL
  let width: number;
  let height: number;
  let drawable: ImageBitmap | HTMLImageElement;

  if (hasImageBitmap && typeof source !== 'string') {
    drawable = await createImageBitmap(source as Blob);
    width = (drawable as ImageBitmap).width;
    height = (drawable as ImageBitmap).height;
  } else if (hasImageBitmap && typeof source === 'string') {
    // Convert data URL to blob, then bitmap
    const resp = await fetch(source);
    const blob = await resp.blob();
    drawable = await createImageBitmap(blob);
    width = (drawable as ImageBitmap).width;
    height = (drawable as ImageBitmap).height;
  } else {
    drawable = await loadImage(source);
    width = drawable.width;
    height = drawable.height;
  }

  let w = width;
  let h = height;
  if (w > maxDim || h > maxDim) {
    if (w >= h) {
      h = Math.round((h / w) * maxDim);
      w = maxDim;
    } else {
      w = Math.round((w / h) * maxDim);
      h = maxDim;
    }
  }

  // Use OffscreenCanvas when possible to free up the main thread
  let dataUrl: string;
  if (hasOffscreenCanvas) {
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(drawable as any, 0, 0, w, h);
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
    dataUrl = await blobToDataUrl(blob);
  } else {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(drawable as any, 0, 0, w, h);
    dataUrl = canvas.toDataURL('image/jpeg', quality);
  }

  // Release bitmap to free GPU memory
  if (hasImageBitmap && 'close' in (drawable as any)) {
    try { (drawable as ImageBitmap).close(); } catch { /* noop */ }
  }

  const compressedSize = Math.ceil((dataUrl.length * 3) / 4);
  return { dataUrl, originalSize, compressedSize, width: w, height: h };
}

function loadImage(source: File | Blob | string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    if (typeof source === 'string') {
      img.src = source;
    } else {
      img.src = URL.createObjectURL(source);
    }
  });
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}