import * as pdfjsLib from 'pdfjs-dist';

// Configure worker src (use .mjs for v4+)
(pdfjsLib as any).GlobalWorkerOptions.workerSrc =
  'https://esm.sh/pdfjs-dist@4.0.379/build/pdf.worker.min.mjs';

export interface RenderedPage {
  pageNumber: number;
  dataUrl: string;
  width: number;
  height: number;
  embeddedText?: { text: string; words: any[] };
}

/**
 * Render every page of a PDF file to a JPEG data URL.
 *
 * HARDENED version:
 * - White-background fill before render (avoids transparent/CMYK black pages)
 * - Honors rotation metadata
 * - Handles password-protected / corrupt PDFs with clear errors
 * - Falls back through worker → no-worker retry if worker fails
 * - Loads cmaps and standard fonts for international/vector PDFs
 */
export async function renderPdfToImages(
  file: File | Blob,
  opts: { scale?: number; onProgress?: (done: number, total: number) => void; concurrency?: number } = {}
): Promise<RenderedPage[]> {
  const scale = opts.scale ?? 1.5;
  const concurrency = opts.concurrency ?? Math.min(4, Math.max(2, (navigator.hardwareConcurrency || 4) - 1));

  const buf = await file.arrayBuffer();
  const pdf = await loadPdfWithFallback(buf);

  const total = pdf.numPages;
  const results: (RenderedPage | null)[] = new Array(total).fill(null);
  let completed = 0;

  const renderOne = async (pageNum: number) => {
    try {
      const page = await pdf.getPage(pageNum);
      const viewport = page.getViewport({ scale, rotation: page.rotate || 0 });

      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.ceil(viewport.width));
      canvas.height = Math.max(1, Math.ceil(viewport.height));
      const ctx = canvas.getContext('2d', { willReadFrequently: false, alpha: false })!;

      // CRITICAL: white bg first
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      await page.render({
        canvasContext: ctx,
        viewport,
        background: 'rgba(255,255,255,1)',
        intent: 'display',
      }).promise;

      const dataUrl = canvas.toDataURL('image/jpeg', 0.88);
      const embeddedText = await extractEmbeddedText(page);

      try { page.cleanup(); } catch { /* noop */ }

      results[pageNum - 1] = {
        pageNumber: pageNum,
        dataUrl,
        width: canvas.width,
        height: canvas.height,
        embeddedText: embeddedText || undefined,
      };
    } catch (e: any) {
      console.error(`Failed to render PDF page ${pageNum}:`, e);
      results[pageNum - 1] = makeErrorPage(pageNum, e?.message || 'Render error');
    } finally {
      completed++;
      opts.onProgress?.(completed, total);
    }
  };

  for (let start = 0; start < total; start += concurrency) {
    const batch: Promise<void>[] = [];
    const end = Math.min(start + concurrency, total);
    for (let p = start + 1; p <= end; p++) {
      batch.push(renderOne(p));
    }
    await Promise.all(batch);
  }

  try { await pdf.cleanup(); } catch { /* noop */ }
  try { await pdf.destroy(); } catch { /* noop */ }

  return results.filter((r): r is RenderedPage => r !== null);
}

async function extractEmbeddedText(page: any): Promise<{text: string, words: any[]} | null> {
  try {
    const textContent = await page.getTextContent();
    if (!textContent || textContent.items.length === 0) return null;
    const viewport = page.getViewport({ scale: 1.0, rotation: page.rotate || 0 });
    let text = '';
    const words: any[] = [];
    let lastY = -1;
    
    for (const item of textContent.items) {
      if (!item.str || !item.str.trim()) continue;
      
      if (lastY !== -1 && Math.abs(item.transform[5] - lastY) > 8) text += '\n';
      else if (text.length > 0 && !text.endsWith('\n') && !text.endsWith(' ')) text += ' ';
      text += item.str;
      lastY = item.transform[5];
      
      const [ptX, ptY] = viewport.convertToViewportPoint(item.transform[4], item.transform[5]);
      const itemW = item.width || Math.abs(item.transform[0]);
      const itemH = item.height || Math.abs(item.transform[3]); 
      const topY = ptY - (itemH * 0.85);

      words.push({
        text: item.str.trim(),
        x: ptX / viewport.width,
        y: topY / viewport.height,
        w: itemW / viewport.width,
        h: itemH / viewport.height
      });
    }
    return { text: text.trim(), words };
  } catch {
    return null;
  }
}

/**
 * Try loading PDF with worker; if the worker fails (CORS, network, etc.),
 * retry with disableWorker so the main thread parses instead. This rescues
 * PDFs that failed silently before.
 */
async function loadPdfWithFallback(buf: ArrayBuffer): Promise<any> {
  const commonOptions = {
    disableFontFace: false,
    useSystemFonts: true,
    isEvalSupported: true,
    cMapUrl: 'https://esm.sh/pdfjs-dist@4.0.379/cmaps/',
    cMapPacked: true,
    standardFontDataUrl: 'https://esm.sh/pdfjs-dist@4.0.379/standard_fonts/',
  };

  try {
    return await (pdfjsLib as any).getDocument({
      data: buf.slice(0), // clone so retry has its own copy
      ...commonOptions,
    }).promise;
  } catch (e: any) {
    console.warn('[pdfToImages] Worker load failed, retrying without worker:', e);
    // Retry with worker disabled
    try {
      return await (pdfjsLib as any).getDocument({
        data: buf.slice(0),
        disableWorker: true,
        ...commonOptions,
      }).promise;
    } catch (e2: any) {
      // Surface a clean error to the caller
      const msg = e2?.message || e?.message || 'Unknown PDF loading error';
      if (/password/i.test(msg)) {
        throw new Error('This PDF is password-protected and cannot be opened.');
      }
      if (/invalid|corrupt|malformed/i.test(msg)) {
        throw new Error('The PDF file is corrupted or not a valid PDF.');
      }
      throw new Error(`Could not open PDF: ${msg}`);
    }
  }
}

function makeErrorPage(pageNum: number, message: string): RenderedPage {
  const canvas = document.createElement('canvas');
  canvas.width = 600;
  canvas.height = 800;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#999';
  ctx.font = '20px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`Page ${pageNum} (render error)`, canvas.width / 2, canvas.height / 2);
  ctx.font = '14px sans-serif';
  ctx.fillText(
    String(message).slice(0, 80),
    canvas.width / 2,
    canvas.height / 2 + 30
  );
  return {
    pageNumber: pageNum,
    dataUrl: canvas.toDataURL('image/jpeg', 0.88),
    width: canvas.width,
    height: canvas.height,
  };
}

export function isPdfFile(f: File): boolean {
  return f.type === 'application/pdf' || /\.pdf$/i.test(f.name);
}