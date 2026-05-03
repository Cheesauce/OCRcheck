import * as jsPDFModule from 'jspdf';
import type { OcrResult } from './ocrService';

function resolveJsPDF(): any {
  const m: any = jsPDFModule;
  if (typeof m === 'function') return m;
  if (m?.jsPDF) return m.jsPDF;
  if (m?.default?.jsPDF) return m.default.jsPDF;
  if (typeof m?.default === 'function') return m.default;
  if (typeof (window as any).jspdf?.jsPDF === 'function') return (window as any).jspdf.jsPDF;
  throw new Error('Unable to resolve jsPDF constructor from module.');
}

async function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load page image'));
    img.src = dataUrl;
  });
}

/**
 * Build a searchable PDF and return it as a Blob (for download, preview, or saving to db).
 */
export async function buildSearchablePdfBlob(
  result: OcrResult,
  onProgress?: (stage: string, progress: number) => void
): Promise<Blob> {
  const JsPDFCtor = resolveJsPDF();
  const pdf = new JsPDFCtor({ unit: 'pt', format: 'a4' });

  const pageW = pdf.internal.pageSize.getWidth();
  const pageH = pdf.internal.pageSize.getHeight();

  onProgress?.('Preparing searchable layer…', 0.02);

  for (let i = 0; i < result.pages.length; i++) {
    const page = result.pages[i];
    if (i > 0) pdf.addPage();

    onProgress?.(
      `Rendering page ${i + 1}/${result.pages.length}`,
      (i + 0.1) / result.pages.length
    );

    const img = await loadImage(page.imageDataUrl);

    const ratio = Math.min(pageW / img.width, pageH / img.height);
    const drawW = img.width * ratio;
    const drawH = img.height * ratio;
    const offX = (pageW - drawW) / 2;
    const offY = (pageH - drawH) / 2;

    pdf.addImage(page.imageDataUrl, 'JPEG', offX, offY, drawW, drawH, undefined, 'FAST');

    // Make text invisible
    pdf.setTextColor(0, 0, 0);
    try {
      (pdf as any).internal.write('3 Tr');
    } catch {}

    // Use the exact word-level boundaries extracted during the OCR step
    if (page.words && page.words.length > 0) {
      for (const w of page.words) {
        if (w.w <= 0 || w.h <= 0 || !w.text.trim()) continue;

        // Convert fractional coordinates back to points
        const px = offX + w.x * drawW;
        // jsPDF baseline is the bottom of the text, so we add the height
        const py = offY + (w.y + w.h * 0.85) * drawH; 
        const widthPt = w.w * drawW;
        const heightPt = w.h * drawH;

        const fontSize = Math.max(1, heightPt * 0.95);
        pdf.setFontSize(fontSize);

        let txtWidth: number;
        try {
          txtWidth = pdf.getTextWidth(w.text);
        } catch {
          txtWidth = widthPt;
        }
        
        // Squeeze or stretch the text to perfectly fit the invisible boundary box
        const scale = txtWidth > 0 ? (widthPt / txtWidth) * 100 : 100;

        try {
          pdf.text(w.text, px, py, {
            baseline: 'alphabetic',
            horizontalScale: Math.max(10, Math.min(400, scale)),
          } as any);
        } catch {
          try {
            pdf.text(w.text, px, py);
          } catch {}
        }
      }
    } else {
      // Fallback for older documents that were processed before word-tracking was added
      pdf.setFontSize(8);
      const lines = page.text.split(/\n/);
      let y = offY + 10;
      for (const line of lines) {
        if (y > offY + drawH - 6) break;
        try {
          pdf.text(line || ' ', offX + 4, y);
        } catch {}
        y += 10;
      }
    }

    // Restore visible rendering
    try {
      (pdf as any).internal.write('0 Tr');
    } catch {}
  }

  onProgress?.('Finalizing PDF…', 0.98);
  const blob = pdf.output('blob');
  onProgress?.('Done', 1);
  return blob;
}

/**
 * Build and trigger download of a searchable PDF (one-shot).
 */
export async function exportSearchablePdf(
  result: OcrResult,
  filename: string,
  onProgress?: (stage: string, progress: number) => void
) {
  const blob = await buildSearchablePdfBlob(result, onProgress);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.replace(/\.pdf$/i, '') + '_OCR.pdf';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Convert a Blob to a data URL so we can persist it in storage.
 */
export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

/**
 * Download a previously-saved data URL as a PDF.
 */
export function downloadPdfFromDataUrl(dataUrl: string, filename: string) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename.replace(/\.pdf$/i, '') + '_OCR.pdf';
  a.click();
}

export function exportPlainText(text: string, filename: string) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.replace(/\.[^/.]+$/, '') + '_OCR.txt';
  a.click();
  URL.revokeObjectURL(url);
}