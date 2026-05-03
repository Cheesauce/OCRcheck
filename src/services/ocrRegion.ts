// OCR for a single region of a larger image.
// Used when training samples / annotations are saved — we extract any
// text inside the sample so it becomes searchable metadata that can
// later help disambiguate matches (e.g. "ACME" text inside a logo).

import { createWorker, type Worker } from 'tesseract.js';

let regionWorker: Worker | null = null;
let regionLang = '';

async function getWorker(lang: string): Promise<Worker> {
  if (regionWorker && regionLang === lang) return regionWorker;
  if (regionWorker) {
    try { await regionWorker.terminate(); } catch { /* noop */ }
    regionWorker = null;
  }
  regionWorker = await createWorker(lang, 1);
  regionLang = lang;
  return regionWorker;
}

export async function terminateRegionWorker() {
  if (regionWorker) {
    try { await regionWorker.terminate(); } catch { /* noop */ }
    regionWorker = null;
    regionLang = '';
  }
}

/**
 * Run Tesseract on a data URL and return detailed text and bounding boxes.
 */
export async function ocrDataUrlDetailed(
  dataUrl: string,
  lang: string = 'eng',
  timeoutMs = 30000
): Promise<{ text: string, words: { text: string; x0: number; y0: number; x1: number; y1: number }[] }> {
  try {
    const worker = await getWorker(lang);
    const p = worker.recognize(dataUrl);
    const timeout = new Promise<null>((resolve) => {
      setTimeout(() => resolve(null), timeoutMs);
    });
    const result = await Promise.race([p, timeout]);
    if (!result) return { text: '', words: [] };
    const data = (result as any).data;
    
    const words = (data.words || []).map((w: any) => ({
      text: w.text,
      x0: w.bbox.x0,
      y0: w.bbox.y0,
      x1: w.bbox.x1,
      y1: w.bbox.y1
    }));

    return {
      text: (data.text as string || '').replace(/\s+/g, ' ').trim(),
      words
    };
  } catch (e) {
    console.warn('Detailed Region OCR failed:', e);
    return { text: '', words: [] };
  }
}

/**
 * Run Tesseract on a data URL and return trimmed text (or empty string).
 * Wrapped in a try/catch so training never fails just because OCR fails.
 */
export async function ocrDataUrl(
  dataUrl: string,
  lang: string = 'eng',
  timeoutMs = 15000
): Promise<string> {
  try {
    const worker = await getWorker(lang);
    const p = worker.recognize(dataUrl);
    const timeout = new Promise<null>((resolve) => {
      setTimeout(() => resolve(null), timeoutMs);
    });
    const result = await Promise.race([p, timeout]);
    if (!result) return '';
    const text = ((result as any)?.data?.text as string) || '';
    return text.replace(/\s+/g, ' ').trim();
  } catch (e) {
    console.warn('Region OCR failed:', e);
    return '';
  }
}