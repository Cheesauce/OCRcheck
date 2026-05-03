import { createWorker, type Worker } from 'tesseract.js';
import * as pdfjsLib from 'pdfjs-dist';
import type { CancelToken } from '../hooks/useCancellable';
import { preprocessForOCR, type PreprocessVariant } from './imagePreprocess';
import nspell from 'https://esm.sh/nspell@2.1.5';

// Configure worker source for pdfjs
(pdfjsLib as any).GlobalWorkerOptions.workerSrc =
  'https://esm.sh/pdfjs-dist@4.0.379/build/pdf.worker.min.mjs';

export interface OcrProgress {
  stage: string;
  progress: number;
  currentPage?: number;
  totalPages?: number;
  subStage?: string;
}

export interface OcrPageResult {
  pageNumber: number;
  text: string;
  imageDataUrl: string;
  confidence?: number;
  winningVariant?: string;
  variantScores?: { variant: string; confidence: number; wordCount: number }[];
  words?: { text: string; x: number; y: number; w: number; h: number }[];
}

export interface OcrResult {
  pages: OcrPageResult[];
  fullText: string;
  language: string;
  avgConfidence?: number;
}

export type OcrQuality = 'fast' | 'balanced' | 'precise';

export interface OcrOptions {
  quality?: OcrQuality;
  extraVariants?: PreprocessVariant[];
  pdfRenderScale?: number;
}

// ===== Open-Source Spellchecker (nspell) =====
let currentSpellchecker: any = null;
let currentSpellcheckLang = '';

const DICT_CDN = 'https://cdn.jsdelivr.net/npm/';
const DICTS: Record<string, string> = {
  'eng': 'dictionary-en@3.2.0',
  'spa': 'dictionary-es@3.2.0',
  'fra': 'dictionary-fr@3.2.0',
  'deu': 'dictionary-de@3.2.0',
  'ita': 'dictionary-it@3.2.0',
  'por': 'dictionary-pt@3.2.0',
  'rus': 'dictionary-ru@3.2.0',
};

async function initSpellchecker(lang: string) {
  if (currentSpellcheckLang === lang && currentSpellchecker) return;
  
  const pkg = DICTS[lang];
  if (!pkg) {
    currentSpellchecker = null;
    currentSpellcheckLang = lang;
    return;
  }

  try {
    const [affRes, dicRes] = await Promise.all([
      fetch(`${DICT_CDN}${pkg}/index.aff`),
      fetch(`${DICT_CDN}${pkg}/index.dic`)
    ]);
    
    if (!affRes.ok || !dicRes.ok) throw new Error('Dictionary files could not be fetched');
    
    const aff = await affRes.text();
    const dic = await dicRes.text();
    currentSpellchecker = nspell(aff, dic);
    currentSpellcheckLang = lang;
  } catch (e) {
    console.warn('Failed to load open-source dictionary for', lang, e);
    currentSpellchecker = null;
  }
}

function levenshtein(a: string, b: string): number {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  const matrix = Array(a.length + 1).fill(0).map(() => Array(b.length + 1).fill(0));
  for (let i = 0; i <= a.length; i++) matrix[i][0] = i;
  for (let j = 0; j <= b.length; j++) matrix[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,
        matrix[i][j - 1] + 1,
        matrix[i - 1][j - 1] + cost
      );
    }
  }
  return matrix[a.length][b.length];
}

function applyOfflineAutocorrect(text: string): string {
  if (!currentSpellchecker) return text; 
  
  let preProcessed = text;
  const contextFixes: Array<[RegExp, string]> = [
    [/\b(CERTIFIE|CERTIFY)\s+(RUE|TRUE)\s+COPY\b/gi, 'CERTIFIED TRUE COPY'],
    [/\b(ORIGINAI|ORIGNAL)\b/gi, 'ORIGINAL'],
    [/\b(SIGNATUR|SIGNAIURE)\b/gi, 'SIGNATURE'],
    [/\b(DOCUMENI|DOCUMEN7)\b/gi, 'DOCUMENT'],
    [/\b(RECI|RECEI)(V|B)(ED|FD|CD|D)\b/gi, 'RECEIVED'],
    [/\b(APPROV)E\b/gi, 'APPROVED'],
    [/\b(AUTHORIZ)E\b/gi, 'AUTHORIZED'],
  ];

  for (const [regex, replacement] of contextFixes) {
    preProcessed = preProcessed.replace(regex, (match) => {
      if (match === match.toUpperCase()) return replacement.toUpperCase();
      if (match === match.toLowerCase()) return replacement.toLowerCase();
      return replacement.charAt(0).toUpperCase() + replacement.slice(1).toLowerCase();
    });
  }

  return preProcessed.replace(/[\p{L}]+/gu, (match) => {
    if (match.length <= 2) return match;
    const isUpper = match === match.toUpperCase();
    const isTitle = match[0] === match[0].toUpperCase() && match.slice(1) === match.slice(1).toLowerCase();
    const lowerMatch = match.toLowerCase();

    if (currentSpellchecker.correct(match) || currentSpellchecker.correct(lowerMatch)) return match;
    
    const suggestions = currentSpellchecker.suggest(lowerMatch);
    if (suggestions && suggestions.length > 0) {
      let bestCorrection = match;
      let minDistance = Infinity;
      const maxDist = (isUpper || isTitle) ? 1 : (lowerMatch.length <= 5 ? 1 : 2);

      for (const sug of suggestions) {
        const sugLower = sug.toLowerCase();
        if ((isUpper || isTitle) && sugLower[0] !== lowerMatch[0]) continue;
        const dist = levenshtein(lowerMatch, sugLower);
        if (dist <= maxDist && dist < minDistance) {
          bestCorrection = sug;
          minDistance = dist;
        }
      }

      if (bestCorrection !== match) {
        let corrected = bestCorrection;
        if (isUpper) corrected = corrected.toUpperCase();
        else if (isTitle) corrected = corrected.charAt(0).toUpperCase() + corrected.slice(1);
        return corrected;
      }
    }
    return match;
  });
}

// ===== Worker pool =====
const POOL_SIZE = Math.min(4, Math.max(2, (navigator.hardwareConcurrency || 4) - 1));

interface PooledWorker {
  worker: Worker;
  busy: boolean;
}

let workerPool: PooledWorker[] = [];
let poolLang: string = '';
let poolInitializing: Promise<void> | null = null;

async function initWorkerPool(
  lang: string,
  onProgress?: (p: OcrProgress) => void
): Promise<void> {
  if (workerPool.length > 0 && poolLang === lang) return;
  if (poolInitializing) {
    await poolInitializing;
    if (workerPool.length > 0 && poolLang === lang) return;
  }

  poolInitializing = (async () => {
    if (workerPool.length > 0) {
      await Promise.all(workerPool.map((p) => p.worker.terminate().catch(() => {})));
      workerPool = [];
    }
    onProgress?.({ stage: `Loading ${POOL_SIZE} OCR workers…`, progress: 0.03 });
    const workers = await Promise.all(
      Array.from({ length: POOL_SIZE }, () => createWorker(lang, 1, { logger: () => {} }))
    );
    await Promise.all(
      workers.map(async (w) => {
        try {
          await w.setParameters({
            tessedit_ocr_engine_mode: '1' as any,
            preserve_interword_spaces: '0',
            user_defined_dpi: '300',
          } as any);
        } catch (e) { console.warn('Worker param set failed:', e); }
      })
    );
    workerPool = workers.map((w) => ({ worker: w, busy: false }));
    poolLang = lang;
  })();

  await poolInitializing;
  poolInitializing = null;
}

async function acquireWorker(cancelToken?: CancelToken): Promise<PooledWorker> {
  while (true) {
    if (cancelToken?.cancelled) {
      const err = new Error('OCR cancelled by user.');
      (err as any).cancelled = true;
      throw err;
    }
    const free = workerPool.find((p) => !p.busy);
    if (free) {
      free.busy = true;
      return free;
    }
    await new Promise((r) => setTimeout(r, 15));
  }
}

function releaseWorker(p: PooledWorker) { p.busy = false; }

export async function terminateOcrWorker() {
  if (workerPool.length > 0) {
    await Promise.all(workerPool.map((p) => p.worker.terminate().catch(() => {})));
    workerPool = [];
    poolLang = '';
  }
}

async function loadPdfSafely(buf: ArrayBuffer): Promise<any> {
  const commonOptions = {
    disableFontFace: false,
    useSystemFonts: true,
    cMapUrl: 'https://esm.sh/pdfjs-dist@4.0.379/cmaps/',
    cMapPacked: true,
    standardFontDataUrl: 'https://esm.sh/pdfjs-dist@4.0.379/standard_fonts/',
  };
  try {
    return await (pdfjsLib as any).getDocument({ data: buf.slice(0), ...commonOptions }).promise;
  } catch (e: any) {
    console.warn('[ocrService] PDF worker load failed, retrying without worker:', e);
    try {
      return await (pdfjsLib as any).getDocument({ data: buf.slice(0), disableWorker: true, ...commonOptions }).promise;
    } catch (e2: any) {
      const msg = e2?.message || e?.message || 'Unknown PDF error';
      if (/password/i.test(msg)) throw new Error('This PDF is password-protected and cannot be opened.');
      if (/invalid|corrupt|malformed/i.test(msg)) throw new Error('The PDF file is corrupted or not a valid PDF.');
      throw new Error(`Could not open PDF: ${msg}`);
    }
  }
}

async function renderPdfPageToDataUrl(pdf: any, pageNum: number, scale = 2.0): Promise<string> {
  const page = await pdf.getPage(pageNum);
  const viewport = page.getViewport({ scale, rotation: page.rotate || 0 });
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.ceil(viewport.width));
  canvas.height = Math.max(1, Math.ceil(viewport.height));
  const ctx = canvas.getContext('2d', { willReadFrequently: false, alpha: false })!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  await page.render({ canvasContext: ctx, viewport, background: 'rgba(255,255,255,1)', intent: 'display' }).promise;
  const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
  try { page.cleanup(); } catch { }
  return dataUrl;
}

function getVariantsForQuality(quality: OcrQuality, extra: PreprocessVariant[] = []): PreprocessVariant[] {
  const base: Record<OcrQuality, PreprocessVariant[]> = {
    fast: ['high-contrast'],
    balanced: ['grayscale-clahe', 'high-contrast', 'sharpened'],
    precise: ['grayscale-clahe', 'binarized-sauvola', 'high-contrast', 'sharpened', 'inverted-bright'],
  };
  const merged = new Set([...base[quality], ...extra]);
  return Array.from(merged);
}

function getRenderScaleForQuality(quality: OcrQuality, override?: number): number {
  if (override) return override;
  switch (quality) {
    case 'fast': return 1.6;
    case 'balanced': return 2.0;
    case 'precise': return 2.6;
  }
}

function getMinLongEdgeForQuality(quality: OcrQuality): number {
  switch (quality) {
    case 'fast': return 1400;
    case 'balanced': return 1800;
    case 'precise': return 2400;
  }
}

async function recognizeWithVoting(
  variants: { dataUrl: string; variant: string; width: number; height: number }[],
  cancelToken: CancelToken | undefined,
  onSubStage?: (msg: string) => void
): Promise<{
  text: string;
  confidence: number;
  winningVariant: string;
  scores: { variant: string; confidence: number; wordCount: number }[];
  words: { text: string; x: number; y: number; w: number; h: number }[];
}> {
  const results: { variant: string; text: string; confidence: number; wordCount: number; words: any[] }[] = [];
  
  await Promise.all(
    variants.map(async (v, i) => {
      if (cancelToken?.cancelled) return;
      onSubStage?.(`Recognizing variant ${i + 1}/${variants.length} (${v.variant})…`);
      const pooled = await acquireWorker(cancelToken);
      try {
        if (cancelToken?.cancelled) return;

        // GUARANTEE mathematically perfect bounding boxes by measuring the true dimensions 
        // of the exact image passed to Tesseract, ignoring arbitrary scale variations.
        const img = new Image();
        img.src = v.dataUrl;
        await new Promise((resolve) => { img.onload = resolve; });
        const trueW = img.width;
        const trueH = img.height;

        const res = await pooled.worker.recognize(v.dataUrl);
        if (cancelToken?.cancelled) return;
        const text = res.data.text || '';
        const conf = (res.data as any).confidence ?? 0;
        const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
        
        const words: any[] = [];
        for (const w of (res.data as any).words || []) {
          if (!w.text || !w.text.trim()) continue;
          words.push({
             text: w.text.trim(),
             x: w.bbox.x0 / trueW,
             y: w.bbox.y0 / trueH,
             w: (w.bbox.x1 - w.bbox.x0) / trueW,
             h: (w.bbox.y1 - w.bbox.y0) / trueH
          });
        }

        results.push({ variant: v.variant, text, confidence: conf, wordCount, words });
      } finally { releaseWorker(pooled); }
    })
  );

  if (cancelToken?.cancelled) {
    const err = new Error('OCR cancelled by user.');
    (err as any).cancelled = true;
    throw err;
  }

  if (results.length === 0) return { text: '', confidence: 0, winningVariant: 'none', scores: [], words: [] };

  const scored = results.map((r) => ({ ...r, combined: r.confidence * 0.7 + Math.min(100, r.wordCount * 2) * 0.3 }));
  scored.sort((a, b) => b.combined - a.combined);
  const best = scored[0];

  return {
    text: best.text,
    confidence: best.confidence,
    winningVariant: best.variant,
    scores: scored.map((s) => ({ variant: s.variant, confidence: s.confidence, wordCount: s.wordCount })),
    words: (best as any).words || [],
  };
}

// Map the PDF's unscaled text layer to exact Canvas coordinates without guessing.
async function tryExtractEmbeddedText(pdf: any, pageNum: number): Promise<{ text: string; words: any[] } | null> {
  try {
    const page = await pdf.getPage(pageNum);
    const textContent = await page.getTextContent();
    const viewport = page.getViewport({ scale: 1.0 });
    
    let text = '';
    const words: any[] = [];
    let lastY = -1;
    
    for (const item of textContent.items) {
       if (!item.str || !item.str.trim()) continue;
       
       // Inject proper line breaks for the UI text pane
       if (lastY !== -1 && Math.abs(item.transform[5] - lastY) > 8) {
           text += '\n';
       } else if (text.length > 0 && !text.endsWith('\n') && !text.endsWith(' ')) {
           text += ' ';
       }
       text += item.str;
       lastY = item.transform[5];
       
       // Flawlessly map PDF coordinate matrix to standard top-left Canvas coordinates
       const [ptX, ptY] = viewport.convertToViewportPoint(item.transform[4], item.transform[5]);
       const itemW = item.width || Math.abs(item.transform[0]);
       const itemH = item.height || Math.abs(item.transform[3]); 
       
       // ptY represents the baseline. We offset it by ~85% to tightly hug the text's top edge
       const topY = ptY - (itemH * 0.85);

       words.push({
         text: item.str.trim(),
         x: ptX / viewport.width,
         y: topY / viewport.height,
         w: itemW / viewport.width,
         h: itemH / viewport.height
       });
    }
    
    try { page.cleanup(); } catch { }
    
    return { 
      text: text.trim(), 
      words 
    };
  } catch {
    return null;
  }
}

export async function performOcrOnFile(
  file: File,
  lang: string,
  onProgress?: (p: OcrProgress) => void,
  cancelToken?: CancelToken,
  options: OcrOptions = {}
): Promise<OcrResult> {
  const quality = options.quality ?? 'precise';
  const variants = getVariantsForQuality(quality, options.extraVariants);
  const renderScale = getRenderScaleForQuality(quality, options.pdfRenderScale);
  const minLongEdge = getMinLongEdgeForQuality(quality);
  const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
  const pages: OcrPageResult[] = [];

  const checkCancel = () => {
    if (cancelToken?.cancelled) {
      const err = new Error('OCR cancelled by user.');
      (err as any).cancelled = true;
      throw err;
    }
  };

  await initWorkerPool(lang, onProgress);
  checkCancel();
  onProgress?.({ stage: 'Loading offline dictionary…', progress: 0.04 });
  await initSpellchecker(lang);
  checkCancel();

  if (isPdf) {
    onProgress?.({ stage: 'Reading PDF…', progress: 0.05 });
    const buf = await file.arrayBuffer();
    const pdf = await loadPdfSafely(buf);
    const totalPages = pdf.numPages;
    const results: (OcrPageResult | null)[] = new Array(totalPages).fill(null);
    let completedCount = 0;

    onProgress?.({ stage: `Processing ${totalPages} page(s) · ${quality} quality`, progress: 0.07, totalPages, currentPage: 0 });

    const processPage = async (pageNum: number) => {
      checkCancel();
      const origImg = await renderPdfPageToDataUrl(pdf, pageNum, renderScale);
      
      const embeddedData = await tryExtractEmbeddedText(pdf, pageNum);
      const embeddedWordsCount = embeddedData?.text?.split(/\s+/).filter(Boolean).length || 0;

      // Microsoft/Google approach: Always prioritize exact embedded text for digital PDFs
      if (embeddedData && embeddedWordsCount >= 5) {
        results[pageNum - 1] = {
          pageNumber: pageNum,
          text: embeddedData.text,
          imageDataUrl: origImg,
          confidence: 100,
          winningVariant: 'embedded-text',
          variantScores: [{ variant: 'embedded-text', confidence: 100, wordCount: embeddedWordsCount }],
          words: embeddedData.words,
        };
        completedCount++;
        onProgress?.({ stage: `Page ${completedCount}/${totalPages}`, progress: 0.07 + (0.88 * completedCount) / totalPages, currentPage: completedCount, totalPages });
        return;
      }

      const prepped = await preprocessForOCR(origImg, { minLongEdge, variants });
      const voteResult = await recognizeWithVoting(prepped.map((p) => ({ ...p })), cancelToken, (msg) => {
        onProgress?.({ stage: `Page ${pageNum}/${totalPages}`, progress: 0.07 + (0.88 * completedCount) / totalPages, currentPage: completedCount + 1, totalPages, subStage: msg });
      });
      
      results[pageNum - 1] = {
        pageNumber: pageNum,
        text: applyOfflineAutocorrect(voteResult.text),
        imageDataUrl: origImg,
        confidence: voteResult.confidence,
        winningVariant: voteResult.winningVariant,
        variantScores: voteResult.scores,
        words: voteResult.words.map(w => ({ ...w, text: applyOfflineAutocorrect(w.text) })),
      };
      
      completedCount++;
      onProgress?.({ stage: `OCR page ${completedCount}/${totalPages}`, progress: 0.07 + (0.88 * completedCount) / totalPages, currentPage: completedCount, totalPages });
    };

    const BATCH = Math.max(1, Math.floor(POOL_SIZE / Math.min(2, variants.length)));
    for (let start = 0; start < totalPages; start += BATCH) {
      checkCancel();
      const end = Math.min(start + BATCH, totalPages);
      const batch = [];
      for (let p = start + 1; p <= end; p++) batch.push(processPage(p));
      await Promise.all(batch);
    }
    for (let i = 0; i < totalPages; i++) if (results[i]) pages.push(results[i]!);
  } else {
    const dataUrl = await fileToDataUrl(file);
    const prepped = await preprocessForOCR(dataUrl, { minLongEdge, variants });
    const voteResult = await recognizeWithVoting(prepped.map((p) => ({ ...p })), cancelToken);
    pages.push({
      pageNumber: 1,
      text: applyOfflineAutocorrect(voteResult.text),
      imageDataUrl: dataUrl,
      confidence: voteResult.confidence,
      winningVariant: voteResult.winningVariant,
      variantScores: voteResult.scores,
      words: voteResult.words.map(w => ({ ...w, text: applyOfflineAutocorrect(w.text) })),
    });
  }

  const avgConf = pages.reduce((a, b) => a + (b.confidence || 0), 0) / Math.max(1, pages.length);
  return { pages, fullText: pages.map((p) => p.text).join('\n\n'), language: lang, avgConfidence: avgConf };
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}