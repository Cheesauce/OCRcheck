// Persistent storage wrapper using window.persistentStorage (required in sandbox).
// IndexedDB is not available in this context, so we serialize collections to JSON
// and store them under fixed keys.

import { persistence } from '../utils/persistence';

export type SampleCategory = 'word' | 'logo' | 'signature' | 'stamp';

export interface TrainingSample {
  id: string;
  name: string;
  category: SampleCategory;
  label: string;
  imageData: string; // compressed base64 (JPEG)
  originalSize: number;
  compressedSize: number;
  createdAt: number;

  // OCR-derived text extracted from this sample (if any).
  ocrText?: string;
  ocrLang?: string;

  sourceAnnotationId?: string;
  origin?: 'upload' | 'annotation';
}

export interface OcrPageRecord {
  pageNumber: number;
  text: string;
  // Page image (rendered from original PDF or the input image itself).
  // This is what we display when the user clicks on a search hit.
  imageDataUrl: string;
  words?: { text: string; x: number; y: number; w: number; h: number }[];
}

export interface OcrDocument {
  id: string;
  name: string;
  pages: number;
  text: string;
  language: string;
  createdAt: number;

  // NEW — rich persistence so we can show pages and export later
  // without re-running OCR.
  pageRecords?: OcrPageRecord[];
  // Searchable PDF (all pages + invisible text layer) as data URL.
  // Built on save so the user can download it from the database anytime.
  searchablePdfDataUrl?: string;
  // MIME of the original file ('application/pdf' or 'image/*').
  originalMimeType?: string;
  // Original filename as uploaded.
  originalFileName?: string;
}

const KEY_SAMPLES = 'ocr_ai_studio.samples';
const KEY_DOCS = 'ocr_ai_studio.ocr_docs';

// --- in-memory cache to avoid repeated JSON parsing ---
let samplesCache: TrainingSample[] | null = null;
let docsCache: OcrDocument[] | null = null;

async function loadSamples(): Promise<TrainingSample[]> {
  if (samplesCache) return samplesCache;
  try {
    const raw = await persistence.getItem(KEY_SAMPLES);
    samplesCache = raw ? (JSON.parse(raw) as TrainingSample[]) : [];
  } catch (e) {
    console.warn('Failed to load samples, resetting:', e);
    samplesCache = [];
  }
  return samplesCache!;
}

async function saveSamples(list: TrainingSample[]): Promise<void> {
  samplesCache = list;
  await persistence.setItem(KEY_SAMPLES, JSON.stringify(list));
}

async function loadDocs(): Promise<OcrDocument[]> {
  if (docsCache) return docsCache;
  try {
    const raw = await persistence.getItem(KEY_DOCS);
    docsCache = raw ? (JSON.parse(raw) as OcrDocument[]) : [];
  } catch (e) {
    console.warn('Failed to load docs, resetting:', e);
    docsCache = [];
  }
  return docsCache!;
}

async function saveDocs(list: OcrDocument[]): Promise<void> {
  docsCache = list;
  try {
    await persistence.setItem(KEY_DOCS, JSON.stringify(list));
  } catch (e: any) {
    // If storage is too small, try to save without the heaviest fields
    console.warn('Doc save failed, retrying without searchable PDF payload:', e);
    const slim = list.map((d) => ({ ...d, searchablePdfDataUrl: undefined }));
    try {
      await persistence.setItem(KEY_DOCS, JSON.stringify(slim));
      docsCache = slim;
    } catch (e2) {
      console.warn('Doc save still failed even without PDF payload:', e2);
      throw e2;
    }
  }
}

// ------------ Samples API ------------
export async function addSample(sample: TrainingSample): Promise<void> {
  const list = await loadSamples();
  const idx = list.findIndex((s) => s.id === sample.id);
  if (idx >= 0) list[idx] = sample;
  else list.push(sample);
  await saveSamples(list);
}

export async function getAllSamples(): Promise<TrainingSample[]> {
  const list = await loadSamples();
  return list.slice();
}

export async function getSamplesByCategory(
  category: SampleCategory
): Promise<TrainingSample[]> {
  const list = await loadSamples();
  return list.filter((s) => s.category === category);
}

export async function deleteSample(id: string): Promise<void> {
  const list = await loadSamples();
  const next = list.filter((s) => s.id !== id);
  await saveSamples(next);
}

export async function clearSamples(): Promise<void> {
  await saveSamples([]);
}

export async function getUniqueLabels(): Promise<{ category: SampleCategory; label: string; count: number }[]> {
  const list = await loadSamples();
  const map = new Map<string, { category: SampleCategory; label: string; count: number }>();
  for (const s of list) {
    const key = `${s.category}::${s.label}`;
    const cur = map.get(key);
    if (cur) cur.count++;
    else map.set(key, { category: s.category, label: s.label, count: 1 });
  }
  return Array.from(map.values()).sort((a, b) => b.count - a.count);
}

// ------------ OCR Docs API ------------
export async function addOcrDoc(doc: OcrDocument): Promise<void> {
  const list = await loadDocs();
  const idx = list.findIndex((d) => d.id === doc.id);
  if (idx >= 0) list[idx] = doc;
  else list.push(doc);
  await saveDocs(list);
}

export async function getAllOcrDocs(): Promise<OcrDocument[]> {
  const list = await loadDocs();
  return list.slice();
}

export async function getOcrDocById(id: string): Promise<OcrDocument | null> {
  const list = await loadDocs();
  return list.find((d) => d.id === id) || null;
}

export async function deleteOcrDoc(id: string): Promise<void> {
  const list = await loadDocs();
  const next = list.filter((d) => d.id !== id);
  await saveDocs(next);
}

export async function updateOcrDocPage(
  docId: string,
  pageNumber: number,
  newText: string
): Promise<OcrDocument | null> {
  const list = await loadDocs();
  const idx = list.findIndex((d) => d.id === docId);
  if (idx < 0) return null;
  const doc = list[idx];
  if (!doc.pageRecords) return doc;
  const nextPages = doc.pageRecords.map((p) =>
    p.pageNumber === pageNumber ? { ...p, text: newText } : p
  );
  const nextDoc: OcrDocument = {
    ...doc,
    pageRecords: nextPages,
    text: nextPages.map((p) => p.text).join('\n\n'),
  };
  list[idx] = nextDoc;
  await saveDocs(list);
  return nextDoc;
}

export async function updateOcrDocPageWords(
  docId: string,
  pageNumber: number,
  newWords: any[]
): Promise<OcrDocument | null> {
  const list = await loadDocs();
  const idx = list.findIndex((d) => d.id === docId);
  if (idx < 0) return null;
  const doc = list[idx];
  if (!doc.pageRecords) return doc;
  const nextPages = doc.pageRecords.map((p) =>
    p.pageNumber === pageNumber ? { ...p, words: newWords } : p
  );
  const nextDoc: OcrDocument = {
    ...doc,
    pageRecords: nextPages,
  };
  list[idx] = nextDoc;
  await saveDocs(list);
  return nextDoc;
}