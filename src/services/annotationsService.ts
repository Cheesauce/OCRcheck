// Region annotations / user feedback service
// ============================================
// Stores per-region user comments, label corrections, and confirmations.
// This feedback loop makes the system progressively smarter:
//   - "Confirmed" regions become high-quality training samples.
//   - "Rejected" regions are tracked as negatives (to avoid similar mistakes).
//   - "Relabeled" regions are added as new training samples under the correct label.
//   - Free-text comments are kept for future audit / LLM-assisted reasoning.

import { persistence } from '../utils/persistence';
import type { SampleCategory } from './database';
import type { RegionMatch } from './recognitionService';

export type AnnotationVerdict = 'confirmed' | 'rejected' | 'relabeled' | 'note_only';

export interface RegionAnnotation {
  id: string;
  createdAt: number;
  updatedAt: number;

  // Where this came from
  sourceImageId: string;      // predict item id
  sourceImageName: string;
  sourceImageDataUrl: string; // full-size image (so we can crop later)

  // The region that was annotated
  region: RegionMatch;
  regionCropDataUrl: string | null; // cropped region snapshot

  // What the AI originally said
  originalLabel: string;
  originalCategory: SampleCategory;
  originalConfidence: number;

  // What the user said
  verdict: AnnotationVerdict;
  correctedLabel?: string;     // if verdict === 'relabeled'
  correctedCategory?: SampleCategory;
  comment: string;             // free-form user justification

  // Optional AI-assisted suggestion (from online engine)
  aiSuggestion?: {
    reasoning: string;
    suggestedLabel?: string;
    suggestedCategory?: SampleCategory;
    confidence?: number;
    source: string; // e.g. "Wikipedia", "Wikidata", "heuristic"
    fetchedAt: number;
  };

  // Did this annotation get converted into a training sample?
  promotedToTrainingSampleId?: string;
}

const KEY = 'ocr_ai_studio.annotations';

let cache: RegionAnnotation[] | null = null;

async function load(): Promise<RegionAnnotation[]> {
  if (cache) return cache;
  try {
    const raw = await persistence.getItem(KEY);
    cache = raw ? (JSON.parse(raw) as RegionAnnotation[]) : [];
  } catch (e) {
    console.warn('Failed to load annotations, resetting:', e);
    cache = [];
  }
  return cache!;
}

async function save(list: RegionAnnotation[]): Promise<void> {
  cache = list;
  await persistence.setItem(KEY, JSON.stringify(list));
}

export async function addAnnotation(a: RegionAnnotation): Promise<void> {
  const list = await load();
  const idx = list.findIndex((x) => x.id === a.id);
  if (idx >= 0) list[idx] = a;
  else list.push(a);
  await save(list);
}

export async function updateAnnotation(id: string, patch: Partial<RegionAnnotation>): Promise<RegionAnnotation | null> {
  const list = await load();
  const idx = list.findIndex((x) => x.id === id);
  if (idx < 0) return null;
  list[idx] = { ...list[idx], ...patch, updatedAt: Date.now() };
  await save(list);
  return list[idx];
}

export async function getAllAnnotations(): Promise<RegionAnnotation[]> {
  return (await load()).slice();
}

export async function getAnnotationsForSource(sourceImageId: string): Promise<RegionAnnotation[]> {
  const list = await load();
  return list.filter((a) => a.sourceImageId === sourceImageId);
}

export async function deleteAnnotation(id: string): Promise<void> {
  const list = await load();
  await save(list.filter((a) => a.id !== id));
}

export async function clearAnnotations(): Promise<void> {
  await save([]);
}

/**
 * Crop a rectangular region out of the full-size image and return a data URL.
 * Used when the user confirms/relabels a region — the crop becomes the
 * training sample.
 */
export async function cropRegion(
  imageDataUrl: string,
  region: RegionMatch,
  opts: { maxDim?: number; padding?: number } = {}
): Promise<string> {
  const maxDim = opts.maxDim ?? 99999;
  const padding = opts.padding ?? 0.05; // 5% padding

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const iw = img.naturalWidth;
      const ih = img.naturalHeight;

      const padX = region.width * padding * iw;
      const padY = region.height * padding * ih;

      const sx = Math.max(0, region.x * iw - padX);
      const sy = Math.max(0, region.y * ih - padY);
      const sw = Math.min(iw - sx, region.width * iw + padX * 2);
      const sh = Math.min(ih - sy, region.height * ih + padY * 2);

      // Scale down if too big
      let dw = sw;
      let dh = sh;
      if (Math.max(dw, dh) > maxDim) {
        const r = maxDim / Math.max(dw, dh);
        dw = Math.round(dw * r);
        dh = Math.round(dh * r);
      }

      const canvas = document.createElement('canvas');
      canvas.width = Math.round(dw);
      canvas.height = Math.round(dh);
      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

      resolve(canvas.toDataURL('image/jpeg', 0.88));
    };
    img.onerror = reject;
    img.src = imageDataUrl;
  });
}

/**
 * Stats helper for the Knowledge tab.
 */
export async function getAnnotationStats(): Promise<{
  total: number;
  confirmed: number;
  rejected: number;
  relabeled: number;
  notes: number;
  promoted: number;
}> {
  const list = await load();
  return {
    total: list.length,
    confirmed: list.filter((a) => a.verdict === 'confirmed').length,
    rejected: list.filter((a) => a.verdict === 'rejected').length,
    relabeled: list.filter((a) => a.verdict === 'relabeled').length,
    notes: list.filter((a) => a.verdict === 'note_only').length,
    promoted: list.filter((a) => !!a.promotedToTrainingSampleId).length,
  };
}