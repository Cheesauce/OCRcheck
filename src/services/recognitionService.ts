import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { getAllSamples, type TrainingSample, type SampleCategory } from './database';
import { getAllAnnotations } from './annotationsService';
import { ocrDataUrl, ocrDataUrlDetailed } from './ocrRegion';
import {
  getWorkerPool,
  isWorkerPoolAvailable,
  type RecognitionWorkerPool,
} from './recognitionWorker';

let mobilenetModel: mobilenet.MobileNet | null = null;
let classifier: knnClassifier.KNNClassifier | null = null;
let workerPool: RecognitionWorkerPool | null = null;
let workerPoolFailed = false;
let isIndexed = false;

interface ClassPrototype {
  classKey: string;
  category: SampleCategory;
  label: string;
  meanEmbedding: Float32Array;
  meanPhash: Float32Array;
  meanEdgeDensity: number;
  meanAspectRatio: number;
  sampleCount: number;
  rotatedEmbeddings: Float32Array[];
  acceptanceThreshold: number;
  minEdgeDensity: number;
  textSnippets: string[];
  normalizedTexts: string[];
  primaryText: string;
}

interface NegativeExample {
  embedding: Float32Array;
  rejectedLabel: string;
  rejectedCategory: SampleCategory;
}

const prototypes = new Map<string, ClassPrototype>();
const negatives: NegativeExample[] = [];

export type RecognitionQuality = 'fast' | 'balanced' | 'precise' | 'exhaustive';

export interface RecognitionQualityConfig {
  key: RecognitionQuality;
  title: string;
  description: string;
  speed: string;
  inputSizes: number[];
  maxRegions: number;
  ocrOnRegions: boolean;
  ocrTopK: number;
  ocrTimeoutMs: number;
  acceptanceThresholdDelta: number;
  arToleranceBoost: number;
  fullImageOcrTimeoutMs: number;
  saliencyThreshold: number;
}

export const QUALITY_PRESETS: Record<RecognitionQuality, RecognitionQualityConfig> = {
  fast: {
    key: 'fast',
    title: 'Fast',
    description: 'Single-pass activation-map matching. Under 0.5s per image — usable for 1000+ page batches.',
    speed: '~0.2–0.5s per image',
    inputSizes: [224],
    maxRegions: 6,
    ocrOnRegions: false,
    ocrTopK: 0,
    ocrTimeoutMs: 2500,
    acceptanceThresholdDelta: 0.04,
    arToleranceBoost: 0.9,
    fullImageOcrTimeoutMs: 3000,
    saliencyThreshold: 0.04,
  },
  balanced: {
    key: 'balanced',
    title: 'Balanced',
    description: 'Two-scale pyramid + text matching on top regions. Excellent accuracy-to-speed ratio.',
    speed: '~0.5–1.2s per image',
    inputSizes: [224, 336],
    maxRegions: 10,
    ocrOnRegions: true,
    ocrTopK: 2,
    ocrTimeoutMs: 4000,
    acceptanceThresholdDelta: 0,
    arToleranceBoost: 1.0,
    fullImageOcrTimeoutMs: 4500,
    saliencyThreshold: 0.03,
  },
  precise: {
    key: 'precise',
    title: 'Precise',
    description: 'Three-scale pyramid with deeper OCR on top matches. Great for one-off scans.',
    speed: '~1.2–2.5s per image',
    inputSizes: [224, 336, 480],
    maxRegions: 14,
    ocrOnRegions: true,
    ocrTopK: 4,
    ocrTimeoutMs: 5500,
    acceptanceThresholdDelta: -0.03,
    arToleranceBoost: 1.15,
    fullImageOcrTimeoutMs: 6000,
    saliencyThreshold: 0.025,
  },
  exhaustive: {
    key: 'exhaustive',
    title: 'Exhaustive',
    description: 'Four-scale pyramid, fine activation grids, OCR on top 8 regions. Use for dense forms.',
    speed: '~3–6s per image',
    inputSizes: [224, 336, 480, 640],
    maxRegions: 22,
    ocrOnRegions: true,
    ocrTopK: 8,
    ocrTimeoutMs: 7000,
    acceptanceThresholdDelta: -0.05,
    arToleranceBoost: 1.3,
    fullImageOcrTimeoutMs: 25000,
    saliencyThreshold: 0.02,
  },
};

export async function ensureEngine(): Promise<void> {
  if (!mobilenetModel) {
    try {
      await tf.setBackend('webgl');
    } catch {
      // fallback
    }
    await tf.ready();
    mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
  }
  if (!classifier) {
    classifier = knnClassifier.create();
  }
  if (!workerPool && !workerPoolFailed && isWorkerPoolAvailable()) {
    getWorkerPool()
      .then((pool) => { workerPool = pool; })
      .catch((e) => {
        console.warn('Worker pool unavailable:', e);
        workerPoolFailed = true;
      });
  }
}

export function getWorkerPoolStatus(): { available: boolean; size: number; ready: boolean } {
  if (!isWorkerPoolAvailable()) return { available: false, size: 0, ready: false };
  if (workerPoolFailed) return { available: false, size: 0, ready: false };
  if (!workerPool) return { available: true, size: 0, ready: false };
  return { available: true, size: workerPool.size(), ready: workerPool.isReady() };
}

function normalizeText(s: string): string {
  return (s || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export async function indexAllSamples(
  onProgress?: (p: { done: number; total: number }) => void
): Promise<number> {
  await ensureEngine();
  classifier!.clearAllClasses();
  prototypes.clear();
  negatives.length = 0;

  const samples = await getAllSamples();

  const groups = new Map<string, TrainingSample[]>();
  for (const s of samples) {
    const k = `${s.category}::${s.label}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(s);
  }

  let done = 0;
  const total = samples.length;

  for (const [classKey, group] of groups) {
    const [category, label] = classKey.split('::');
    const embeddings: Float32Array[] = [];
    const rotatedEmbeddings: Float32Array[][] = [];
    const phashes: Float32Array[] = [];
    const edgeDensities: number[] = [];
    const aspectRatios: number[] = [];
    const textSnippets: string[] = [];
    const normalizedTexts: string[] = [];

    for (const s of group) {
      try {
        const features = await extractFeaturesWithRotations(s.imageData);
        const emb = tf.tensor(features.embedding);
        classifier!.addExample(emb, classKey);
        emb.dispose();

        embeddings.push(features.embedding);
        rotatedEmbeddings.push(features.rotatedEmbeddings);
        phashes.push(features.phash);
        edgeDensities.push(features.edgeDensity);
        aspectRatios.push(features.aspectRatio);
        if (s.ocrText && s.ocrText.trim().length >= 1) {
          textSnippets.push(s.ocrText.trim());
          normalizedTexts.push(normalizeText(s.ocrText));
        }
      } catch (e) {
        console.warn('Failed to index sample', s.id, e);
      }
      done++;
      onProgress?.({ done, total });
      if (done % 3 === 0) await new Promise((r) => setTimeout(r, 0));
    }

    if (embeddings.length > 0) {
      const NUM_ROT = rotatedEmbeddings[0].length;
      const avgRotated: Float32Array[] = [];
      for (let r = 0; r < NUM_ROT; r++) {
        const rAtIdx = rotatedEmbeddings.map((re) => re[r]);
        avgRotated.push(averageArrays(rAtIdx));
      }

      const meanEmb = averageArrays(embeddings);
      let intraMin = 1.0;
      let intraAvg = 0;
      for (const e of embeddings) {
        const sim = cosineSimilarity(e, meanEmb);
        intraMin = Math.min(intraMin, sim);
        intraAvg += sim;
      }
      intraAvg /= Math.max(1, embeddings.length);

      const baseThresh = category === 'word' ? 0.45 : 0.62;
      const upperThresh = category === 'word' ? 0.80 : 0.88;
      const mult = category === 'word' ? 0.70 : 0.85;
      const acceptanceThreshold = Math.max(
        baseThresh,
        Math.min(upperThresh, intraMin * mult)
      );
      const minEdgeDensity = Math.min(...edgeDensities) * 0.5;

      let primaryText = normalizeText(label);
      if (normalizedTexts.length > 0) {
        const counts = new Map<string, number>();
        for (const t of normalizedTexts) if (t) counts.set(t, (counts.get(t) || 0) + 1);
        let best = '';
        let bestCount = 0;
        for (const [t, c] of counts) {
          if (c > bestCount) { bestCount = c; best = t; }
        }
        if (best) primaryText = best;
      }

      const proto: ClassPrototype = {
        classKey,
        category: category as SampleCategory,
        label,
        meanEmbedding: meanEmb,
        meanPhash: averageArrays(phashes),
        meanEdgeDensity: edgeDensities.reduce((a, b) => a + b, 0) / edgeDensities.length,
        meanAspectRatio: aspectRatios.reduce((a, b) => a + b, 0) / aspectRatios.length,
        sampleCount: embeddings.length,
        rotatedEmbeddings: avgRotated,
        acceptanceThreshold,
        minEdgeDensity,
        textSnippets,
        normalizedTexts,
        primaryText,
      };
      prototypes.set(classKey, proto);

      for (let i = 0; i < Math.min(3, Math.ceil(embeddings.length / 2)); i++) {
        const protoTensor = tf.tensor(proto.meanEmbedding);
        classifier!.addExample(protoTensor, classKey);
        protoTensor.dispose();
      }
    }
  }

  try {
    const annotations = await getAllAnnotations();
    const rejectedOnes = annotations.filter(
      (a) => a.verdict === 'rejected' && a.regionCropDataUrl
    );
    for (const a of rejectedOnes) {
      try {
        const feat = await extractFeatures(a.regionCropDataUrl!);
        negatives.push({
          embedding: feat.embedding,
          rejectedLabel: a.originalLabel,
          rejectedCategory: a.originalCategory,
        });
      } catch (e) {
        console.warn('Failed to index negative', a.id, e);
      }
    }
  } catch (e) {
    console.warn('Negative indexing failed:', e);
  }

  isIndexed = true;
  return samples.length;
}

export async function addSampleToIndex(s: TrainingSample): Promise<void> {
  await ensureEngine();
  const features = await extractFeaturesWithRotations(s.imageData);
  const classKey = `${s.category}::${s.label}`;

  const emb = tf.tensor(features.embedding);
  classifier!.addExample(emb, classKey);
  emb.dispose();

  const existing = prototypes.get(classKey);
  if (existing) {
    const n = existing.sampleCount;
    existing.meanEmbedding = runningMean(existing.meanEmbedding, features.embedding, n);
    existing.meanPhash = runningMean(existing.meanPhash, features.phash, n);
    existing.meanEdgeDensity = (existing.meanEdgeDensity * n + features.edgeDensity) / (n + 1);
    existing.meanAspectRatio = (existing.meanAspectRatio * n + features.aspectRatio) / (n + 1);
    for (let r = 0; r < existing.rotatedEmbeddings.length; r++) {
      existing.rotatedEmbeddings[r] = runningMean(
        existing.rotatedEmbeddings[r],
        features.rotatedEmbeddings[r],
        n
      );
    }
    existing.sampleCount = n + 1;
    existing.acceptanceThreshold = Math.max(
      s.category === 'word' ? 0.45 : 0.62,
      existing.acceptanceThreshold - 0.005
    );
    if (s.ocrText && s.ocrText.trim().length >= 1) {
      existing.textSnippets.push(s.ocrText.trim());
      existing.normalizedTexts.push(normalizeText(s.ocrText));
    }
  } else {
    const baseThresh = s.category === 'word' ? 0.45 : 0.72;
    prototypes.set(classKey, {
      classKey,
      category: s.category,
      label: s.label,
      meanEmbedding: features.embedding,
      meanPhash: features.phash,
      meanEdgeDensity: features.edgeDensity,
      meanAspectRatio: features.aspectRatio,
      sampleCount: 1,
      rotatedEmbeddings: features.rotatedEmbeddings,
      acceptanceThreshold: baseThresh,
      minEdgeDensity: features.edgeDensity * 0.5,
      textSnippets: s.ocrText && s.ocrText.trim().length >= 1 ? [s.ocrText.trim()] : [],
      normalizedTexts: s.ocrText && s.ocrText.trim().length >= 1 ? [normalizeText(s.ocrText)] : [],
      primaryText: s.ocrText ? normalizeText(s.ocrText) : normalizeText(s.label),
    });
  }
  isIndexed = true;
}

export async function addNegativeExample(
  regionCropDataUrl: string,
  rejectedLabel: string,
  rejectedCategory: SampleCategory
): Promise<void> {
  await ensureEngine();
  const feat = await extractFeatures(regionCropDataUrl);
  negatives.push({
    embedding: feat.embedding,
    rejectedLabel,
    rejectedCategory,
  });
}

export function isIndexReady(): boolean {
  return isIndexed && (classifier?.getNumClasses() ?? 0) > 0;
}

export function getIndexStats(): { classes: number; examples: number; negatives: number; wordClasses: number } {
  if (!classifier) return { classes: 0, examples: 0, negatives: 0, wordClasses: 0 };
  const counts = classifier.getClassExampleCount();
  const classes = Object.keys(counts).length;
  const examples = Object.values(counts).reduce((a, b) => a + b, 0);
  let wordClasses = 0;
  for (const p of prototypes.values()) {
    if (p.category === 'word') wordClasses++;
  }
  return { classes, examples, negatives: negatives.length, wordClasses };
}

export function getPrototypes(): ClassPrototype[] {
  return Array.from(prototypes.values());
}

export async function getCompositeImage(
  category: SampleCategory,
  label: string
): Promise<string | null> {
  const classKey = `${category}::${label}`;
  const proto = prototypes.get(classKey);
  if (!proto) return null;

  const samples = await getAllSamples();
  const group = samples.filter((s) => s.category === category && s.label === label);
  if (group.length === 0) return null;
  if (group.length === 1) return group[0].imageData;

  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < group.length; i++) {
    try {
      const feat = await extractFeaturesWithRotations(group[i].imageData);
      const d = l2Distance(feat.embedding, proto.meanEmbedding);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    } catch {
      /* skip */
    }
  }
  return group[bestIdx].imageData;
}

export interface RegionMatch {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  category: string;
  confidence: number;
  rotation?: number;
  textMatch?: {
    ocrText: string;
    matchedText: string;
    similarity: number;
  };
  priority?: number;
}

export interface PredictionResult {
  label: string;
  category: string;
  confidence: number;
  scores: {
    mobilenet: number;
    phash: number;
    edgeDensity: number;
    textMatch: number;
    ensemble: number;
    colorHist?: number;
  };
  all: { label: string; category: string; confidence: number }[];
  regions?: RegionMatch[];
  meta?: {
    quality: RecognitionQuality;
    windowsScanned: number;
    parallelWorkers: number;
    elapsedMs: number;
  };
}

function categoryPriority(cat: string): number {
  switch (cat) {
    case 'word': return 100;
    case 'logo': return 50;
    case 'stamp': return 40;
    case 'signature': return 30;
    default: return 10;
  }
}

function textSimilarity(a: string, b: string): number {
  if (!a || !b) return 0;
  const na = normalizeText(a);
  const nb = normalizeText(b);
  if (!na || !nb) return 0;
  if (na === nb) return 1.0;

  if (nb.length >= 3 && na.includes(nb)) return 0.95;
  if (na.length >= 3 && nb.includes(na)) return 0.92;

  const wa = new Set(na.split(' ').filter((w) => w.length >= 2));
  const wb = new Set(nb.split(' ').filter((w) => w.length >= 2));
  if (wa.size === 0 || wb.size === 0) return charSimilarity(na, nb);
  let inter = 0;
  for (const w of wa) if (wb.has(w)) inter++;
  const jaccard = inter / (wa.size + wb.size - inter);
  if (jaccard > 0.5) return 0.6 + jaccard * 0.35;
  const charSim = charSimilarity(na, nb);
  return Math.max(jaccard, charSim);
}

function charSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 0;
  const distance = levenshtein(a, b, Math.min(20, Math.ceil(maxLen * 0.5)));
  if (distance < 0) return 0;
  return Math.max(0, 1 - distance / maxLen);
}

function levenshtein(a: string, b: string, maxDist: number): number {
  const m = a.length;
  const n = b.length;
  if (Math.abs(m - n) > maxDist) return -1;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = new Array(n + 1);
  let curr = new Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    let minInRow = curr[0];
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
      if (curr[j] < minInRow) minInRow = curr[j];
    }
    if (minInRow > maxDist) return -1;
    [prev, curr] = [curr, prev];
  }
  return prev[n];
}

function bestTextMatchForProto(
  ocrText: string,
  proto: ClassPrototype
): { similarity: number; matchedText: string } {
  if (!ocrText || proto.normalizedTexts.length === 0) {
    if (proto.primaryText) {
      const sim = textSimilarity(ocrText, proto.primaryText);
      return { similarity: sim, matchedText: proto.primaryText };
    }
    return { similarity: 0, matchedText: '' };
  }
  let bestSim = 0;
  let bestText = proto.primaryText || '';
  for (const snippet of proto.normalizedTexts) {
    const sim = textSimilarity(ocrText, snippet);
    if (sim > bestSim) { bestSim = sim; bestText = snippet; }
  }
  const primSim = textSimilarity(ocrText, proto.primaryText);
  if (primSim > bestSim) { bestSim = primSim; bestText = proto.primaryText; }
  return { similarity: bestSim, matchedText: bestText };
}

export async function predictFromDataUrl(
  dataUrl: string,
  k = 5,
  onProgress?: (stage: string, progress: number) => void,
  quality: RecognitionQuality = 'balanced',
  embeddedText?: { text: string; words: any[] }
): Promise<PredictionResult | null> {
  await ensureEngine();
  if (!classifier || classifier.getNumClasses() === 0) return null;
  if (prototypes.size === 0) return null;

  const config = QUALITY_PRESETS[quality];
  const startTime = performance.now();
  let windowsScanned = 0;

  onProgress?.(`Preparing image (${config.title} mode)…`, 0.05);

  const img = await loadImg(dataUrl);
  const origW = img.naturalWidth;
  const origH = img.naturalHeight;

  // --- NEW OCR-FIRST PIPELINE ---
  onProgress?.(`Extracting full-page text (OCR First)…`, 0.10);

  const wordProtos = Array.from(prototypes.values()).filter((p) => p.category === 'word');
  const ocrRegions: RegionMatch[] = [];
  let fullImageText = '';

  if (config.ocrOnRegions) {
    try {
      let ocrResult: { text: string; words: any[] } | undefined;
      
      if (embeddedText && embeddedText.words.length >= 5) {
        onProgress?.(`Using embedded PDF text layer…`, 0.10);
        ocrResult = embeddedText;
      } else {
        onProgress?.(`Extracting full-page text (OCR First)…`, 0.10);
        ocrResult = await ocrDataUrlDetailed(dataUrl, 'eng', config.fullImageOcrTimeoutMs || 25000);
      }

      if (ocrResult) {
        fullImageText = ocrResult.text;
        
        if (wordProtos.length > 0) {
          for (const w of ocrResult.words) {
            if (w.text.length < 2) continue;
            
            let bestProto: ClassPrototype | null = null;
            let bestSim = 0;
            let bestMatchedText = '';

            for (const wp of wordProtos) {
              const { similarity, matchedText } = bestTextMatchForProto(w.text, wp);
              if (similarity > bestSim) {
                bestSim = similarity;
                bestProto = wp;
                bestMatchedText = matchedText;
              }
            }

            if (bestProto && bestSim >= 0.55) {
              const nx = w.x !== undefined ? w.x : (w.x0 / origW);
              const ny = w.y !== undefined ? w.y : (w.y0 / origH);
              const nw = w.w !== undefined ? w.w : ((w.x1 - w.x0) / origW);
              const nh = w.h !== undefined ? w.h : ((w.y1 - w.y0) / origH);
              
              if (nw > 0 && nh > 0) {
                ocrRegions.push({
                  x: nx,
                  y: ny,
                  width: nw,
                  height: nh,
                  label: bestProto.label,
                  category: 'word',
                  confidence: Math.max(0.6, 0.6 + bestSim * 0.35),
                  priority: categoryPriority('word'),
                  textMatch: {
                    ocrText: w.text,
                    matchedText: bestMatchedText,
                    similarity: bestSim,
                  },
                });
              }
            }
          }
        }
      }
    } catch { /* ignore OCR fail */ }
  }

  const normalizedFullText = normalizeText(fullImageText);

  onProgress?.('Extracting visual activation maps (AI)…', 0.35);

  const pyramidCandidates: Candidate[] = [];
  for (let si = 0; si < config.inputSizes.length; si++) {
    const inputSize = config.inputSizes[si];
    const scaleCandidates = await extractActivationMapCandidates(
      img, inputSize, config
    );
    pyramidCandidates.push(...scaleCandidates);
    windowsScanned += scaleCandidates.length;
    onProgress?.(
      `Scale ${si + 1}/${config.inputSizes.length} · ${windowsScanned} cells`,
      0.35 + 0.45 * ((si + 1) / config.inputSizes.length)
    );
  }

  const byClass = new Map<string, Candidate[]>();
  for (const c of pyramidCandidates) {
    if (!byClass.has(c.classKey)) byClass.set(c.classKey, []);
    byClass.get(c.classKey)!.push(c);
  }

  const kept: Candidate[] = [];
  for (const [, list] of byClass) {
    list.sort((a, b) => b.score - a.score);
    const suppressed = new Set<number>();
    let keptForClass = 0;
    for (let i = 0; i < list.length; i++) {
      if (suppressed.has(i)) continue;
      kept.push(list[i]);
      keptForClass++;
      if (keptForClass >= 6) {
        for (let j = i + 1; j < list.length; j++) suppressed.add(j);
        break;
      }
      for (let j = i + 1; j < list.length; j++) {
        if (suppressed.has(j)) continue;
        if (iou(list[i], list[j]) > 0.3) {
          suppressed.add(j);
        }
      }
    }
  }

  kept.sort((a, b) => b.score - a.score);
  const finalKept: Candidate[] = [];
  for (const cand of kept) {
    let overlapsHigher = false;
    for (const k of finalKept) {
      const sameClass = k.classKey === cand.classKey;
      const iouThresh = sameClass ? 0.4 : 0.7;
      if (iou(cand, k) > iouThresh) { overlapsHigher = true; break; }
    }
    if (!overlapsHigher) finalKept.push(cand);
  }

  const aiRegions: RegionMatch[] = finalKept.slice(0, config.maxRegions).map((c) => {
    const [cat, lab] = c.classKey.split('::');
    return {
      x: c.nx,
      y: c.ny,
      width: c.nw,
      height: c.nh,
      label: lab,
      category: cat,
      confidence: c.score,
      rotation: 0,
      priority: categoryPriority(cat),
    };
  });

  let regions = [...ocrRegions, ...aiRegions];

  const mergedRegions: RegionMatch[] = [];
  regions.sort((a, b) => {
    const aPri = (a.priority ?? 0) * 0.01 + (a.textMatch?.similarity ?? 0) * 0.3 + a.confidence;
    const bPri = (b.priority ?? 0) * 0.01 + (b.textMatch?.similarity ?? 0) * 0.3 + b.confidence;
    return bPri - aPri;
  });

  for (const r of regions) {
    let overlaps = false;
    for (const m of mergedRegions) {
      if (iouRegions(r, m) > 0.4) { overlaps = true; break; }
    }
    if (!overlaps) mergedRegions.push(r);
  }
  regions = mergedRegions;

  onProgress?.('Analyzing overall image…', 0.85);
  const queryFeatures = await extractFeatures(dataUrl);

  const emb = tf.tensor(queryFeatures.embedding);
  const effectiveK = Math.min(k, Math.max(1, minClassCount()));
  const knnRes = await classifier.predictClass(emb, effectiveK);
  emb.dispose();

  const mobilenetKnnVote = new Map<string, number>();
  for (const [k2, v] of Object.entries(knnRes.confidences)) {
    mobilenetKnnVote.set(k2, v as number);
  }

  const mobilenetCosine = new Map<string, number>();
  const phashScores = new Map<string, number>();
  const edgeScores = new Map<string, number>();
  const textScores = new Map<string, number>();

  for (const proto of prototypes.values()) {
    const cos = cosineSimilarity(queryFeatures.embedding, proto.meanEmbedding);
    mobilenetCosine.set(proto.classKey, Math.max(0, Math.min(1, cos)));
    const phashDist = hammingLike(queryFeatures.phash, proto.meanPhash);
    phashScores.set(proto.classKey, Math.max(0, Math.min(1, 1 - phashDist)));
    const edgeDiff = Math.abs(queryFeatures.edgeDensity - proto.meanEdgeDensity);
    edgeScores.set(proto.classKey, Math.max(0, Math.min(1, 1 - edgeDiff * 4)));
    const { similarity } = bestTextMatchForProto(normalizedFullText, proto);
    textScores.set(proto.classKey, similarity);
  }

  const ensemble: { classKey: string; score: number; breakdown: any; priority: number }[] = [];
  for (const proto of prototypes.values()) {
    const mk = mobilenetKnnVote.get(proto.classKey) || 0;
    const mc = mobilenetCosine.get(proto.classKey) || 0;
    const ph = phashScores.get(proto.classKey) || 0;
    const ed = edgeScores.get(proto.classKey) || 0;
    const tx = textScores.get(proto.classKey) || 0;

    let W_KNN = 0.25, W_MC = 0.35, W_PH = 0.15, W_ED = 0.05, W_TX = 0.20;
    if (proto.category === 'word') {
      W_TX = 0.70; W_KNN = 0.08; W_MC = 0.14; W_PH = 0.04; W_ED = 0.04;
    }

    const score = mk * W_KNN + mc * W_MC + ph * W_PH + ed * W_ED + tx * W_TX;
    ensemble.push({
      classKey: proto.classKey,
      score,
      priority: categoryPriority(proto.category),
      breakdown: { mobilenetKnn: mk, mobilenetCos: mc, phash: ph, edge: ed, text: tx },
    });
  }

  ensemble.sort((a, b) => {
    const aBoost = (a.priority / 100) * (a.breakdown.text > 0.5 ? 1.5 : 1.0);
    const bBoost = (b.priority / 100) * (b.breakdown.text > 0.5 ? 1.5 : 1.0);
    return b.score * (0.5 + 0.5 * bBoost) - a.score * (0.5 + 0.5 * aBoost);
  });

  if (ensemble.length === 0) return null;

  let top = ensemble[0];
  if (regions.length > 0) {
    const best = regions[0];
    const bestKey = `${best.category}::${best.label}`;
    const match = ensemble.find((e) => e.classKey === bestKey);
    if (match) top = match;
    top = { ...top, score: best.confidence };
  }

  const [category, label] = top.classKey.split('::');
  const all = ensemble.map((e) => {
    const [cat, lab] = e.classKey.split('::');
    return { label: lab, category: cat, confidence: e.score };
  });

  const elapsedMs = performance.now() - startTime;
  onProgress?.('Finalizing…', 1);

  return {
    label,
    category,
    confidence: top.score,
    scores: {
      mobilenet: (top.breakdown.mobilenetKnn + top.breakdown.mobilenetCos) / 2,
      phash: top.breakdown.phash,
      edgeDensity: top.breakdown.edge,
      textMatch: top.breakdown.text,
      ensemble: top.score,
      colorHist: 0
    },
    all,
    regions,
    meta: {
      quality,
      windowsScanned,
      parallelWorkers: workerPool?.size() || 1,
      elapsedMs,
    },
  };
}

interface Candidate {
  nx: number; ny: number; nw: number; nh: number;
  classKey: string;
  score: number;
}

async function extractActivationMapCandidates(
  img: HTMLImageElement,
  inputSize: number,
  config: RecognitionQualityConfig
): Promise<Candidate[]> {
  if (!mobilenetModel) return [];

  const origW = img.naturalWidth;
  const origH = img.naturalHeight;

  const ratio = Math.min(inputSize / origW, inputSize / origH);
  const drawW = Math.round(origW * ratio);
  const drawH = Math.round(origH * ratio);
  const offX = Math.floor((inputSize - drawW) / 2);
  const offY = Math.floor((inputSize - drawH) / 2);

  const canvas = document.createElement('canvas');
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext('2d', { willReadFrequently: false, alpha: false })!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, inputSize, inputSize);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, offX, offY, drawW, drawH);

  const saliencyGrid = computeQuickSaliency(ctx, inputSize);

  const candidates: Candidate[] = [];

  const cellFractions: number[] = [];
  if (inputSize >= 640) cellFractions.push(0.20, 0.35, 0.55);
  else if (inputSize >= 480) cellFractions.push(0.25, 0.45, 0.70);
  else if (inputSize >= 336) cellFractions.push(0.35, 0.55, 0.80);
  else cellFractions.push(0.45, 0.75, 1.0);

  interface Window {
    x: number; y: number; w: number; h: number; sal: number;
  }
  const windows: Window[] = [];

  for (const frac of cellFractions) {
    const cellSize = Math.round(inputSize * frac);
    if (cellSize < 48) continue;
    const stride = Math.max(16, Math.round(cellSize * 0.5));

    const arSet = new Set<number>();
    arSet.add(1.0);
    for (const p of prototypes.values()) {
      const ar = clamp(p.meanAspectRatio, 0.3, 4.0);
      arSet.add(Math.round(ar * 2) / 2);
    }

    for (const ar of arSet) {
      let cw: number, ch: number;
      if (ar >= 1) {
        cw = cellSize;
        ch = Math.round(cellSize / ar);
      } else {
        cw = Math.round(cellSize * ar);
        ch = cellSize;
      }
      if (cw < 40 || ch < 40) continue;

      for (let y = 0; y + ch <= inputSize; y += stride) {
        for (let x = 0; x + cw <= inputSize; x += stride) {
          const ix0 = Math.max(x, offX);
          const iy0 = Math.max(y, offY);
          const ix1 = Math.min(x + cw, offX + drawW);
          const iy1 = Math.min(y + ch, offY + drawH);
          const iw = ix1 - ix0;
          const ih = iy1 - iy0;
          if (iw <= 0 || ih <= 0) continue;
          if ((iw * ih) / (cw * ch) < 0.6) continue;

          const sal = sampleSaliency(saliencyGrid, x, y, cw, ch);
          if (sal < config.saliencyThreshold) continue;
          windows.push({ x, y, w: cw, h: ch, sal });
        }
      }
    }
  }

  if (windows.length === 0) return [];

  const MAX_WINDOWS = 200;
  windows.sort((a, b) => b.sal - a.sal);
  const finalWindows = windows.slice(0, MAX_WINDOWS);

  const BATCH = 64;
  const MOBILENET_INPUT = 224;

  for (let start = 0; start < finalWindows.length; start += BATCH) {
    const batch = finalWindows.slice(start, start + BATCH);
    const embeddings = await embedWindowsBatched(canvas, batch, MOBILENET_INPUT);

    for (let i = 0; i < batch.length; i++) {
      const win = batch[i];
      const embCrop = embeddings[i];

      const originalX = Math.max(0, (win.x - offX) / drawW);
      const originalY = Math.max(0, (win.y - offY) / drawH);
      const originalW = Math.min(1 - originalX, win.w / drawW);
      const originalH = Math.min(1 - originalY, win.h / drawH);
      if (originalW <= 0.02 || originalH <= 0.02) continue;

      for (const proto of prototypes.values()) {
        const windowAR = win.w / win.h;
        const arRatio = Math.max(windowAR, proto.meanAspectRatio) /
                        Math.min(windowAR, proto.meanAspectRatio);
        const baseArLimit = proto.category === 'word' ? 3.0 : 2.2;
        const arLimit = baseArLimit * config.arToleranceBoost;
        if (arRatio > arLimit) continue;

        let bestSim = -1;
        let bestRot = 0;
        const rotLabels = [0, 90, 180, 270, -1];
        for (let r = 0; r < proto.rotatedEmbeddings.length; r++) {
          const sim = cosineSimilarity(embCrop, proto.rotatedEmbeddings[r]);
          if (sim > bestSim) { bestSim = sim; bestRot = rotLabels[r] ?? 0; }
        }

        const threshold = proto.acceptanceThreshold + config.acceptanceThresholdDelta;
        if (bestSim < threshold) continue;

        let negSuppressed = false;
        for (const neg of negatives) {
          if (neg.rejectedLabel !== proto.label) continue;
          if (neg.rejectedCategory !== proto.category) continue;
          const negSim = cosineSimilarity(embCrop, neg.embedding);
          if (negSim >= bestSim - 0.02 && negSim > 0.82) { negSuppressed = true; break; }
        }
        if (negSuppressed) continue;

        candidates.push({
          nx: originalX,
          ny: originalY,
          nw: originalW,
          nh: originalH,
          classKey: proto.classKey,
          score: bestSim,
        });
      }
    }

    if (start + BATCH < finalWindows.length) {
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  return candidates;
}

async function embedWindowsBatched(
  source: HTMLCanvasElement,
  windows: { x: number; y: number; w: number; h: number }[],
  inputSize: number
): Promise<Float32Array[]> {
  if (windows.length === 0) return [];

  const pool = (!workerPoolFailed && workerPool && workerPool.isReady()) ? workerPool : null;

  if (pool) {
    const bitmaps: ImageBitmap[] = [];
    const tmp = document.createElement('canvas');
    tmp.width = inputSize;
    tmp.height = inputSize;
    const tctx = tmp.getContext('2d')!;

    for (const w of windows) {
      tctx.fillStyle = '#ffffff';
      tctx.fillRect(0, 0, inputSize, inputSize);
      tctx.drawImage(source, w.x, w.y, w.w, w.h, 0, 0, inputSize, inputSize);
      const bmp = await createImageBitmap(tmp);
      bitmaps.push(bmp);
    }

    try {
      return await pool.embedBatch(bitmaps);
    } catch (e) {
      console.warn('Worker embed failed, falling back to main thread:', e);
    }
  }

  return embedBatchMainThread(source, windows, inputSize);
}

async function embedBatchMainThread(
  source: HTMLCanvasElement,
  windows: { x: number; y: number; w: number; h: number }[],
  inputSize: number
): Promise<Float32Array[]> {
  if (windows.length === 0) return [];

  const tensors: tf.Tensor3D[] = [];
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = inputSize;
  cropCanvas.height = inputSize;
  const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true })!;

  for (const win of windows) {
    cropCtx.fillStyle = '#ffffff';
    cropCtx.fillRect(0, 0, inputSize, inputSize);
    cropCtx.drawImage(source, win.x, win.y, win.w, win.h, 0, 0, inputSize, inputSize);
    const t = tf.browser.fromPixels(cropCanvas);
    tensors.push(t);
  }

  const batched = tf.stack(tensors) as tf.Tensor4D;
  tensors.forEach((t) => t.dispose());

  const logits = (mobilenetModel as any).infer(batched, 'conv_preds') as tf.Tensor;
  const allData = await logits.data();
  const dim = logits.shape[logits.shape.length - 1] as number;

  batched.dispose();
  logits.dispose();

  const results: Float32Array[] = [];
  for (let i = 0; i < windows.length; i++) {
    results.push(new Float32Array(allData.buffer, i * dim * 4, dim).slice());
  }
  return results;
}

function computeQuickSaliency(
  ctx: CanvasRenderingContext2D,
  size: number
): { grid: Float32Array; gridN: number; cell: number } {
  const GRID = 24;
  const cell = size / GRID;
  const grid = new Float32Array(GRID * GRID);
  const id = ctx.getImageData(0, 0, size, size);
  const data = id.data;
  const stride = size * 4;

  for (let gy = 0; gy < GRID; gy++) {
    const y0 = Math.floor(gy * cell);
    const y1 = Math.floor((gy + 1) * cell);
    for (let gx = 0; gx < GRID; gx++) {
      const x0 = Math.floor(gx * cell);
      const x1 = Math.floor((gx + 1) * cell);

      let ink = 0;
      let total = 0;
      for (let y = y0; y < y1; y += 3) {
        const rowStart = y * stride;
        for (let x = x0; x < x1; x += 3) {
          const i = rowStart + x * 4;
          const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          if (lum < 210) ink++;
          total++;
        }
      }
      grid[gy * GRID + gx] = total > 0 ? ink / total : 0;
    }
  }
  return { grid, gridN: GRID, cell };
}

function sampleSaliency(
  sal: { grid: Float32Array; gridN: number; cell: number },
  x: number, y: number, w: number, h: number
): number {
  const gx0 = Math.max(0, Math.floor(x / sal.cell));
  const gy0 = Math.max(0, Math.floor(y / sal.cell));
  const gx1 = Math.min(sal.gridN - 1, Math.ceil((x + w) / sal.cell));
  const gy1 = Math.min(sal.gridN - 1, Math.ceil((y + h) / sal.cell));
  let sum = 0;
  let count = 0;
  for (let gy = gy0; gy <= gy1; gy++) {
    for (let gx = gx0; gx <= gx1; gx++) {
      sum += sal.grid[gy * sal.gridN + gx];
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

export function resetIndex() {
  classifier?.clearAllClasses();
  prototypes.clear();
  negatives.length = 0;
  isIndexed = false;
}

function minClassCount(): number {
  if (!classifier) return 1;
  const counts = Object.values(classifier.getClassExampleCount());
  return counts.length ? Math.min(...(counts as number[])) : 1;
}

function iouRegions(a: RegionMatch, b: RegionMatch): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const ua = a.width * a.height + b.width * b.height - inter;
  return inter / ua;
}

function iou(
  a: { nx: number; ny: number; nw: number; nh: number },
  b: { nx: number; ny: number; nw: number; nh: number }
): number {
  const x1 = Math.max(a.nx, b.nx);
  const y1 = Math.max(a.ny, b.ny);
  const x2 = Math.min(a.nx + a.nw, b.nx + b.nw);
  const y2 = Math.min(a.ny + a.nh, b.ny + b.nh);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const ua = a.nw * a.nh + b.nw * b.nh - inter;
  return inter / ua;
}

interface ImageFeatures {
  embedding: Float32Array;
  phash: Float32Array;
  edgeDensity: number;
  aspectRatio: number;
}

interface ImageFeaturesWithRot extends ImageFeatures {
  rotatedEmbeddings: Float32Array[];
}

async function extractFeatures(dataUrl: string): Promise<ImageFeatures> {
  const img = await loadImg(dataUrl);
  const aspectRatio = img.width / img.height;

  const tensor = tf.browser.fromPixels(img);
  const logits = (mobilenetModel as any).infer(tensor, 'conv_preds') as tf.Tensor;
  const embData = await logits.data();
  const embedding = new Float32Array(embData);
  tensor.dispose();
  logits.dispose();

  const { phash, edgeDensity } = computeClassicFeatures(img);
  return { embedding, phash, edgeDensity, aspectRatio };
}

async function extractFeaturesWithRotations(dataUrl: string): Promise<ImageFeaturesWithRot> {
  const img = await loadImg(dataUrl);
  const aspectRatio = img.width / img.height;

  const tensor = tf.browser.fromPixels(img);
  const logits = (mobilenetModel as any).infer(tensor, 'conv_preds') as tf.Tensor;
  const embData = await logits.data();
  const embedding = new Float32Array(embData);
  tensor.dispose();
  logits.dispose();

  const rotations = [0, 90, 180, 270];
  const rotatedEmbeddings: Float32Array[] = [];

  for (const angle of rotations) {
    const rotCanvas = drawRotated(img, angle);
    const rt = tf.browser.fromPixels(rotCanvas);
    const rl = (mobilenetModel as any).infer(rt, 'conv_preds') as tf.Tensor;
    const rd = await rl.data();
    rotatedEmbeddings.push(new Float32Array(rd));
    rt.dispose();
    rl.dispose();
  }

  const mirrorCanvas = drawMirrored(img);
  const mt = tf.browser.fromPixels(mirrorCanvas);
  const ml = (mobilenetModel as any).infer(mt, 'conv_preds') as tf.Tensor;
  const md = await ml.data();
  rotatedEmbeddings.push(new Float32Array(md));
  mt.dispose();
  ml.dispose();

  const { phash, edgeDensity } = computeClassicFeatures(img);
  return { embedding, phash, edgeDensity, aspectRatio, rotatedEmbeddings };
}

function drawRotated(img: HTMLImageElement, angle: number): HTMLCanvasElement {
  const SIZE = 224;
  const c = document.createElement('canvas');
  c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.save();
  ctx.translate(SIZE / 2, SIZE / 2);
  ctx.rotate((angle * Math.PI) / 180);
  const ratio = Math.min(SIZE / img.width, SIZE / img.height) * 0.95;
  const w = img.width * ratio;
  const h = img.height * ratio;
  ctx.drawImage(img, -w / 2, -h / 2, w, h);
  ctx.restore();
  return c;
}

function drawMirrored(img: HTMLImageElement): HTMLCanvasElement {
  const SIZE = 224;
  const c = document.createElement('canvas');
  c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.translate(SIZE, 0);
  ctx.scale(-1, 1);
  const ratio = Math.min(SIZE / img.width, SIZE / img.height) * 0.95;
  const w = img.width * ratio;
  const h = img.height * ratio;
  ctx.drawImage(img, (SIZE - w) / 2, (SIZE - h) / 2, w, h);
  return c;
}

function computeClassicFeatures(img: HTMLImageElement): {
  phash: Float32Array;
  edgeDensity: number;
} {
  const SIZE = 64;
  const canvas = document.createElement('canvas');
  canvas.width = SIZE; canvas.height = SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(img, 0, 0, SIZE, SIZE);
  const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
  const pixels = imageData.data;

  const gray = new Float32Array(SIZE * SIZE);
  for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
    gray[j] = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
  }

  return {
    phash: computePerceptualHash(gray, SIZE),
    edgeDensity: computeEdgeDensity(gray, SIZE),
  };
}

function computePerceptualHash(gray: Float32Array, size: number): Float32Array {
  const N = 8;
  const block = size / N;
  const small = new Float32Array(N * N);
  for (let by = 0; by < N; by++) {
    for (let bx = 0; bx < N; bx++) {
      let sum = 0; let count = 0;
      for (let y = by * block; y < (by + 1) * block; y++) {
        for (let x = bx * block; x < (bx + 1) * block; x++) {
          sum += gray[y * size + x];
          count++;
        }
      }
      small[by * N + bx] = sum / count;
    }
  }
  let mean = 0;
  for (let i = 0; i < small.length; i++) mean += small[i];
  mean /= small.length;

  const hash = new Float32Array(N * N);
  for (let i = 0; i < small.length; i++) {
    hash[i] = small[i] > mean ? 1 : 0;
  }
  return hash;
}

function computeEdgeDensity(gray: Float32Array, size: number): number {
  let edgeCount = 0;
  const threshold = 50;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const tl = gray[(y - 1) * size + (x - 1)];
      const t = gray[(y - 1) * size + x];
      const tr = gray[(y - 1) * size + (x + 1)];
      const l = gray[y * size + (x - 1)];
      const r = gray[y * size + (x + 1)];
      const bl = gray[(y + 1) * size + (x - 1)];
      const b = gray[(y + 1) * size + x];
      const br = gray[(y + 1) * size + (x + 1)];
      const gx = -tl - 2 * l - bl + tr + 2 * r + br;
      const gy = -tl - 2 * t - tr + bl + 2 * b + br;
      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag > threshold) edgeCount++;
    }
  }
  return edgeCount / ((size - 2) * (size - 2));
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

function averageArrays(arrs: Float32Array[]): Float32Array {
  if (arrs.length === 0) return new Float32Array();
  const len = arrs[0].length;
  const out = new Float32Array(len);
  for (const a of arrs) for (let i = 0; i < len; i++) out[i] += a[i];
  for (let i = 0; i < len; i++) out[i] /= arrs.length;
  return out;
}

function runningMean(current: Float32Array, next: Float32Array, n: number): Float32Array {
  const out = new Float32Array(current.length);
  for (let i = 0; i < current.length; i++) {
    out[i] = (current[i] * n + next[i]) / (n + 1);
  }
  return out;
}

function l2Distance(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function hammingLike(a: Float32Array, b: Float32Array): number {
  let diff = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    diff += Math.abs(a[i] - b[i]);
  }
  return diff / n;
}

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}