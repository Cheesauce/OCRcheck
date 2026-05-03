import React, { useEffect, useRef, useState } from 'react';
import './RecognitionStudio.css';
import Card from '../../components/Card/Card';
import Button from '../../components/Button/Button';
import ProgressBar from '../../components/ProgressBar/ProgressBar';
import Toast from '../../components/Toast/Toast';
import { useToasts } from '../../hooks/useToasts';
import { useCancellable, isCancelledError } from '../../hooks/useCancellable';
import { compressImage, formatBytes } from '../../services/imageCompression';
import { addSample, getAllSamples, getAllOcrDocs, type TrainingSample, type SampleCategory, type OcrDocument } from '../../services/database';
import {
  ensureEngine,
  addSampleToIndex,
  addNegativeExample,
  getIndexStats,
  predictFromDataUrl,
  resetIndex,
  indexAllSamples,
  getWorkerPoolStatus,
  QUALITY_PRESETS,
  type PredictionResult,
  type RecognitionQuality,
  type RecognitionQualityConfig,
} from '../../services/recognitionService';
import { ocrDataUrl } from '../../services/ocrRegion';
import { isPdfFile, renderPdfToImages } from '../../services/pdfToImages';
import {
  recognitionWorkspaceState,
  type StagedImageState,
  type PredictImageState,
} from '../../state/workspaceState';
import CompositeGallery from './CompositeGallery';
import RegionOverlay from './RegionOverlay';
import AnnotationPanel from './AnnotationPanel';
import { exportRecognitionReport, downloadAnnotatedImage } from '../../services/recognitionReport';
import { updateAnnotation, getAnnotationStats, type RegionAnnotation } from '../../services/annotationsService';

type Category = SampleCategory;

const CATEGORIES: { key: Category; label: string; icon: string; color: string; description: string }[] = [
  { key: 'word', label: 'Word / Text', icon: '🔤', color: '#06b6d4', description: 'Text phrases — highest priority match' },
  { key: 'logo', label: 'Logo', icon: '🏷️', color: '#8b5cf6', description: 'Brand marks and wordmarks' },
  { key: 'signature', label: 'Signature', icon: '✍️', color: '#ec4899', description: 'Handwritten signatures' },
  { key: 'stamp', label: 'Stamp', icon: '🔖', color: '#f59e0b', description: 'Rubber stamps and seals' },
];

const QUALITY_ORDER: RecognitionQuality[] = ['fast', 'balanced', 'precise', 'exhaustive'];

function formatLocalTimestamp(ts: number): string {
  const d = new Date(ts);
  try {
    const datePart = d.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
    });
    const timePart = d.toLocaleTimeString(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
    });
    return `${datePart} · ${timePart}`;
  } catch {
    return d.toString();
  }
}

function describePresetDetails(preset: RecognitionQualityConfig): string {
  const p: any = preset;

  if (Array.isArray(p.refineScales)) {
    const refineCount = p.refineScales.length;
    const coarseStride = typeof p.coarseStride === 'number'
      ? `${(p.coarseStride * 100).toFixed(0)}%`
      : '—';
    const budget = typeof p.maxWindowBudget === 'number'
      ? p.maxWindowBudget
      : null;
    const parts = [
      `1 coarse + ${refineCount} refinement pass${refineCount === 1 ? '' : 'es'}`,
      `stride ${coarseStride}`,
    ];
    if (budget !== null) parts.push(`≤ ${budget} windows`);
    return parts.join(' · ');
  }

  if (Array.isArray(p.windowScales)) {
    const n = p.windowScales.length;
    const stride = typeof p.stride === 'number' ? `${(p.stride * 100).toFixed(0)}%` : '—';
    return `${n} window scale${n === 1 ? '' : 's'} · stride ${stride}`;
  }

  return preset.description || '';
}

const RecognitionStudio: React.FC = () => {
  const { toasts, push, remove } = useToasts();

  const [mode, setMode] = useState<'train' | 'predict' | 'gallery'>(
    (recognitionWorkspaceState.mode as any) || 'train'
  );
  const [defaultCategories, setDefaultCategories] = useState<Category[]>(
    recognitionWorkspaceState.defaultCategories
  );
  const [label, setLabel] = useState(recognitionWorkspaceState.label);
  const [ocrTrainingEnabled, setOcrTrainingEnabled] = useState<boolean>(
    recognitionWorkspaceState.ocrTrainingEnabled ?? true
  );
  const [ocrLang, setOcrLang] = useState<string>(
    recognitionWorkspaceState.ocrLang || 'eng'
  );
  const [recognitionQuality, setRecognitionQuality] = useState<RecognitionQuality>(
    recognitionWorkspaceState.recognitionQuality || 'balanced'
  );
  const [staged, setStaged] = useState<StagedImageState[]>(recognitionWorkspaceState.staged);
  const [predictItems, setPredictItems] = useState<PredictImageState[]>(recognitionWorkspaceState.predictItems);
  const [existingLabels, setExistingLabels] = useState<string[]>([]);

  const trainInput = useRef<HTMLInputElement>(null);
  const [staging, setStaging] = useState(false);
  const [stagingProgress, setStagingProgress] = useState<{ done: number; total: number } | null>(null);
  const [saveProgress, setSaveProgress] = useState<{ stage: string; done: number; total: number } | null>(null);
  const [saving, setSaving] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [indexProg, setIndexProg] = useState<{ done: number; total: number } | null>(null);
  const [stats, setStats] = useState({ classes: 0, examples: 0, totalSamples: 0, negatives: 0, wordClasses: 0 });
  const [annStats, setAnnStats] = useState({ total: 0, confirmed: 0, rejected: 0, relabeled: 0, notes: 0, promoted: 0 });
  const [engineReady, setEngineReady] = useState(false);
  const [workerStatus, setWorkerStatus] = useState({ available: false, size: 0, ready: false });

  const predictInput = useRef<HTMLInputElement>(null);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingPredictProg, setLoadingPredictProg] = useState<{ done: number; total: number } | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictProg, setPredictProg] = useState<{ stage: string; currentItem: number; totalItems: number; inner: number } | null>(null);
  const [scanStartTime, setScanStartTime] = useState<number | null>(null);
  const [scanElapsed, setScanElapsed] = useState(0);

  const [reviewItem, setReviewItem] = useState<PredictImageState | null>(null);
  const [reportExporting, setReportExporting] = useState<{ stage: string; progress: number } | null>(null);

  const [annotating, setAnnotating] = useState<{ item: PredictImageState; regionIndex: number } | null>(null);

  const [showDocPicker, setShowDocPicker] = useState(false);
  const [dbDocs, setDbDocs] = useState<OcrDocument[]>([]);

  const { start: startIndex, cancel: cancelIndex, clear: clearIndex } = useCancellable();
  const { start: startPredict, cancel: cancelPredict, clear: clearPredict } = useCancellable();
  const { start: startSave, cancel: cancelSave, clear: clearSave } = useCancellable();

  useEffect(() => { recognitionWorkspaceState.mode = mode as any; }, [mode]);
  useEffect(() => { recognitionWorkspaceState.defaultCategories = defaultCategories; }, [defaultCategories]);
  useEffect(() => { recognitionWorkspaceState.label = label; }, [label]);
  useEffect(() => { recognitionWorkspaceState.staged = staged; }, [staged]);
  useEffect(() => { recognitionWorkspaceState.predictItems = predictItems; }, [predictItems]);
  useEffect(() => { recognitionWorkspaceState.ocrTrainingEnabled = ocrTrainingEnabled; }, [ocrTrainingEnabled]);
  useEffect(() => { recognitionWorkspaceState.ocrLang = ocrLang; }, [ocrLang]);
  useEffect(() => { recognitionWorkspaceState.recognitionQuality = recognitionQuality; }, [recognitionQuality]);

  useEffect(() => {
    (async () => {
      try {
        await ensureEngine();
        setEngineReady(true);
        await reindex();
      } catch (e: any) {
        push('error', `Engine init failed: ${e?.message || e}`);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const tick = () => setWorkerStatus(getWorkerPoolStatus());
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!predicting || scanStartTime === null) {
      setScanElapsed(0);
      return;
    }
    const interval = setInterval(() => {
      setScanElapsed(Date.now() - scanStartTime);
    }, 100);
    return () => clearInterval(interval);
  }, [predicting, scanStartTime]);

  const refreshStats = async () => {
    try {
      const all = await getAllSamples();
      const idx = getIndexStats();
      setStats({
        classes: idx.classes,
        examples: idx.examples,
        totalSamples: all.length,
        negatives: idx.negatives,
        wordClasses: idx.wordClasses,
      });
      const labels = Array.from(new Set(all.map((s) => s.label))).sort();
      setExistingLabels(labels);
      setAnnStats(await getAnnotationStats());
    } catch (e: any) {
      console.error('Stats refresh failed:', e);
    }
  };

  const reindex = async () => {
    setIndexing(true);
    const token = startIndex();
    try {
      await ensureEngine();
      resetIndex();
      const total = await indexAllSamples((p) => {
        if (token.cancelled) return;
        setIndexProg(p);
      });
      await refreshStats();
      if (token.cancelled) {
        push('info', 'Indexing stopped.');
      } else if (total > 0) {
        push('info', `Index refreshed — ${total} samples · GPU-accelerated.`);
      }
    } catch (e: any) {
      if (isCancelledError(e)) {
        push('info', 'Indexing cancelled.');
      } else {
        push('error', `Indexing failed: ${e?.message || e}`);
      }
    } finally {
      setIndexing(false);
      setIndexProg(null);
      clearIndex();
    }
  };

  const toggleDefaultCategory = (c: Category) => {
    setDefaultCategories((prev) => {
      const next = prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c];
      return next.length === 0 ? prev : next;
    });
  };

  const processFilesForTraining = async (files: File[]) => {
    setStaging(true);
    setStagingProgress({ done: 0, total: files.length });
    try {
      const next: StagedImageState[] = [];
      let processed = 0;
      for (const f of files) {
        if (isPdfFile(f)) {
          try {
            const pages = await renderPdfToImages(f, { scale: 1.5 });
            for (const p of pages) {
              const c = await compressImage(p.dataUrl);
              next.push({
                id: `stg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
                sourceName: `${f.name} · p.${p.pageNumber}`,
                pageNumber: p.pageNumber,
                previewUrl: c.dataUrl,
                dataUrl: c.dataUrl,
                originalSize: c.originalSize,
                compressedSize: c.compressedSize,
                categories: [...defaultCategories],
              });
            }
          } catch (e: any) {
            push('error', `Failed to read PDF "${f.name}": ${e?.message || e}`);
          }
        } else if (f.type.startsWith('image/')) {
          try {
            const c = await compressImage(f);
            next.push({
              id: `stg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
              sourceName: f.name,
              previewUrl: c.dataUrl,
              dataUrl: c.dataUrl,
              originalSize: c.originalSize,
              compressedSize: c.compressedSize,
              categories: [...defaultCategories],
            });
          } catch (e: any) {
            push('error', `Failed to read image "${f.name}": ${e?.message || e}`);
          }
        } else {
          push('error', `Unsupported file type: ${f.name}`);
        }
        processed++;
        setStagingProgress({ done: processed, total: files.length });
      }
      setStaged((prev) => [...prev, ...next]);
    } finally {
      setStaging(false);
      setStagingProgress(null);
    }
  };

  const handleTrainFiles = (files: FileList | null) => {
    if (!files) return;
    processFilesForTraining(Array.from(files));
  };

  const removeStaged = (id: string) => {
    setStaged((prev) => prev.filter((s) => s.id !== id));
  };

  const toggleStagedCategory = (id: string, c: Category) => {
    setStaged((prev) =>
      prev.map((s) => {
        if (s.id !== id) return s;
        const has = s.categories.includes(c);
        const nextCats = has ? s.categories.filter((x) => x !== c) : [...s.categories, c];
        if (nextCats.length === 0) return s;
        return { ...s, categories: nextCats };
      })
    );
  };

  const applyDefaultsToAll = () => {
    setStaged((prev) =>
      prev.map((s) => ({ ...s, categories: [...defaultCategories] }))
    );
    push('info', 'Categories applied to all staged images.');
  };

  const hasWordCategory = defaultCategories.includes('word') || staged.some((s) => s.categories.includes('word'));
  const effectiveOcr = hasWordCategory ? true : ocrTrainingEnabled;

  const saveTrainingSet = async () => {
    if (!label.trim()) {
      push('error', 'Please enter a label (e.g., "PAID", "Crizil").');
      return;
    }
    if (staged.length === 0) {
      push('error', 'Please add at least one image or PDF.');
      return;
    }

    const noCat = staged.find((s) => s.categories.length === 0);
    if (noCat) {
      push('error', `"${noCat.sourceName}" has no category selected.`);
      return;
    }

    setSaving(true);
    setSaveProgress({ stage: 'Starting…', done: 0, total: staged.length });
    const token = startSave();
    try {
      let totalOrig = 0;
      let totalComp = 0;
      let sampleCount = 0;
      let processed = 0;
      let ocrCount = 0;

      for (const s of staged) {
        if (token.cancelled) break;
        const needsOcr = effectiveOcr || s.categories.includes('word');
        setSaveProgress({
          stage: needsOcr ? `Processing "${s.sourceName}" (OCR + indexing)` : `Indexing "${s.sourceName}"`,
          done: processed,
          total: staged.length,
        });

        let ocrText = '';
        if (needsOcr) {
          try {
            ocrText = await ocrDataUrl(s.dataUrl, ocrLang, 15000);
            if (ocrText) ocrCount++;
          } catch {
            /* noop */
          }
        }

        for (const cat of s.categories) {
          if (token.cancelled) break;
          let effectiveOcrText = ocrText;
          if (cat === 'word' && !effectiveOcrText) {
            effectiveOcrText = label.trim();
          }
          totalOrig += s.originalSize;
          totalComp += s.compressedSize;
          const sample: TrainingSample = {
            id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}_${cat}`,
            name: s.sourceName,
            category: cat,
            label: label.trim(),
            imageData: s.dataUrl,
            originalSize: s.originalSize,
            compressedSize: s.compressedSize,
            createdAt: Date.now(),
            ocrText: effectiveOcrText || undefined,
            ocrLang: effectiveOcrText ? ocrLang : undefined,
            origin: 'upload',
          };
          try {
            await addSample(sample);
            await addSampleToIndex(sample);
            sampleCount++;
          } catch (e: any) {
            console.error('Failed saving sample:', e);
            push('error', `Save failed for "${s.sourceName}" [${cat}]: ${e?.message || e}`);
          }
        }
        processed++;
        setSaveProgress({
          stage: `Saved ${processed}/${staged.length}`,
          done: processed,
          total: staged.length,
        });
      }

      if (token.cancelled) {
        push('info', `Saving stopped. ${processed}/${staged.length} images processed.`);
      } else {
        const ratio = totalOrig > 0 ? ((1 - totalComp / totalOrig) * 100).toFixed(1) : '0';
        const ocrNote = effectiveOcr ? ` · OCR text in ${ocrCount}/${staged.length}` : '';
        push(
          'success',
          `${sampleCount} sample(s) saved · Compressed ${formatBytes(totalOrig)} → ${formatBytes(totalComp)} (${ratio}%)${ocrNote}`
        );
        setStaged([]);
        setLabel('');
      }
      await refreshStats();
    } catch (e: any) {
      if (isCancelledError(e)) {
        push('info', 'Save cancelled.');
      } else {
        console.error('Training save failed:', e);
        push('error', `Save failed: ${e?.message || e}`);
      }
    } finally {
      setSaving(false);
      setSaveProgress(null);
      clearSave();
    }
  };

  const openDocPicker = async () => {
    const docs = await getAllOcrDocs();
    setDbDocs(docs.sort((a, b) => b.createdAt - a.createdAt));
    setShowDocPicker(true);
  };

  const handleSelectDbDoc = (doc: OcrDocument) => {
    if (!doc.pageRecords || doc.pageRecords.length === 0) {
      push('error', 'This document has no saved pages.');
      return;
    }

    const next: PredictImageState[] = doc.pageRecords
      .filter((p) => !!p.imageDataUrl)
      .map((p) => ({
        id: `pr_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        sourceName: `${doc.name} · p.${p.pageNumber}`,
        pageNumber: p.pageNumber,
        previewUrl: p.imageDataUrl,
        dataUrl: p.imageDataUrl,
        prediction: null,
        status: 'pending',
        sourceFileId: doc.id,
        embeddedText: p.words && p.words.length > 0 ? { text: p.text, words: p.words } : undefined,
      }));

    if (next.length === 0) {
      push('error', 'No images found in this document.');
      return;
    }

    setPredictItems((prev) => [...prev, ...next]);
    setShowDocPicker(false);
  };

  const processFilesForPrediction = async (files: File[]) => {
    setLoadingPredict(true);
    setLoadingPredictProg({ done: 0, total: files.length });
    try {
      const next: PredictImageState[] = [];
      let processed = 0;
      for (const f of files) {
        const sourceFileId = `f_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        if (isPdfFile(f)) {
          try {
            const pages = await renderPdfToImages(f, { scale: 1.5 });
            for (const p of pages) {
              const c = await compressImage(p.dataUrl);
              next.push({
                id: `pr_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
                sourceName: `${f.name} · p.${p.pageNumber}`,
                pageNumber: p.pageNumber,
                previewUrl: c.dataUrl,
                dataUrl: c.dataUrl,
                prediction: null,
                status: 'pending',
                sourceFileId,
                embeddedText: p.embeddedText,
              });
            }
          } catch (e: any) {
            push('error', `Failed to read PDF "${f.name}": ${e?.message || e}`);
          }
        } else if (f.type.startsWith('image/')) {
          try {
            const c = await compressImage(f);
            next.push({
              id: `pr_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
              sourceName: f.name,
              previewUrl: c.dataUrl,
              dataUrl: c.dataUrl,
              prediction: null,
              status: 'pending',
              sourceFileId,
            });
          } catch (e: any) {
            push('error', `Failed to read image "${f.name}": ${e?.message || e}`);
          }
        } else {
          push('error', `Unsupported file type: ${f.name}`);
        }
        processed++;
        setLoadingPredictProg({ done: processed, total: files.length });
      }
      setPredictItems((prev) => [...prev, ...next]);
    } finally {
      setLoadingPredict(false);
      setLoadingPredictProg(null);
    }
  };

  const handlePredictFiles = (files: FileList | null) => {
    if (!files) return;
    processFilesForPrediction(Array.from(files));
  };

  const removePredict = (id: string) => {
    setPredictItems((prev) => prev.filter((p) => p.id !== id));
  };

  const clearAllPredict = () => setPredictItems([]);

  const editPredictionLabel = (id: string, newLabel: string) => {
    setPredictItems((prev) =>
      prev.map((p) =>
        p.id === id && p.prediction
          ? { ...p, prediction: { ...p.prediction, label: newLabel } }
          : p
      )
    );
  };

  const updatePrediction = (id: string, newPrediction: PredictionResult) => {
    setPredictItems((prev) =>
      prev.map((p) =>
        p.id === id ? { ...p, prediction: newPrediction } : p
      )
    );
  };

  const runPredictionAll = async () => {
    if (predictItems.length === 0) return;
    if (stats.examples === 0) {
      push('error', 'No trained samples yet. Add some in the Training tab first.');
      return;
    }
    setPredicting(true);
    setScanStartTime(Date.now());
    setPredictProg({ stage: 'Initializing…', currentItem: 0, totalItems: predictItems.length, inner: 0 });
    const token = startPredict();
    let completed = 0;
    let foundRegions = 0;
    let foundWords = 0;
    try {
      const updated = [...predictItems];
      for (let i = 0; i < updated.length; i++) {
        if (token.cancelled) break;
        updated[i] = { ...updated[i], status: 'running' };
        setPredictItems([...updated]);
        setPredictProg({
          stage: `Scanning "${updated[i].sourceName}"`,
          currentItem: i + 1,
          totalItems: updated.length,
          inner: 0,
        });
        try {
          const r = await predictFromDataUrl(
            updated[i].dataUrl,
            5,
            (_stage, p) => {
              setPredictProg({
                stage: `Scanning "${updated[i].sourceName}"`,
                currentItem: i + 1,
                totalItems: updated.length,
                inner: p,
              });
            },
            recognitionQuality,
            updated[i].embeddedText
          );
          if (token.cancelled) break;
          const hasRegions = r && r.regions && r.regions.length > 0;
          updated[i] = {
            ...updated[i],
            prediction: hasRegions ? r : null,
            status: hasRegions ? 'done' : 'no_match',
            error: hasRegions ? undefined : 'No trained element located in this image.',
            scannedAt: Date.now(),
            qualityUsed: recognitionQuality,
          };
          if (hasRegions) {
            foundRegions += r!.regions!.length;
            foundWords += r!.regions!.filter((reg: any) => reg.category === 'word').length;
          }
          completed++;
        } catch (e: any) {
          updated[i] = {
            ...updated[i],
            status: 'error',
            error: e?.message || String(e),
            scannedAt: Date.now(),
          };
        }
        setPredictItems([...updated]);
      }
      if (token.cancelled) {
        setPredictItems((prev) =>
          prev.map((p) => (p.status === 'running' ? { ...p, status: 'pending' } : p))
        );
        push('info', `Scan stopped. ${completed} of ${predictItems.length} completed.`);
      } else {
        const elapsedSec = ((Date.now() - (scanStartTime || Date.now())) / 1000).toFixed(1);
        if (foundRegions > 0) {
          const wordNote = foundWords > 0 ? ` (incl. ${foundWords} word match${foundWords === 1 ? '' : 'es'})` : '';
          push('success', `Scan complete in ${elapsedSec}s — ${foundRegions} element${foundRegions === 1 ? '' : 's'}${wordNote} located.`);
        } else {
          push('info', `Scanned ${updated.length} image(s) in ${elapsedSec}s — no trained elements located.`);
        }
      }
    } finally {
      setPredicting(false);
      setScanStartTime(null);
      setPredictProg(null);
      clearPredict();
    }
  };

  const stopPrediction = () => {
    cancelPredict();
    push('info', 'Stopping scan…');
  };

  const stopIndexing = () => {
    cancelIndex();
    push('info', 'Stopping indexing…');
  };

  const stopSave = () => {
    cancelSave();
    push('info', 'Stopping save…');
  };

  const exportReport = async () => {
    const done = predictItems.filter((p) => p.status === 'done' && p.prediction);
    if (done.length === 0) {
      push('error', 'No successful matches to export.');
      return;
    }
    setReportExporting({ stage: 'Starting…', progress: 0 });
    try {
      await exportRecognitionReport(done, 'recognition_report.pdf', (stage, progress) => {
        setReportExporting({ stage, progress });
      });
      push('success', 'Annotated PDF report saved.');
    } catch (e: any) {
      push('error', `Export failed: ${e?.message || e}`);
    } finally {
      setReportExporting(null);
    }
  };

  const downloadSingleAnnotated = async (item: PredictImageState) => {
    try {
      await downloadAnnotatedImage(item);
      push('success', 'Annotated image saved.');
    } catch (e: any) {
      push('error', `Save failed: ${e?.message || e}`);
    }
  };

  const promoteAnnotationToTraining = async (ann: RegionAnnotation) => {
    if (!ann.regionCropDataUrl) {
      push('error', 'No region crop available to promote.');
      return;
    }
    const category: SampleCategory = ann.correctedCategory || ann.originalCategory;
    const label = (ann.correctedLabel || ann.originalLabel).trim();
    if (!label) {
      push('error', 'Missing label — cannot promote.');
      return;
    }

    try {
      const compressed = await compressImage(ann.regionCropDataUrl);

      let ocrText = '';
      if (ocrTrainingEnabled || category === 'word') {
        try {
          ocrText = await ocrDataUrl(compressed.dataUrl, ocrLang, 10000);
        } catch { /* noop */ }
      }
      if (category === 'word' && !ocrText) ocrText = label;

      const sample: TrainingSample = {
        id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}_${category}`,
        name: `${ann.sourceImageName} · region`,
        category,
        label,
        imageData: compressed.dataUrl,
        originalSize: compressed.originalSize,
        compressedSize: compressed.compressedSize,
        createdAt: Date.now(),
        ocrText: ocrText || undefined,
        ocrLang: ocrText ? ocrLang : undefined,
        sourceAnnotationId: ann.id,
        origin: 'annotation',
      };

      await addSample(sample);
      await addSampleToIndex(sample);
      await updateAnnotation(ann.id, { promotedToTrainingSampleId: sample.id });

      await refreshStats();
      push('success', `Feedback stored — "${label}" strengthened with a new training sample.`);
    } catch (e: any) {
      push('error', `Promotion failed: ${e?.message || e}`);
    }
  };

  const registerAnnotationAsNegative = async (ann: RegionAnnotation) => {
    if (!ann.regionCropDataUrl) return;
    try {
      await addNegativeExample(
        ann.regionCropDataUrl,
        ann.originalLabel,
        ann.originalCategory
      );
      await refreshStats();
      push('info', `Negative example recorded — similar regions won't be labeled "${ann.originalLabel}" again.`);
    } catch (e: any) {
      push('error', `Couldn't register negative: ${e?.message || e}`);
    }
  };

  const matchedCount = predictItems.filter((p) => p.status === 'done').length;
  const noMatchCount = predictItems.filter((p) => p.status === 'no_match').length;

  const predictOverallPct = predictProg
    ? (((predictProg.currentItem - 1) + predictProg.inner) / predictProg.totalItems) * 100
    : 0;

  return (
    <div className="rec-studio">
      <Toast toasts={toasts} onRemove={remove} />

      <header className="page-head">
        <div>
          <h1>AI Recognition Studio</h1>
          <p>
            <strong>Multi-threaded recognition</strong> with <strong>word-priority matching</strong>. Parallel worker pool distributes sliding-window scans across CPU cores for <strong>3–4× faster</strong> localization.
          </p>
        </div>
        <div className="engine-stack">
          <div className="engine-badge">
            <span className={`dot ${engineReady ? 'on' : 'off'}`} />
            {engineReady ? 'Engine Ready' : 'Loading…'}
          </div>
          {engineReady && (
            <div className={`worker-badge ${workerStatus.ready ? 'on' : workerStatus.available ? 'loading' : 'off'}`}>
              <span className="worker-icon">⚡</span>
              {workerStatus.ready
                ? `${workerStatus.size} workers`
                : workerStatus.available
                  ? 'Workers starting…'
                  : 'Single-thread'}
            </div>
          )}
        </div>
      </header>

      <div className="mode-tabs">
        <button className={`mode-tab ${mode === 'train' ? 'active' : ''}`} onClick={() => setMode('train')}>
          🎯 Train
        </button>
        <button className={`mode-tab ${mode === 'predict' ? 'active' : ''}`} onClick={() => setMode('predict')}>
          🔎 Locate & Annotate
        </button>
        <button className={`mode-tab ${mode === 'gallery' ? 'active' : ''}`} onClick={() => setMode('gallery')}>
          🎨 Composite Gallery
        </button>
      </div>

      <div className="rec-stats">
        <StatPill label="Trained Classes" value={stats.classes} />
        {stats.wordClasses > 0 && (
          <StatPill label="Word Classes" value={stats.wordClasses} tint priority />
        )}
        <StatPill label="Indexed Examples" value={stats.examples} />
        <StatPill label="Total Samples" value={stats.totalSamples} />
        <StatPill label="Negatives" value={stats.negatives} tint />
        <StatPill label="Annotations" value={annStats.total} tint accent />
        {annStats.promoted > 0 && (
          <StatPill label="Feedback → Training" value={annStats.promoted} tint accent />
        )}
        {!indexing ? (
          <Button variant="ghost" icon="↻" onClick={reindex}>Re-index</Button>
        ) : (
          <Button variant="danger" icon="⏹" onClick={stopIndexing}>Stop Indexing</Button>
        )}
      </div>

      {indexProg && (
        <div style={{ marginBottom: 16 }}>
          <ProgressBar
            label={`Building prototypes… ${indexProg.done}/${indexProg.total}`}
            value={(indexProg.done / Math.max(1, indexProg.total)) * 100}
          />
        </div>
      )}

      {mode === 'train' && (
        <div className="rec-grid">
          <Card title="Categories" subtitle="Words are prioritized — OCR text always wins over visual match">
            <div className="cat-multi">
              {CATEGORIES.map((c) => {
                const active = defaultCategories.includes(c.key);
                const isPriority = c.key === 'word';
                return (
                  <button
                    key={c.key}
                    className={`cat-multi-pill ${active ? 'active' : ''} ${isPriority ? 'priority' : ''}`}
                    onClick={() => toggleDefaultCategory(c.key)}
                    style={{ ['--pill' as any]: c.color }}
                    type="button"
                    title={c.description}
                  >
                    <span className={`check-box ${active ? 'checked' : ''}`}>
                      {active ? '✓' : ''}
                    </span>
                    <span className="pill-icon">{c.icon}</span>
                    <span className="pill-label">{c.label}</span>
                    {isPriority && <span className="pill-badge">Priority</span>}
                  </button>
                );
              })}
            </div>

            {hasWordCategory && (
              <div className="word-priority-notice">
                <span className="wp-icon">🔤</span>
                <div>
                  <div className="wp-title">Word-priority mode active</div>
                  <div className="wp-desc">OCR will be run on every sample to extract text. Word matches will be prioritized over logo/stamp visual matches during recognition.</div>
                </div>
              </div>
            )}

            <label className="field-label" style={{ marginTop: 18 }}>
              Label / Name
              {hasWordCategory && <span className="field-hint"> — for word samples, this is the canonical text (e.g. "PAID", "CONFIDENTIAL")</span>}
            </label>
            <input
              className="input"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder={hasWordCategory ? 'e.g. PAID, INVOICE, CONFIDENTIAL' : 'e.g. Crizil Stamp'}
              list="existing-labels"
            />
            <datalist id="existing-labels">
              {existingLabels.map((l) => <option key={l} value={l} />)}
            </datalist>
            {existingLabels.length > 0 && (
              <div className="label-suggestions">
                <div className="label-suggestions-title">Existing labels:</div>
                <div className="label-chips">
                  {existingLabels.slice(0, 10).map((l) => (
                    <button
                      key={l}
                      type="button"
                      className={`label-chip ${label === l ? 'active' : ''}`}
                      onClick={() => setLabel(l)}
                    >
                      {l}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="ocr-toggle-box">
              <label className="ocr-toggle">
                <input
                  type="checkbox"
                  checked={effectiveOcr}
                  disabled={hasWordCategory}
                  onChange={(e) => setOcrTrainingEnabled(e.target.checked)}
                />
                <span className="ocr-toggle-slider"></span>
                <span className="ocr-toggle-label">
                  <span className="ocr-toggle-title">
                    🔤 OCR training samples
                    {hasWordCategory && <span className="wp-required">required for words</span>}
                  </span>
                  <span className="ocr-toggle-desc">
                    Extract text from each sample to enrich the index. Adds ~1–3s per image.
                  </span>
                </span>
              </label>
              {effectiveOcr && (
                <div className="ocr-lang-row">
                  <label className="field-label" style={{ margin: 0 }}>OCR Language</label>
                  <select
                    className="select"
                    style={{ maxWidth: 180 }}
                    value={ocrLang}
                    onChange={(e) => setOcrLang(e.target.value)}
                  >
                    <option value="eng">English</option>
                    <option value="spa">Spanish</option>
                    <option value="fra">French</option>
                    <option value="deu">German</option>
                    <option value="ita">Italian</option>
                    <option value="por">Portuguese</option>
                    <option value="chi_sim">Chinese (Simplified)</option>
                    <option value="jpn">Japanese</option>
                    <option value="kor">Korean</option>
                    <option value="ara">Arabic</option>
                    <option value="rus">Russian</option>
                    <option value="hin">Hindi</option>
                  </select>
                </div>
              )}
            </div>

            {staged.length > 0 && (
              <Button
                variant="ghost"
                icon="↻"
                onClick={applyDefaultsToAll}
                className="full-width"
                style={{ marginTop: 10 }}
              >
                Apply these categories to all {staged.length} staged
              </Button>
            )}
          </Card>

          <Card title="Training Files" subtitle={`${staged.length} staged · Click tags to change categories per image`}>
            <div
              className="dropzone"
              onClick={() => trainInput.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                handleTrainFiles(e.dataTransfer.files);
              }}
            >
              <input
                ref={trainInput}
                type="file"
                accept="image/*,application/pdf,.pdf"
                multiple
                hidden
                onChange={(e) => handleTrainFiles(e.target.files)}
              />
              <div className="drop-icon">🖼️📄</div>
              <div className="drop-title">Drop images or PDFs, or click to browse</div>
              <div className="drop-sub">PNG · JPG · PDF · auto-compressed · PDFs split per page</div>
            </div>

            {staging && stagingProgress && (
              <div style={{ marginTop: 12 }}>
                <ProgressBar
                  label={`Processing files… ${stagingProgress.done}/${stagingProgress.total}`}
                  value={(stagingProgress.done / Math.max(1, stagingProgress.total)) * 100}
                />
              </div>
            )}

            {staged.length > 0 && (
              <>
                <div className="thumbs-multi">
                  {staged.map((s) => (
                    <div key={s.id} className="thumb-multi" title={s.sourceName}>
                      <div className="tm-img-wrap">
                        <img src={s.previewUrl} alt={s.sourceName} />
                        {s.pageNumber && <span className="thumb-badge">p.{s.pageNumber}</span>}
                        <button className="thumb-remove" onClick={() => removeStaged(s.id)}>×</button>
                      </div>
                      <div className="tm-name">{s.sourceName}</div>
                      <div className="tm-cats">
                        {CATEGORIES.map((c) => {
                          const active = s.categories.includes(c.key);
                          return (
                            <button
                              key={c.key}
                              className={`tm-cat ${active ? 'active' : ''} ${c.key === 'word' ? 'priority' : ''}`}
                              style={{ ['--pill' as any]: c.color }}
                              onClick={() => toggleStagedCategory(s.id, c.key)}
                              type="button"
                              title={active ? `Remove ${c.label}` : `Add ${c.label}`}
                            >
                              <span className="tm-cat-icon">{c.icon}</span>
                              <span>{c.label.split(' ')[0]}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
                <Button
                  variant="ghost"
                  icon="🗑️"
                  onClick={() => setStaged([])}
                  disabled={saving}
                  className="full-width"
                  style={{ marginTop: 10 }}
                >
                  Clear All Staged
                </Button>
              </>
            )}

            <div style={{ marginTop: 14 }}>
              {!saving ? (
                <Button
                  variant="primary"
                  icon="💾"
                  onClick={saveTrainingSet}
                  disabled={staged.length === 0 || !label.trim() || staging}
                  className="full-width"
                >
                  Save & Train ({staged.reduce((acc, s) => acc + s.categories.length, 0)} sample{staged.reduce((acc, s) => acc + s.categories.length, 0) === 1 ? '' : 's'})
                </Button>
              ) : (
                <Button variant="danger" icon="⏹" onClick={stopSave} className="full-width">
                  Stop Saving
                </Button>
              )}
            </div>

            {saveProgress && (
              <div style={{ marginTop: 12 }}>
                <ProgressBar
                  label={saveProgress.stage}
                  value={(saveProgress.done / Math.max(1, saveProgress.total)) * 100}
                />
              </div>
            )}
          </Card>
        </div>
      )}

      {mode === 'predict' && (
        <div className="predict-layout">
          <Card
            title="Upload Images or PDFs to Scan"
            subtitle="Word matches are prioritized — click any region to annotate."
          >
            <div className="field-label">Precision Mode</div>
            <div className="quality-group">
              {QUALITY_ORDER.map((q) => {
                const preset = QUALITY_PRESETS[q];
                const active = recognitionQuality === q;
                return (
                  <button
                    key={q}
                    type="button"
                    className={`quality-btn ${active ? 'active' : ''}`}
                    onClick={() => !predicting && setRecognitionQuality(q)}
                    disabled={predicting}
                  >
                    <div className="qb-head">
                      <span className="qb-title">{preset.title}</span>
                      <span className="qb-speed">{preset.speed}</span>
                    </div>
                    <div className="qb-desc">{preset.description}</div>
                    <div className="qb-variants">
                      <span className="qb-variant-dot" />
                      {describePresetDetails(preset)}
                    </div>
                  </button>
                );
              })}
            </div>

            {recognitionQuality === 'exhaustive' && (
              <div className="precision-hint">
                <span className="ph-icon">🔬</span>
                <div>
                  <div className="ph-title">Exhaustive mode</div>
                  <div className="ph-desc">
                    Densest coarse-to-fine grid with multiple refinement passes. Parallel workers scale this across {workerStatus.size || 'all'} CPU cores. Use for dense documents where even small or partially-occluded elements matter.
                  </div>
                </div>
              </div>
            )}

            <div
              className="dropzone"
              style={{ marginTop: 16 }}
              onClick={() => predictInput.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                handlePredictFiles(e.dataTransfer.files);
              }}
            >
              <input
                ref={predictInput}
                type="file"
                accept="image/*,application/pdf,.pdf"
                multiple
                hidden
                onChange={(e) => handlePredictFiles(e.target.files)}
              />
              <div className="drop-icon">🔎</div>
              <div className="drop-title">Drop images or PDFs to scan</div>
              <div className="drop-sub">
                {stats.wordClasses > 0
                  ? `${stats.wordClasses} word class${stats.wordClasses === 1 ? '' : 'es'} prioritized · all visual prototypes matched`
                  : 'Matches against all trained prototypes · GPU-accelerated'}
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 16 }}>
              <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
              <div style={{ fontSize: 11, color: 'var(--text-2)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em' }}>OR</div>
              <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
            </div>

            <Button
              variant="secondary"
              icon="🗄️"
              onClick={openDocPicker}
              className="full-width"
              style={{ marginTop: 16 }}
              disabled={predicting || loadingPredict}
            >
              Load from Database
            </Button>

            {loadingPredict && loadingPredictProg && (
              <div style={{ marginTop: 12 }}>
                <ProgressBar
                  label={`Preparing files… ${loadingPredictProg.done}/${loadingPredictProg.total}`}
                  value={(loadingPredictProg.done / Math.max(1, loadingPredictProg.total)) * 100}
                />
              </div>
            )}

            <div style={{ marginTop: 14, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {!predicting ? (
                <Button
                  variant="primary"
                  icon="⚡"
                  onClick={runPredictionAll}
                  disabled={predictItems.length === 0 || stats.examples === 0 || loadingPredict}
                >
                  Scan {predictItems.length > 0 ? `(${predictItems.length})` : ''}
                </Button>
              ) : (
                <Button variant="danger" icon="⏹" onClick={stopPrediction}>
                  Stop Scan
                </Button>
              )}
              {predictItems.length > 0 && !predicting && (
                <Button variant="ghost" icon="🗑️" onClick={clearAllPredict}>
                  Clear All
                </Button>
              )}
            </div>

            {predictProg && (
              <div style={{ marginTop: 14 }}>
                <ProgressBar
                  label={`${predictProg.stage} · ${predictProg.currentItem}/${predictProg.totalItems} · ${(scanElapsed / 1000).toFixed(1)}s elapsed`}
                  value={predictOverallPct}
                />
                <div className="scan-live-stats">
                  <span>⚡ {workerStatus.size || 1} parallel worker{(workerStatus.size || 1) === 1 ? '' : 's'}</span>
                  <span>·</span>
                  <span>📊 {QUALITY_PRESETS[recognitionQuality].title} mode</span>
                </div>
              </div>
            )}

            {matchedCount > 0 && (
              <div className="export-section">
                <div className="export-section-title">📤 Export Annotated Report</div>
                <Button
                  variant="primary"
                  icon="📄"
                  onClick={exportReport}
                  loading={!!reportExporting}
                  disabled={!!reportExporting}
                  className="full-width"
                >
                  {reportExporting ? 'Building report…' : `Export Annotated PDF (${matchedCount})`}
                </Button>
                {reportExporting && (
                  <div style={{ marginTop: 10 }}>
                    <ProgressBar label={reportExporting.stage} value={reportExporting.progress * 100} />
                  </div>
                )}
              </div>
            )}

            <div className="priority-legend">
              <div className="pl-title">📊 Match Priority</div>
              <div className="pl-items">
                <div className="pl-item pl-word">
                  <span className="pl-icon">🔤</span>
                  <span className="pl-name">Words</span>
                  <span className="pl-rank">1st</span>
                </div>
                <div className="pl-item pl-logo">
                  <span className="pl-icon">🏷️</span>
                  <span className="pl-name">Logos</span>
                  <span className="pl-rank">2nd</span>
                </div>
                <div className="pl-item pl-stamp">
                  <span className="pl-icon">🔖</span>
                  <span className="pl-name">Stamps</span>
                  <span className="pl-rank">3rd</span>
                </div>
                <div className="pl-item pl-sig">
                  <span className="pl-icon">✍️</span>
                  <span className="pl-name">Signatures</span>
                  <span className="pl-rank">4th</span>
                </div>
              </div>
              <p className="pl-hint">
                When a region contains readable text matching a trained word class, it's classified as a word match — regardless of visual similarity to other classes.
              </p>
            </div>

            <div className="feedback-hero">
              <div className="fh-title">🧠 Feedback Loop</div>
              <div className="fh-stats">
                <div className="fh-stat">
                  <div className="fh-num" style={{color:'#10b981'}}>{annStats.confirmed}</div>
                  <div className="fh-lbl">Confirmed</div>
                </div>
                <div className="fh-stat">
                  <div className="fh-num" style={{color:'#f59e0b'}}>{annStats.relabeled}</div>
                  <div className="fh-lbl">Relabeled</div>
                </div>
                <div className="fh-stat">
                  <div className="fh-num" style={{color:'#ef4444'}}>{annStats.rejected}</div>
                  <div className="fh-lbl">Rejected</div>
                </div>
                <div className="fh-stat">
                  <div className="fh-num" style={{color:'#6366f1'}}>{annStats.notes}</div>
                  <div className="fh-lbl">Notes</div>
                </div>
              </div>
              <p className="fh-hint">
                Click any region below to add a comment, confirm it, relabel it, or reject it. Confirmed and relabeled regions become training samples automatically.
              </p>
            </div>

            {stats.examples === 0 && (
              <p className="hint" style={{ marginTop: 12, color: 'var(--warn)' }}>
                ⚠️ No training samples yet. Switch to Train mode to add some.
              </p>
            )}
          </Card>

          <Card
            title="Scan Results"
            subtitle={
              predictItems.length === 0
                ? 'Drop files to begin'
                : `${predictItems.length} item(s) · ${matchedCount} matched${noMatchCount ? ` · ${noMatchCount} no match` : ''}`
            }
          >
            {predictItems.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">🧠</div>
                <div className="empty-title">No files queued</div>
                <div className="empty-sub">Upload images or PDFs to locate trained elements.</div>
              </div>
            )}

            {predictItems.length > 0 && (
              <div className="predict-grid">
                {predictItems.map((p) => (
                  <PredictCard
                    key={p.id}
                    item={p}
                    onRemove={() => removePredict(p.id)}
                    onEditLabel={(newLabel) => editPredictionLabel(p.id, newLabel)}
                    onOpenReview={() => setReviewItem(p)}
                    onDownload={() => downloadSingleAnnotated(p)}
                    onAnnotateRegion={(rIdx) => setAnnotating({ item: p, regionIndex: rIdx })}
                  />
                ))}
              </div>
            )}
          </Card>
        </div>
      )}

      {mode === 'gallery' && <CompositeGallery onRefreshStats={refreshStats} />}

      {reviewItem && reviewItem.prediction && (
        <ReviewModal
          item={reviewItem}
          onClose={() => setReviewItem(null)}
          onDownload={() => downloadSingleAnnotated(reviewItem)}
          onAnnotateRegion={(rIdx) => {
            setAnnotating({ item: reviewItem, regionIndex: rIdx });
          }}
          onUpdatePrediction={(newPred) => {
            updatePrediction(reviewItem.id, newPred);
            setReviewItem((prev) => prev ? { ...prev, prediction: newPred } : prev);
          }}
        />
      )}

      {annotating && (
        <AnnotationPanel
          item={annotating.item}
          regionIndex={annotating.regionIndex}
          onClose={() => setAnnotating(null)}
          onPromoteToTraining={promoteAnnotationToTraining}
          onRejected={registerAnnotationAsNegative}
        />
      )}

      {showDocPicker && (
        <div className="review-modal-backdrop" onClick={() => setShowDocPicker(false)}>
          <div className="review-modal" style={{ maxWidth: 600 }} onClick={e => e.stopPropagation()}>
            <div className="review-head">
              <div className="review-title">
                <span style={{ fontSize: 24, marginRight: 8 }}>🗄️</span>
                Select Document
              </div>
              <button className="review-close" onClick={() => setShowDocPicker(false)}>×</button>
            </div>
            <div className="review-body" style={{ display: 'block', padding: 20, maxHeight: '60vh', overflowY: 'auto' }}>
              {dbDocs.length === 0 ? (
                <div className="empty-state" style={{ padding: 40 }}>
                  <div className="empty-icon">📄</div>
                  <div className="empty-title">No OCR documents</div>
                  <div className="empty-sub">Process a scanned PDF in the OCR Workspace first to save it to the database.</div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {dbDocs.map(d => (
                    <div 
                      key={d.id} 
                      className="rs-region-row clickable" 
                      onClick={() => handleSelectDbDoc(d)}
                      style={{ padding: '14px 18px', borderLeftWidth: 1 }}
                    >
                      <div className="rs-region-info">
                        <div className="rs-region-label" style={{ fontSize: 14 }}>{d.name}</div>
                        <div className="rs-region-meta" style={{ marginTop: 6 }}>
                          {d.pages} page{d.pages !== 1 ? 's' : ''} · {new Date(d.createdAt).toLocaleDateString()}
                        </div>
                      </div>
                      <div style={{ color: 'var(--accent)', fontSize: 20 }}>→</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const PredictCard: React.FC<{
  item: PredictImageState;
  onRemove: () => void;
  onEditLabel: (v: string) => void;
  onOpenReview: () => void;
  onDownload: () => void;
  onAnnotateRegion: (regionIndex: number) => void;
}> = ({ item, onRemove, onEditLabel, onOpenReview, onDownload, onAnnotateRegion }) => {
  const pred = item.prediction as PredictionResult | null;
  const confPct = pred ? pred.confidence * 100 : null;
  const catMeta = pred ? CATEGORIES.find((c) => c.key === (pred.category as Category)) : null;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(pred?.label || '');

  useEffect(() => {
    setDraft(pred?.label || '');
  }, [pred?.label]);

  const commit = () => {
    const val = draft.trim();
    if (val) onEditLabel(val);
    setEditing(false);
  };

  const regions = pred?.regions || [];
  const regionCount = regions.length;
  const wordCount = regions.filter((r) => r.category === 'word').length;

  return (
    <div className={`predict-card ${item.status === 'no_match' ? 'no-match' : ''}`}>
      <div
        className={`pc-preview ${item.status === 'done' ? 'annotatable' : ''}`}
        title={item.status === 'done' ? 'Click a region to annotate, or the image to review' : ''}
      >
        {item.status === 'done' && pred ? (
          <RegionOverlay
            imageUrl={item.previewUrl}
            regions={regions}
            onRegionClick={(idx) => onAnnotateRegion(idx)}
            onBackgroundClick={onOpenReview}
          />
        ) : (
          <img src={item.previewUrl} alt={item.sourceName} />
        )}
        {item.pageNumber && <span className="pc-page-badge">Page {item.pageNumber}</span>}
        <button
          className="pc-remove"
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          title="Remove"
        >
          ×
        </button>
      </div>
      <div className="pc-body">
        <div className="pc-filename" title={item.sourceName}>{item.sourceName}</div>

        {item.status === 'pending' && <div className="pc-status pending">Waiting to scan…</div>}
        {item.status === 'running' && <div className="pc-status running"><span className="mini-spinner" /> Scanning page…</div>}
        {item.status === 'error' && <div className="pc-status error">⚠ {item.error || 'Unknown error'}</div>}
        {item.status === 'no_match' && (
          <div className="pc-status no-match">
            <span className="no-match-icon">○</span>
            <span>No trained element found</span>
          </div>
        )}

        {item.status === 'done' && pred && (
          <>
            <div className="pc-top-result">
              <div className="pc-top-icon">{catMeta?.icon || '🎯'}</div>
              <div className="pc-top-info">
                {editing ? (
                  <input
                    autoFocus
                    className="pc-label-edit"
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={commit}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') commit();
                      if (e.key === 'Escape') { setDraft(pred.label); setEditing(false); }
                    }}
                  />
                ) : (
                  <div
                    className="pc-top-label"
                    title="Click to edit"
                    onClick={() => setEditing(true)}
                  >
                    {pred.label} <span className="edit-hint">✏️</span>
                  </div>
                )}
                <div className="pc-top-meta">
                  <span className={`chip chip-${pred.category}`}>{pred.category}</span>
                  <span className="pc-conf">{confPct != null ? `${confPct.toFixed(0)}%` : '—'}</span>
                </div>
              </div>
            </div>

            <div className="pc-region-summary">
              <div className="pc-region-count-badge">
                📍 {regionCount} location{regionCount === 1 ? '' : 's'}
                {wordCount > 0 && <span className="pc-word-badge">🔤 {wordCount} word</span>}
                · click to annotate
              </div>
            </div>

            {pred.meta && (
              <div className="pc-perf-meta">
                <span className="pc-perf-chip">
                  ⏱ {(pred.meta.elapsedMs / 1000).toFixed(1)}s
                </span>
                <span className="pc-perf-chip">
                  🔍 {pred.meta.windowsScanned} windows
                </span>
                <span className="pc-perf-chip">
                  ⚡ {pred.meta.parallelWorkers} worker{pred.meta.parallelWorkers === 1 ? '' : 's'}
                </span>
                <span className={`pc-perf-chip q-${pred.meta.quality}`}>
                  {QUALITY_PRESETS[pred.meta.quality]?.title || pred.meta.quality}
                </span>
              </div>
            )}

            <div className="pc-actions">
              <button className="pc-action-btn" onClick={onOpenReview} title="Full-size review">
                🔍 Review
              </button>
              <button className="pc-action-btn" onClick={onDownload} title="Download annotated">
                ⬇️ Save
              </button>
            </div>

            {item.scannedAt && (
              <div className="pc-timestamp" title={new Date(item.scannedAt).toString()}>
                <span className="pc-ts-icon">🕒</span>
                <span>Scanned {formatLocalTimestamp(item.scannedAt)}</span>
              </div>
            )}
          </>
        )}

        {item.status === 'no_match' && item.scannedAt && (
          <div className="pc-timestamp" title={new Date(item.scannedAt).toString()}>
            <span className="pc-ts-icon">🕒</span>
            <span>Scanned {formatLocalTimestamp(item.scannedAt)}</span>
          </div>
        )}
      </div>
    </div>
  );
};

const ReviewModal: React.FC<{
  item: PredictImageState;
  onClose: () => void;
  onDownload: () => void;
  onAnnotateRegion: (idx: number) => void;
  onUpdatePrediction: (newPrediction: PredictionResult) => void;
}> = ({ item, onClose, onDownload, onAnnotateRegion, onUpdatePrediction }) => {
  const pred = item.prediction as PredictionResult | null;
  const [isAdjusting, setIsAdjusting] = useState(false);
  const [adjustedRegions, setAdjustedRegions] = useState<any[] | null>(null);
  const imageFrameRef = useRef<HTMLDivElement>(null);
  
  const interactionRef = useRef<{
    idx: number;
    type: string;
    startX: number;
    startY: number;
    startBox: any;
  } | null>(null);

  useEffect(() => {
    if (isAdjusting && !adjustedRegions && pred) {
      setAdjustedRegions(JSON.parse(JSON.stringify(pred.regions || [])));
    }
  }, [isAdjusting, adjustedRegions, pred]);

  useEffect(() => {
    const handlePointerMove = (e: PointerEvent) => {
      const int = interactionRef.current;
      if (!int || !isAdjusting || !adjustedRegions || !imageFrameRef.current) return;
      
      const rect = imageFrameRef.current.getBoundingClientRect();
      const dxFrac = (e.clientX - int.startX) / rect.width;
      const dyFrac = (e.clientY - int.startY) / rect.height;

      setAdjustedRegions(prev => {
        if (!prev) return prev;
        const newRegions = [...prev];
        const box = { ...int.startBox };

        if (int.type === 'move') {
          box.x += dxFrac;
          box.y += dyFrac;
        } else {
          if (int.type.includes('n')) { box.y += dyFrac; box.height -= dyFrac; }
          if (int.type.includes('s')) { box.height += dyFrac; }
          if (int.type.includes('w')) { box.x += dxFrac; box.width -= dxFrac; }
          if (int.type.includes('e')) { box.width += dxFrac; }
        }

        if (box.width < 0.005) box.width = 0.005;
        if (box.height < 0.005) box.height = 0.005;

        newRegions[int.idx] = box;
        return newRegions;
      });
    };

    const handlePointerUp = () => { interactionRef.current = null; };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [isAdjusting, adjustedRegions]);

  const startInteraction = (e: React.PointerEvent, idx: number, type: string) => {
    if (!adjustedRegions) return;
    e.preventDefault();
    e.stopPropagation();
    interactionRef.current = {
      idx,
      type,
      startX: e.clientX,
      startY: e.clientY,
      startBox: { ...adjustedRegions[idx] }
    };
  };

  const deleteRegion = (idx: number) => {
    if (!adjustedRegions) return;
    setAdjustedRegions(prev => prev ? prev.filter((_, i) => i !== idx) : prev);
  };

  const saveRegions = () => {
    if (!adjustedRegions || !pred) return;
    onUpdatePrediction({
      ...pred,
      regions: adjustedRegions
    });
    setIsAdjusting(false);
  };

  if (!pred) return null;

  const activeRegions = isAdjusting ? (adjustedRegions || []) : (pred.regions || []);
  const wordRegions = activeRegions.filter((r) => r.category === 'word');
  const otherRegions = activeRegions.filter((r) => r.category !== 'word');

  return (
    <div className="review-modal-backdrop" onClick={onClose}>
      <div className="review-modal" onClick={(e) => e.stopPropagation()}>
        <div className="review-head">
          <div>
            <div className="review-title">
              <span className={`chip chip-${pred.category}`}>{pred.category}</span>
              <span className="review-label">{pred.label}</span>
              <span className="review-conf">{(pred.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="review-sub">
              {item.sourceName} · {activeRegions.length} element{activeRegions.length === 1 ? '' : 's'} located
              {wordRegions.length > 0 && ` · ${wordRegions.length} word match${wordRegions.length === 1 ? '' : 'es'}`}
              {pred.meta && ` · ${(pred.meta.elapsedMs / 1000).toFixed(1)}s · ${QUALITY_PRESETS[pred.meta.quality]?.title || pred.meta.quality} mode`}
              {' · Click regions to annotate'}
            </div>
          </div>
          <div className="review-actions">
            {!isAdjusting ? (
              <Button variant="ghost" icon="🛠️" onClick={() => setIsAdjusting(true)}>Edit Regions</Button>
            ) : (
              <>
                <Button variant="ghost" onClick={() => { setIsAdjusting(false); setAdjustedRegions(null); }}>Cancel</Button>
                <Button variant="primary" icon="💾" onClick={saveRegions}>Save Regions</Button>
              </>
            )}
            <Button variant="secondary" icon="⬇️" onClick={onDownload}>Save annotated image</Button>
            <button className="review-close" onClick={onClose}>×</button>
          </div>
        </div>

        <div className="review-body">
          <div className="review-image-wrap" ref={imageFrameRef}>
            {!isAdjusting ? (
              <RegionOverlay
                imageUrl={item.dataUrl}
                regions={activeRegions}
                onRegionClick={(idx) => onAnnotateRegion(idx)}
              />
            ) : (
              <div style={{ position: 'relative', display: 'inline-block', width: '100%', lineHeight: 0 }}>
                <img src={item.dataUrl} alt="scanned" style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 8 }} />
                {adjustedRegions?.map((r, i) => {
                  const catColor = CATEGORIES.find(c => c.key === r.category)?.color || '#6366f1';
                  return (
                    <div
                      key={i}
                      className="edit-highlight-box"
                      onPointerDown={(e) => startInteraction(e, i, 'move')}
                      style={{
                        left: `${r.x * 100}%`,
                        top: `${r.y * 100}%`,
                        width: `${r.width * 100}%`,
                        height: `${r.height * 100}%`,
                        borderColor: catColor,
                        backgroundColor: `${catColor}22`
                      }}
                    >
                      <span className="pv-image-highlight-num" style={{ background: catColor }}>{i + 1}</span>
                      <div className="edit-handle nw" onPointerDown={(e) => startInteraction(e, i, 'nw')} />
                      <div className="edit-handle ne" onPointerDown={(e) => startInteraction(e, i, 'ne')} />
                      <div className="edit-handle sw" onPointerDown={(e) => startInteraction(e, i, 'sw')} />
                      <div className="edit-handle se" onPointerDown={(e) => startInteraction(e, i, 'se')} />
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="review-sidebar">
            {wordRegions.length > 0 && (
              <div className="rs-section">
                <div className="rs-section-title priority">🔤 Word Matches (priority)</div>
                <div className="rs-regions">
                  {wordRegions.map((r) => {
                    const idx = activeRegions.indexOf(r);
                    const cat = CATEGORIES.find((c) => c.key === r.category);
                    return (
                      <RegionRow
                        key={idx}
                        region={r}
                        index={idx}
                        cat={cat}
                        onClick={() => !isAdjusting && onAnnotateRegion(idx)}
                        isAdjusting={isAdjusting}
                        onDelete={() => deleteRegion(idx)}
                      />
                    );
                  })}
                </div>
              </div>
            )}

            {otherRegions.length > 0 && (
              <div className="rs-section">
                <div className="rs-section-title">
                  {wordRegions.length > 0 ? '📌 Other Elements' : '📍 Located Elements'}
                </div>
                <div className="rs-regions">
                  {otherRegions.map((r) => {
                    const idx = activeRegions.indexOf(r);
                    const cat = CATEGORIES.find((c) => c.key === r.category);
                    return (
                      <RegionRow
                        key={idx}
                        region={r}
                        index={idx}
                        cat={cat}
                        onClick={() => !isAdjusting && onAnnotateRegion(idx)}
                        isAdjusting={isAdjusting}
                        onDelete={() => deleteRegion(idx)}
                      />
                    );
                  })}
                </div>
              </div>
            )}

            {activeRegions.length === 0 && (
              <div className="rs-empty">No specific regions located.</div>
            )}

            {pred.scores && (
              <div className="rs-section">
                <div className="rs-section-title">🧠 Overall Match Scores</div>
                <div className="rs-scores">
                  {pred.scores.textMatch > 0 && (
                    <BreakdownRow label="Text" value={pred.scores.textMatch} color="#06b6d4" />
                  )}
                  <BreakdownRow label="MobileNet" value={pred.scores.mobilenet} color="#6366f1" />
                  <BreakdownRow label="pHash" value={pred.scores.phash} color="#ec4899" />
                  <BreakdownRow label="Edges" value={pred.scores.edgeDensity} color="#10b981" />
                </div>
              </div>
            )}

            {pred.meta && (
              <div className="rs-section">
                <div className="rs-section-title">⚡ Performance</div>
                <div className="rs-perf">
                  <div className="rs-perf-row">
                    <span>Mode</span>
                    <strong>{QUALITY_PRESETS[pred.meta.quality]?.title || pred.meta.quality}</strong>
                  </div>
                  <div className="rs-perf-row">
                    <span>Elapsed</span>
                    <strong>{(pred.meta.elapsedMs / 1000).toFixed(2)}s</strong>
                  </div>
                  <div className="rs-perf-row">
                    <span>Windows scanned</span>
                    <strong>{pred.meta.windowsScanned}</strong>
                  </div>
                  <div className="rs-perf-row">
                    <span>Parallel workers</span>
                    <strong>{pred.meta.parallelWorkers}</strong>
                  </div>
                  <div className="rs-perf-row">
                    <span>Windows / sec</span>
                    <strong>
                      {pred.meta.elapsedMs > 0
                        ? (pred.meta.windowsScanned / (pred.meta.elapsedMs / 1000)).toFixed(0)
                        : '—'}
                    </strong>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const RegionRow: React.FC<{
  region: any;
  index: number;
  cat?: { icon: string; color: string };
  onClick: () => void;
  isAdjusting?: boolean;
  onDelete?: () => void;
}> = ({ region, index, cat, onClick, isAdjusting, onDelete }) => {
  const r = region;
  return (
    <div
      className={`rs-region-row ${!isAdjusting ? 'clickable' : ''} ${r.category === 'word' ? 'is-word' : ''}`}
      style={{ ['--rs-color' as any]: cat?.color || '#6366f1' }}
      onClick={onClick}
      title={!isAdjusting ? "Click to annotate / confirm / relabel" : ""}
    >
      <div className="rs-region-num">{index + 1}</div>
      <div className="rs-region-info">
        <div className="rs-region-label">{r.label}</div>
        <div className="rs-region-meta">
          <span className="rs-region-cat">{cat?.icon} {r.category}</span>
          <span className="rs-region-coords">
            {(r.x * 100).toFixed(0)}%, {(r.y * 100).toFixed(0)}% · {(r.width * 100).toFixed(0)}×{(r.height * 100).toFixed(0)}
          </span>
          {r.rotation !== undefined && r.rotation !== 0 && (
            <span className="rs-region-rot">↻ {r.rotation === -1 ? 'mirrored' : `${r.rotation}°`}</span>
          )}
        </div>
        {r.textMatch && (
          <div className="rs-region-text">
            <span className="rs-tm-label">OCR:</span>
            <span className="rs-tm-val">"{r.textMatch.ocrText.slice(0, 40)}{r.textMatch.ocrText.length > 40 ? '…' : ''}"</span>
            <span className="rs-tm-sim">{(r.textMatch.similarity * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>
      <div className="rs-region-conf">{(r.confidence * 100).toFixed(0)}%</div>
      {isAdjusting && (
        <button 
          className="rs-region-del-btn" 
          onClick={(e) => { e.stopPropagation(); onDelete && onDelete(); }}
          title="Delete Region"
        >
          ✕
        </button>
      )}
    </div>
  );
};

const BreakdownRow: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => (
  <div className="br-row">
    <span className="br-label">{label}</span>
    <div className="br-bar-wrap">
      <div className="br-bar" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%`, background: color }} />
    </div>
    <span className="br-val">{(value * 100).toFixed(0)}%</span>
  </div>
);

const StatPill: React.FC<{ label: string; value: number | string; tint?: boolean; accent?: boolean; priority?: boolean }> = ({ label, value, tint, accent, priority }) => (
  <div className={`stat-pill ${tint ? 'tint' : ''} ${accent ? 'accent' : ''} ${priority ? 'priority' : ''}`}>
    <div className="sp-value">{value}</div>
    <div className="sp-label">{label}</div>
  </div>
);

export default RecognitionStudio;