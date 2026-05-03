import React, { useRef, useState, useEffect } from 'react';
import './OcrWorkspace.css';
import Button from '../../components/Button/Button';
import Card from '../../components/Card/Card';
import ProgressBar from '../../components/ProgressBar/ProgressBar';
import Toast from '../../components/Toast/Toast';
import { useToasts } from '../../hooks/useToasts';
import { useCancellable, isCancelledError } from '../../hooks/useCancellable';
import {
  performOcrOnFile,
  terminateOcrWorker,
  type OcrProgress,
  type OcrResult,
  type OcrQuality,
} from '../../services/ocrService';
import {
  exportSearchablePdf,
  exportPlainText,
  buildSearchablePdfBlob,
  blobToDataUrl,
} from '../../services/pdfExport';
import { addOcrDoc, type OcrDocument } from '../../services/database';
import { ocrWorkspaceState, resetOcrState } from '../../state/workspaceState';
import { autoCorrectOcrText } from '../../services/llmService';

const LANGS = [
  { code: 'eng', name: 'English' },
  { code: 'spa', name: 'Spanish' },
  { code: 'fra', name: 'French' },
  { code: 'deu', name: 'German' },
  { code: 'ita', name: 'Italian' },
  { code: 'por', name: 'Portuguese' },
  { code: 'chi_sim', name: 'Chinese (Simplified)' },
  { code: 'jpn', name: 'Japanese' },
  { code: 'kor', name: 'Korean' },
  { code: 'ara', name: 'Arabic' },
  { code: 'rus', name: 'Russian' },
  { code: 'hin', name: 'Hindi' },
  { code: 'ind', name: 'Indonesian' },
];

const QUALITY_PRESETS: { key: OcrQuality; title: string; desc: string; speed: string; variants: number }[] = [
  { key: 'fast', title: 'Fast', desc: 'Single-pass with light contrast enhancement. Good for clean, modern documents.', speed: '~2–4s/pg', variants: 1 },
  { key: 'balanced', title: 'Balanced', desc: 'Three enhanced variants voted by confidence. Recovers most faded text.', speed: '~5–9s/pg', variants: 3 },
  { key: 'precise', title: 'Precise', desc: 'Five enhanced variants + higher DPI + Sauvola binarization. Best for very faint or damaged scans.', speed: '~10–18s/pg', variants: 5 },
];

const OcrWorkspace: React.FC = () => {
  const [file, setFile] = useState<File | null>(ocrWorkspaceState.file);
  const [fileName, setFileName] = useState<string>(ocrWorkspaceState.fileName);
  const [lang, setLang] = useState(ocrWorkspaceState.lang);
  const [quality, setQuality] = useState<OcrQuality>(
    (ocrWorkspaceState as any).quality || 'precise'
  );
  const [result, setResult] = useState<OcrResult | null>(ocrWorkspaceState.result);
  const [savedDocId, setSavedDocId] = useState<string | null>(null);
  const [progress, setProgress] = useState<OcrProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [exporting, setExporting] = useState<{ stage: string; progress: number } | null>(null);
  const [buildingPdfForDb, setBuildingPdfForDb] = useState<{ stage: string; progress: number } | null>(null);
  const [activePage, setActivePage] = useState(ocrWorkspaceState.activePage);
  const [isCorrecting, setIsCorrecting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const { toasts, push, remove } = useToasts();
  const { start, cancel, clear } = useCancellable();

  useEffect(() => { ocrWorkspaceState.file = file; ocrWorkspaceState.fileName = file ? file.name : fileName; }, [file, fileName]);
  useEffect(() => { ocrWorkspaceState.lang = lang; }, [lang]);
  useEffect(() => { ocrWorkspaceState.result = result; }, [result]);
  useEffect(() => { ocrWorkspaceState.activePage = activePage; }, [activePage]);
  useEffect(() => { (ocrWorkspaceState as any).quality = quality; }, [quality]);

  const handleFile = (f: File | null) => {
    setFile(f);
    setFileName(f ? f.name : '');
    setResult(null);
    setActivePage(0);
    setSavedDocId(null);
    ocrWorkspaceState.edited = false;
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (running) return;
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const saveToDatabase = async (
    r: OcrResult,
    f: File,
    language: string,
    edited = false
  ): Promise<string> => {
    setBuildingPdfForDb({ stage: 'Starting…', progress: 0 });
    try {
      const blob = await buildSearchablePdfBlob(r, (stage, progress) => {
        setBuildingPdfForDb({ stage, progress });
      });
      const pdfDataUrl = await blobToDataUrl(blob);

      const docId = `doc_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      const doc: OcrDocument = {
        id: docId,
        name: edited ? `${f.name} (edited)` : f.name,
        pages: r.pages.length,
        text: r.fullText,
        language,
        createdAt: Date.now(),
        pageRecords: r.pages.map((p) => ({
          pageNumber: p.pageNumber,
          text: p.text,
          imageDataUrl: p.imageDataUrl,
        })),
        searchablePdfDataUrl: pdfDataUrl,
        originalMimeType: f.type || (f.name.toLowerCase().endsWith('.pdf') ? 'application/pdf' : 'image/*'),
        originalFileName: f.name,
      };
      await addOcrDoc(doc);
      return docId;
    } finally {
      setBuildingPdfForDb(null);
    }
  };

  const run = async () => {
    if (!file) return;
    setRunning(true);
    setResult(null);
    setSavedDocId(null);
    setProgress({ stage: 'Starting…', progress: 0 });
    const token = start();
    try {
      const r = await performOcrOnFile(
        file,
        lang,
        (p) => setProgress(p),
        token,
        { quality }
      );
      if (token.cancelled) return;
      setResult(r);
      setActivePage(0);
      ocrWorkspaceState.edited = false;

      try {
        setProgress({ stage: 'Saving to database…', progress: 0.95 });
        const docId = await saveToDatabase(r, file, lang, false);
        setSavedDocId(docId);
        const confNote = r.avgConfidence ? ` · avg confidence ${r.avgConfidence.toFixed(0)}%` : '';
        push('success', `OCR complete — ${r.pages.length} page(s) saved${confNote}.`);
      } catch (e: any) {
        console.error('Failed to save to DB:', e);
        push('error', `OCR done, but saving failed: ${e?.message || e}`);
      }
    } catch (e: any) {
      if (isCancelledError(e)) {
        push('info', 'OCR cancelled.');
      } else {
        console.error(e);
        push('error', `OCR failed: ${e?.message || 'Unknown error'}`);
      }
    } finally {
      setRunning(false);
      setProgress(null);
      clear();
    }
  };

  const stop = async () => {
    cancel();
    push('info', 'Stopping OCR…');
    try { await terminateOcrWorker(); } catch {}
  };

  const updatePageText = (pageIndex: number, newText: string) => {
    if (!result) return;
    const nextPages = result.pages.map((p, i) =>
      i === pageIndex ? { ...p, text: newText } : p
    );
    const next: OcrResult = {
      ...result,
      pages: nextPages,
      fullText: nextPages.map((p) => p.text).join('\n\n'),
    };
    ocrWorkspaceState.edited = true;
    setResult(next);
  };

  const handleAutoCorrectPage = async () => {
    const currentPageToCorrect = result?.pages[activePage];
    if (!result || !currentPageToCorrect || !currentPageToCorrect.text) return;
    setIsCorrecting(true);
    try {
      const corrected = await autoCorrectOcrText(currentPageToCorrect.text);
      updatePageText(activePage, corrected);
      push('success', 'Page text auto-corrected using AI.');
    } catch (e: any) {
      console.error("Auto-correct failed", e);
      push('error', `AI correction failed: ${e?.message || 'Unknown error'}`);
    } finally {
      setIsCorrecting(false);
    }
  };

  const saveEditedToDb = async () => {
    if (!result || !file) return;
    try {
      const docId = await saveToDatabase(result, file, lang, true);
      setSavedDocId(docId);
      push('success', 'Edited version saved to database with refreshed searchable PDF.');
    } catch (e: any) {
      push('error', `Save failed: ${e?.message || e}`);
    }
  };

  const saveSearchable = async () => {
    if (!result || !file) return;
    setExporting({ stage: 'Starting…', progress: 0 });
    try {
      await exportSearchablePdf(result, file.name, (stage, progress) => {
        setExporting({ stage, progress });
      });
      push('success', 'Searchable PDF saved — text aligned to original layout.');
    } catch (e: any) {
      console.error(e);
      push('error', `Export failed: ${e?.message || 'Unknown'}`);
    } finally {
      setExporting(null);
    }
  };

  const saveText = () => {
    if (!result || !file) return;
    exportPlainText(result.fullText, file.name);
    push('success', 'Text file saved.');
  };

  const clearAll = () => {
    if (running) return;
    setFile(null);
    setFileName('');
    setResult(null);
    setActivePage(0);
    setSavedDocId(null);
    resetOcrState();
    push('info', 'Workspace cleared.');
  };

  const currentPage = result?.pages[activePage];
  const cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;
  const poolSize = Math.min(4, Math.max(2, cores - 1));

  return (
    <div className="ocr-workspace">
      <Toast toasts={toasts} onRemove={remove} />

      <header className="page-head">
        <div>
          <h1>OCR Workspace</h1>
          <p>
            <strong>Precision OCR</strong> with multi-variant image enhancement — the system creates up to 5 contrast-boosted versions of each page (CLAHE, Sauvola binarization, unsharp mask, etc.), OCRs them in parallel across {poolSize} workers, and picks the highest-confidence result. Recovers text from <strong>faded, low-contrast, or damaged scans</strong>.
          </p>
        </div>
        {(file || result) && !running && (
          <Button variant="ghost" icon="🗑️" onClick={clearAll}>Clear Workspace</Button>
        )}
      </header>

      <div className="ocr-grid">
        <div className="ocr-left">
          <Card title="1. Upload Document" subtitle="PDF, PNG, JPG up to 50MB">
            <div
              className={`dropzone ${file ? 'has-file' : ''} ${running ? 'disabled' : ''}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => !running && fileRef.current?.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept="application/pdf,image/*"
                hidden
                onChange={(e) => handleFile(e.target.files?.[0] || null)}
              />
              {file ? (
                <div className="file-info">
                  <div className="file-icon">{file.type.includes('pdf') ? '📕' : '🖼️'}</div>
                  <div>
                    <div className="file-name">{file.name}</div>
                    <div className="file-meta">
                      {(file.size / 1024 / 1024).toFixed(2)} MB · {file.type || 'unknown'}
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="drop-icon">⬆</div>
                  <div className="drop-title">Drop file or click to browse</div>
                  <div className="drop-sub">Scanned PDFs and images work best</div>
                </>
              )}
            </div>
          </Card>

          <Card title="2. Precision Mode" subtitle="More variants = more accurate on faint text">
            <div className="quality-group">
              {QUALITY_PRESETS.map((q) => {
                const active = quality === q.key;
                return (
                  <button
                    key={q.key}
                    type="button"
                    className={`quality-btn ${active ? 'active' : ''}`}
                    onClick={() => !running && setQuality(q.key)}
                    disabled={running}
                  >
                    <div className="qb-head">
                      <span className="qb-title">{q.title}</span>
                      <span className="qb-speed">{q.speed}</span>
                    </div>
                    <div className="qb-desc">{q.desc}</div>
                    <div className="qb-variants">
                      <span className="qb-variant-dot" />
                      {q.variants} enhancement variant{q.variants === 1 ? '' : 's'} per page
                    </div>
                  </button>
                );
              })}
            </div>
            {quality === 'precise' && (
              <div className="precision-hint">
                <span className="ph-icon">🔬</span>
                <div>
                  <div className="ph-title">Maximum precision mode</div>
                  <div className="ph-desc">
                    Each page is rendered at ~2.6× DPI, then enhanced into 5 variants: CLAHE grayscale, Sauvola binarization, high-contrast gamma, unsharp mask, and inverted (for white-on-dark text). Tesseract OCRs all 5 in parallel, and the result with the highest confidence × word-count score wins.
                  </div>
                </div>
              </div>
            )}
          </Card>

          <Card title="3. Language" subtitle="Tesseract language model">
            <select
              className="select"
              value={lang}
              onChange={(e) => setLang(e.target.value)}
              disabled={running}
            >
              {LANGS.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name} ({l.code})
                </option>
              ))}
            </select>
          </Card>

          <Card title="4. Run OCR" subtitle="Auto-saves to database + builds searchable PDF">
            {!running ? (
              <Button
                variant="primary"
                icon="⚡"
                onClick={run}
                disabled={!file}
                className="full-width"
              >
                Extract Text & Save
              </Button>
            ) : (
              <Button
                variant="danger"
                icon="⏹"
                onClick={stop}
                className="full-width"
              >
                Stop OCR
              </Button>
            )}

            {progress && (
              <div style={{ marginTop: 16 }}>
                <ProgressBar label={progress.stage} value={progress.progress * 100} />
                {progress.subStage && (
                  <div className="sub-stage">{progress.subStage}</div>
                )}
              </div>
            )}

            {buildingPdfForDb && (
              <div style={{ marginTop: 14 }}>
                <ProgressBar label={buildingPdfForDb.stage} value={buildingPdfForDb.progress * 100} />
                <p className="hint" style={{ marginTop: 8 }}>
                  Building searchable PDF with word-accurate text overlay…
                </p>
              </div>
            )}
          </Card>

          {result && savedDocId && (
            <Card title="✓ Saved to Database" subtitle="This document is persistent and text-searchable">
              <div className="saved-badge">
                <span className="saved-icon">🗄️</span>
                <div>
                  <div className="saved-title">Auto-saved</div>
                  <div className="saved-sub">
                    {result.pages.length} page{result.pages.length === 1 ? '' : 's'}
                    {result.avgConfidence !== undefined && ` · avg ${result.avgConfidence.toFixed(0)}% confidence`}
                  </div>
                </div>
              </div>
              <p className="hint" style={{ marginTop: 10 }}>
                💡 Use the <strong>Text Search</strong> tab to find any word across all saved documents.
              </p>
            </Card>
          )}

          {result && (
            <Card title="5. Export" subtitle={ocrWorkspaceState.edited ? '✏️ You have unsaved edits' : 'Download copies'}>
              <div className="export-row">
                {ocrWorkspaceState.edited && (
                  <Button variant="secondary" icon="💾" onClick={saveEditedToDb} disabled={!!exporting || !!buildingPdfForDb} className="full-width">
                    Save Edits to Database
                  </Button>
                )}
                <Button variant="primary" icon="📄" onClick={saveSearchable} loading={!!exporting} disabled={!!exporting} className="full-width">
                  {exporting ? 'Building PDF…' : 'Download Searchable PDF'}
                </Button>
                <Button variant="secondary" icon="📝" onClick={saveText} disabled={!!exporting} className="full-width">
                  Download Text File
                </Button>
              </div>
              {exporting && (
                <div style={{ marginTop: 14 }}>
                  <ProgressBar label={exporting.stage} value={exporting.progress * 100} />
                  <p className="hint" style={{ marginTop: 8 }}>
                    Aligning invisible text to each word's exact position…
                  </p>
                </div>
              )}
            </Card>
          )}
        </div>

        <div className="ocr-right">
          <Card
            title="Results"
            subtitle={
              result
                ? `${result.pages.length} page(s) · ${result.language}${
                    result.avgConfidence !== undefined ? ` · avg ${result.avgConfidence.toFixed(0)}% confidence` : ''
                  }${ocrWorkspaceState.edited ? ' · Edited' : ''}`
                : 'Run OCR to see output'
            }
          >
            {!result && !running && (
              <div className="empty-state">
                <div className="empty-icon">📄</div>
                <div className="empty-title">No results yet</div>
                <div className="empty-sub">Upload a file and run OCR to extract text.</div>
              </div>
            )}

            {!result && running && progress && (
              <div className="running-state">
                <div className="running-spinner" />
                <div className="running-title">{progress.stage}</div>
                <div className="running-sub">
                  {progress.currentPage && progress.totalPages
                    ? `Page ${progress.currentPage} of ${progress.totalPages}`
                    : 'Please wait…'}
                </div>
                {progress.subStage && (
                  <div className="running-substage">{progress.subStage}</div>
                )}
                <div className="running-progress">
                  <ProgressBar value={progress.progress * 100} />
                </div>
                <Button variant="danger" icon="⏹" onClick={stop}>Stop</Button>
              </div>
            )}

            {result && currentPage && (
              <>
                {result.pages.length > 1 && (
                  <div className="page-tabs">
                    {result.pages.map((p, i) => {
                      const conf = p.confidence;
                      const color = conf === undefined
                        ? 'neutral'
                        : conf >= 85 ? 'good'
                        : conf >= 65 ? 'ok'
                        : 'low';
                      return (
                        <button
                          key={i}
                          className={`page-tab ${activePage === i ? 'active' : ''} conf-${color}`}
                          onClick={() => setActivePage(i)}
                          title={conf !== undefined ? `Confidence: ${conf.toFixed(1)}%` : ''}
                        >
                          Page {p.pageNumber}
                          {conf !== undefined && (
                            <span className="page-tab-conf">{conf.toFixed(0)}%</span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}

                {currentPage.winningVariant && currentPage.variantScores && currentPage.variantScores.length > 1 && (
                  <div className="variant-summary">
                    <div className="vs-title">
                      🔬 Multi-variant analysis
                      <span className="vs-winner">
                        winner: <strong>{currentPage.winningVariant}</strong>
                      </span>
                    </div>
                    <div className="vs-bars">
                      {currentPage.variantScores.map((v) => {
                        const isWinner = v.variant === currentPage.winningVariant;
                        return (
                          <div key={v.variant} className={`vs-bar ${isWinner ? 'winner' : ''}`}>
                            <div className="vs-bar-label">
                              <span className="vs-bar-name">{v.variant}</span>
                              <span className="vs-bar-stats">
                                {v.confidence.toFixed(0)}% · {v.wordCount}w
                              </span>
                            </div>
                            <div className="vs-bar-track">
                              <div
                                className="vs-bar-fill"
                                style={{ width: `${Math.max(2, v.confidence)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                <div className="result-split">
                  <div className="result-preview">
                    <img src={currentPage.imageDataUrl} alt="page" />
                  </div>
                  <div className="result-text">
                    <div className="result-head">
                      <span>
                        Extracted Text {ocrWorkspaceState.edited && <span className="edited-pill">edited</span>}
                      </span>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button
                          className="copy-btn ai-correct-btn"
                          onClick={handleAutoCorrectPage}
                          disabled={isCorrecting || !currentPage.text}
                          title="Use AI to automatically fix spelling and formatting errors"
                        >
                          {isCorrecting ? '✨ Correcting...' : '✨ AI Auto-Correct'}
                        </button>
                        <button
                          className="copy-btn"
                          onClick={() => {
                            navigator.clipboard.writeText(currentPage.text);
                            push('info', 'Text copied to clipboard.');
                          }}
                        >
                          Copy
                        </button>
                      </div>
                    </div>
                    <textarea
                      value={currentPage.text}
                      onChange={(e) => updatePageText(activePage, e.target.value)}
                      placeholder="Extracted text appears here. You can edit it directly."
                      spellCheck={true}
                    />
                    <div className="text-stats">
                      {currentPage.text.length} chars ·{' '}
                      {currentPage.text.trim().split(/\s+/).filter(Boolean).length} words
                      {currentPage.confidence !== undefined && (
                        <span className={`text-conf conf-${
                          currentPage.confidence >= 85 ? 'good' :
                          currentPage.confidence >= 65 ? 'ok' : 'low'
                        }`}>
                          · {currentPage.confidence.toFixed(1)}% confidence
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default OcrWorkspace;