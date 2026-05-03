import React, { useEffect, useState } from 'react';
import './DatabaseExplorer.css';
import Card from '../../components/Card/Card';
import Button from '../../components/Button/Button';
import Toast from '../../components/Toast/Toast';
import { useToasts } from '../../hooks/useToasts';
import {
  getAllSamples,
  deleteSample,
  clearSamples,
  getAllOcrDocs,
  deleteOcrDoc,
  addOcrDoc,
  type TrainingSample,
  type OcrDocument,
} from '../../services/database';
import {
  getAllAnnotations,
  deleteAnnotation,
  clearAnnotations,
  type RegionAnnotation,
} from '../../services/annotationsService';
import { formatBytes } from '../../services/imageCompression';
import { resetIndex, indexAllSamples } from '../../services/recognitionService';
import { downloadPdfFromDataUrl } from '../../services/pdfExport';

const DatabaseExplorer: React.FC = () => {
  const { toasts, push, remove } = useToasts();
  const [tab, setTab] = useState<'samples' | 'docs' | 'knowledge'>('samples');
  const [samples, setSamples] = useState<TrainingSample[]>([]);
  const [docs, setDocs] = useState<OcrDocument[]>([]);
  const [annotations, setAnnotations] = useState<RegionAnnotation[]>([]);
  const [filter, setFilter] = useState<'all' | 'logo' | 'signature' | 'stamp' | 'word'>('all');
  const [selectedDoc, setSelectedDoc] = useState<OcrDocument | null>(null);
  const [selectedAnn, setSelectedAnn] = useState<RegionAnnotation | null>(null);
  const [editingDoc, setEditingDoc] = useState(false);
  const [docDraft, setDocDraft] = useState('');
  const [docNameDraft, setDocNameDraft] = useState('');
  const [activeDocPage, setActiveDocPage] = useState(0);

  const [confirmDialog, setConfirmDialog] = useState<{
    message: string;
    onConfirm: () => void;
  } | null>(null);

  const confirmAction = (message: string, onConfirm: () => void) => {
    setConfirmDialog({ message, onConfirm });
  };

  const refresh = async () => {
    const [s, d, a] = await Promise.all([getAllSamples(), getAllOcrDocs(), getAllAnnotations()]);
    setSamples(s.sort((a, b) => b.createdAt - a.createdAt));
    setDocs(d.sort((a, b) => b.createdAt - a.createdAt));
    setAnnotations(a.sort((x, y) => y.createdAt - x.createdAt));
  };

  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    if (selectedDoc) {
      setDocDraft(selectedDoc.text);
      setDocNameDraft(selectedDoc.name);
      setEditingDoc(false);
      setActiveDocPage(0);
    }
  }, [selectedDoc?.id]);

  const onDeleteSample = (id: string) => {
    confirmAction('Delete this training sample?', async () => {
      await deleteSample(id);
      push('success', 'Sample deleted.');
      await refresh();
      resetIndex();
      await indexAllSamples();
    });
  };

  const onClearAll = () => {
    confirmAction('Delete ALL training samples?', async () => {
      await clearSamples();
      resetIndex();
      push('success', 'All samples cleared.');
      await refresh();
    });
  };

  const onDeleteDoc = (id: string) => {
    confirmAction('Delete this OCR document? This also removes the saved searchable PDF.', async () => {
      await deleteOcrDoc(id);
      if (selectedDoc?.id === id) setSelectedDoc(null);
      push('success', 'Document deleted.');
      await refresh();
    });
  };

  const onDeleteAnn = (id: string) => {
    confirmAction('Delete this annotation?', async () => {
      await deleteAnnotation(id);
      if (selectedAnn?.id === id) setSelectedAnn(null);
      push('success', 'Annotation deleted.');
      await refresh();
    });
  };

  const onClearAnns = () => {
    confirmAction('Delete ALL annotations? This will not remove training samples already promoted.', async () => {
      await clearAnnotations();
      push('success', 'Annotations cleared.');
      await refresh();
    });
  };

  const saveDocEdits = async () => {
    if (!selectedDoc) return;
    const updated: OcrDocument = {
      ...selectedDoc,
      name: docNameDraft.trim() || selectedDoc.name,
      text: docDraft,
    };
    await addOcrDoc(updated);
    push('success', 'Document updated.');
    setSelectedDoc(updated);
    setEditingDoc(false);
    await refresh();
  };

  const cancelDocEdits = () => {
    if (selectedDoc) {
      setDocDraft(selectedDoc.text);
      setDocNameDraft(selectedDoc.name);
    }
    setEditingDoc(false);
  };

  const downloadDocPdf = (doc: OcrDocument) => {
    if (!doc.searchablePdfDataUrl) {
      push('error', 'No searchable PDF saved for this document.');
      return;
    }
    downloadPdfFromDataUrl(doc.searchablePdfDataUrl, doc.name);
  };

  const filteredSamples = samples.filter((s) => filter === 'all' || s.category === filter);
  const totalOrig = samples.reduce((a, b) => a + b.originalSize, 0);
  const totalComp = samples.reduce((a, b) => a + b.compressedSize, 0);
  const ratio = totalOrig > 0 ? ((1 - totalComp / totalOrig) * 100).toFixed(1) : '0';

  const annByVerdict = {
    confirmed: annotations.filter((a) => a.verdict === 'confirmed').length,
    relabeled: annotations.filter((a) => a.verdict === 'relabeled').length,
    rejected: annotations.filter((a) => a.verdict === 'rejected').length,
    note_only: annotations.filter((a) => a.verdict === 'note_only').length,
  };

  return (
    <div className="db-explorer">
      <Toast toasts={toasts} onRemove={remove} />

      <header className="page-head">
        <div>
          <h1>Database Explorer</h1>
          <p>Manage training samples, OCR documents (with their embedded searchable PDFs), and user feedback / knowledge.</p>
        </div>
      </header>

      <div className="db-tabs">
        <button className={`db-tab ${tab === 'samples' ? 'active' : ''}`} onClick={() => setTab('samples')}>
          🎯 Training Samples <span className="count">{samples.length}</span>
        </button>
        <button className={`db-tab ${tab === 'docs' ? 'active' : ''}`} onClick={() => setTab('docs')}>
          📚 OCR Documents <span className="count">{docs.length}</span>
        </button>
        <button className={`db-tab ${tab === 'knowledge' ? 'active' : ''}`} onClick={() => setTab('knowledge')}>
          🧠 Knowledge <span className="count">{annotations.length}</span>
        </button>
      </div>

      {tab === 'samples' && (
        <>
          <div className="db-toolbar">
            <div className="filter-chips">
              {(['all', 'word', 'logo', 'signature', 'stamp'] as const).map((f) => (
                <button
                  key={f}
                  className={`chip-btn ${filter === f ? 'active' : ''}`}
                  onClick={() => setFilter(f)}
                >
                  {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
                </button>
              ))}
            </div>
            <div className="toolbar-right">
              <div className="compression-note">
                💾 Saved {formatBytes(totalOrig - totalComp)} ({ratio}%) via compression
              </div>
              {samples.length > 0 && (
                <Button variant="danger" icon="🗑️" onClick={onClearAll}>
                  Clear All
                </Button>
              )}
            </div>
          </div>

          {filteredSamples.length === 0 ? (
            <Card>
              <div className="empty-state">
                <div className="empty-icon">📭</div>
                <div className="empty-title">No samples yet</div>
                <div className="empty-sub">Head to the Recognition Studio to train your first model.</div>
              </div>
            </Card>
          ) : (
            <div className="sample-grid">
              {filteredSamples.map((s) => (
                <div key={s.id} className="sample-card">
                  <div className="sample-img" style={{ backgroundImage: `url(${s.imageData})` }} />
                  <div className="sample-info">
                    <div className="sample-label">{s.label}</div>
                    <div className="sample-meta">
                      <span className={`cat-tag cat-${s.category}`}>{s.category}</span>
                      <span className="file-size">{formatBytes(s.compressedSize)}</span>
                    </div>
                    {s.ocrText && (
                      <div className="sample-ocr" title={s.ocrText}>
                        🔤 {s.ocrText.length > 40 ? s.ocrText.slice(0, 40) + '…' : s.ocrText}
                      </div>
                    )}
                    {s.origin === 'annotation' && (
                      <div className="sample-origin">↳ from feedback</div>
                    )}
                  </div>
                  <button className="sample-del" onClick={() => onDeleteSample(s.id)} title="Delete">
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {tab === 'docs' && (
        <div className="docs-layout">
          <div className="docs-list">
            {docs.length === 0 ? (
              <Card>
                <div className="empty-state">
                  <div className="empty-icon">📄</div>
                  <div className="empty-title">No OCR documents</div>
                  <div className="empty-sub">Process a scanned PDF in the OCR Workspace.</div>
                </div>
              </Card>
            ) : (
              docs.map((d) => {
                const hasImages = d.pageRecords?.some((p) => !!p.imageDataUrl);
                return (
                  <div
                    key={d.id}
                    className={`doc-item ${selectedDoc?.id === d.id ? 'active' : ''}`}
                    onClick={() => setSelectedDoc(d)}
                  >
                    <div className="doc-icon">📄</div>
                    <div className="doc-info">
                      <div className="doc-name">{d.name}</div>
                      <div className="doc-meta">
                        {d.pages} page{d.pages === 1 ? '' : 's'} · {d.language} ·{' '}
                        {new Date(d.createdAt).toLocaleDateString()}
                      </div>
                      <div className="doc-badges">
                        {hasImages && <span className="doc-badge good">🖼 pages</span>}
                        {d.searchablePdfDataUrl && <span className="doc-badge good">📄 searchable PDF</span>}
                      </div>
                    </div>
                    <div className="doc-actions">
                      {d.searchablePdfDataUrl && (
                        <button
                          className="doc-dl"
                          onClick={(e) => {
                            e.stopPropagation();
                            downloadDocPdf(d);
                          }}
                          title="Download searchable PDF"
                        >
                          ⬇
                        </button>
                      )}
                      <button
                        className="doc-del"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteDoc(d.id);
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          <div className="docs-preview">
            {selectedDoc ? (
              <Card
                title={editingDoc ? 'Editing Document' : selectedDoc.name}
                subtitle={`${selectedDoc.pages} pages · Language: ${selectedDoc.language}${selectedDoc.searchablePdfDataUrl ? ' · Searchable PDF embedded' : ''}`}
                right={
                  editingDoc ? (
                    <div style={{ display: 'flex', gap: 8 }}>
                      <Button variant="primary" icon="💾" onClick={saveDocEdits}>Save</Button>
                      <Button variant="ghost" onClick={cancelDocEdits}>Cancel</Button>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', gap: 8 }}>
                      {selectedDoc.searchablePdfDataUrl && (
                        <Button variant="secondary" icon="⬇" onClick={() => downloadDocPdf(selectedDoc)}>
                          Download PDF
                        </Button>
                      )}
                      <Button variant="secondary" icon="✏️" onClick={() => setEditingDoc(true)}>Edit</Button>
                    </div>
                  )
                }
              >
                {editingDoc ? (
                  <>
                    <label className="edit-label">Document name</label>
                    <input
                      className="doc-name-input"
                      value={docNameDraft}
                      onChange={(e) => setDocNameDraft(e.target.value)}
                    />
                    <label className="edit-label" style={{ marginTop: 12 }}>Extracted text (all pages)</label>
                    <textarea
                      className="doc-text-edit"
                      value={docDraft}
                      onChange={(e) => setDocDraft(e.target.value)}
                      spellCheck={true}
                    />
                  </>
                ) : (
                  <>
                    {selectedDoc.pageRecords && selectedDoc.pageRecords.length > 0 ? (
                      <>
                        {selectedDoc.pageRecords.length > 1 && (
                          <div className="doc-page-tabs">
                            {selectedDoc.pageRecords.map((p, i) => (
                              <button
                                key={p.pageNumber}
                                className={`doc-page-tab ${activeDocPage === i ? 'active' : ''}`}
                                onClick={() => setActiveDocPage(i)}
                              >
                                Page {p.pageNumber}
                              </button>
                            ))}
                          </div>
                        )}
                        <div className="doc-page-split">
                          {selectedDoc.pageRecords[activeDocPage]?.imageDataUrl ? (
                            <div className="doc-page-image">
                              <img
                                src={selectedDoc.pageRecords[activeDocPage].imageDataUrl}
                                alt={`page ${selectedDoc.pageRecords[activeDocPage].pageNumber}`}
                              />
                            </div>
                          ) : (
                            <div className="doc-page-image empty">
                              <div className="empty-state" style={{ padding: 32 }}>
                                <div className="empty-icon">🖼</div>
                                <div className="empty-sub">No page image saved</div>
                              </div>
                            </div>
                          )}
                          <pre className="doc-text">
                            {selectedDoc.pageRecords[activeDocPage]?.text || '(empty)'}
                          </pre>
                        </div>
                      </>
                    ) : (
                      <pre className="doc-text">{selectedDoc.text || '(empty)'}</pre>
                    )}
                  </>
                )}
              </Card>
            ) : (
              <Card>
                <div className="empty-state">
                  <div className="empty-icon">👈</div>
                  <div className="empty-title">Select a document</div>
                  <div className="empty-sub">Click an item on the left to preview its pages, edit text, or download the searchable PDF.</div>
                </div>
              </Card>
            )}
          </div>
        </div>
      )}

      {tab === 'knowledge' && (
        <>
          <div className="knowledge-summary">
            <div className="ks-stat ks-stat-confirmed">
              <div className="ks-stat-num">{annByVerdict.confirmed}</div>
              <div className="ks-stat-lbl">Confirmed</div>
            </div>
            <div className="ks-stat ks-stat-relabeled">
              <div className="ks-stat-num">{annByVerdict.relabeled}</div>
              <div className="ks-stat-lbl">Relabeled</div>
            </div>
            <div className="ks-stat ks-stat-rejected">
              <div className="ks-stat-num">{annByVerdict.rejected}</div>
              <div className="ks-stat-lbl">Rejected</div>
            </div>
            <div className="ks-stat ks-stat-notes">
              <div className="ks-stat-num">{annByVerdict.note_only}</div>
              <div className="ks-stat-lbl">Notes</div>
            </div>
            <div className="ks-stat-grow" />
            {annotations.length > 0 && (
              <Button variant="danger" icon="🗑️" onClick={onClearAnns}>Clear All</Button>
            )}
          </div>

          {annotations.length === 0 ? (
            <Card>
              <div className="empty-state">
                <div className="empty-icon">🧠</div>
                <div className="empty-title">No feedback yet</div>
                <div className="empty-sub">Annotate detected regions in the Recognition Studio to build your knowledge base.</div>
              </div>
            </Card>
          ) : (
            <div className="knowledge-layout">
              <div className="knowledge-list">
                {annotations.map((a) => (
                  <div
                    key={a.id}
                    className={`ann-row ${selectedAnn?.id === a.id ? 'active' : ''}`}
                    onClick={() => setSelectedAnn(a)}
                  >
                    {a.regionCropDataUrl ? (
                      <div className="ann-row-img" style={{ backgroundImage: `url(${a.regionCropDataUrl})` }} />
                    ) : (
                      <div className="ann-row-img empty">—</div>
                    )}
                    <div className="ann-row-body">
                      <div className="ann-row-top">
                        <span className={`ann-verdict-badge v-${a.verdict}`}>{a.verdict}</span>
                        <span className="ann-row-label">
                          {a.correctedLabel || a.originalLabel}
                        </span>
                        {a.promotedToTrainingSampleId && (
                          <span className="ann-promoted">✓ trained</span>
                        )}
                      </div>
                      {a.comment && <div className="ann-row-comment">{a.comment}</div>}
                      <div className="ann-row-meta">
                        {new Date(a.createdAt).toLocaleString()} · {a.sourceImageName}
                      </div>
                    </div>
                    <button
                      className="ann-row-del"
                      onClick={(e) => { e.stopPropagation(); onDeleteAnn(a.id); }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
              <div className="knowledge-detail">
                {selectedAnn ? (
                  <Card title="Annotation Details">
                    <div className="ad-image-row">
                      {selectedAnn.regionCropDataUrl && (
                        <div className="ad-crop">
                          <div className="ad-crop-title">Region</div>
                          <img src={selectedAnn.regionCropDataUrl} alt="region" />
                        </div>
                      )}
                      <div className="ad-context">
                        <div className="ad-crop-title">Source Image</div>
                        <img src={selectedAnn.sourceImageDataUrl} alt="context" />
                      </div>
                    </div>

                    <div className="ad-grid">
                      <div>
                        <div className="ad-lbl">Verdict</div>
                        <div className={`ann-verdict-badge v-${selectedAnn.verdict}`}>{selectedAnn.verdict}</div>
                      </div>
                      <div>
                        <div className="ad-lbl">Original</div>
                        <div className="ad-val">{selectedAnn.originalLabel} ({selectedAnn.originalCategory})</div>
                      </div>
                      {selectedAnn.correctedLabel && (
                        <div>
                          <div className="ad-lbl">Corrected</div>
                          <div className="ad-val ad-val-correction">
                            {selectedAnn.correctedLabel} ({selectedAnn.correctedCategory})
                          </div>
                        </div>
                      )}
                      <div>
                        <div className="ad-lbl">Confidence</div>
                        <div className="ad-val">{(selectedAnn.originalConfidence * 100).toFixed(1)}%</div>
                      </div>
                    </div>

                    {selectedAnn.comment && (
                      <>
                        <div className="ad-lbl" style={{ marginTop: 14 }}>User Comment</div>
                        <div className="ad-quote">"{selectedAnn.comment}"</div>
                      </>
                    )}

                    {selectedAnn.aiSuggestion && (
                      <div className="ad-ai-box">
                        <div className="ad-ai-head">🌐 AI-Assisted Context</div>
                        <div className="ad-ai-src">{selectedAnn.aiSuggestion.source}</div>
                        <div className="ad-ai-text">{selectedAnn.aiSuggestion.reasoning}</div>
                      </div>
                    )}

                    {selectedAnn.promotedToTrainingSampleId && (
                      <div className="ad-promoted-box">
                        ✓ This annotation became training sample{' '}
                        <code>{selectedAnn.promotedToTrainingSampleId.slice(0, 16)}…</code>
                      </div>
                    )}
                  </Card>
                ) : (
                  <Card>
                    <div className="empty-state">
                      <div className="empty-icon">👈</div>
                      <div className="empty-title">Select an annotation</div>
                      <div className="empty-sub">Click an entry on the left to view full details.</div>
                    </div>
                  </Card>
                )}
              </div>
            </div>
          )}
        </>
      )}
      
      {confirmDialog && (
        <div className="modal-backdrop" onClick={() => setConfirmDialog(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Confirm Action</h3>
            <p>{confirmDialog.message}</p>
            <div className="modal-actions">
              <Button variant="ghost" onClick={() => setConfirmDialog(null)}>Cancel</Button>
              <Button variant="danger" onClick={() => {
                confirmDialog.onConfirm();
                setConfirmDialog(null);
              }}>Confirm</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DatabaseExplorer;