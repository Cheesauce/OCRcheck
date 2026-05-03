import React, { useEffect, useMemo, useState, useRef } from 'react';
import './TextSearch.css';
import Card from '../../components/Card/Card';
import Button from '../../components/Button/Button';
import Toast from '../../components/Toast/Toast';
import { useToasts } from '../../hooks/useToasts';
import { getAllOcrDocs, updateOcrDocPage, updateOcrDocPageWords, type OcrDocument, type OcrPageRecord } from '../../services/database';
import { searchTextState } from '../../state/workspaceState';
import { downloadPdfFromDataUrl } from '../../services/pdfExport';

interface PageHit {
  docId: string;
  docName: string;
  language: string;
  pageNumber: number;
  pageText: string;
  pageImageDataUrl: string | null;
  matchCount: number;
  snippets: Snippet[];
  matches: { start: number; end: number }[];
  words?: { text: string; x: number; y: number; w: number; h: number }[];
  createdAt: number;
  hasSearchablePdf: boolean;
}

interface Snippet {
  before: string;
  match: string;
  after: string;
  absoluteIndex: number;
}

interface DocumentHits {
  doc: OcrDocument;
  pageHits: PageHit[];
  totalMatches: number;
}

interface HighlightBox {
  xFrac: number;
  yFrac: number;
  wFrac: number;
  hFrac: number;
  wordIndices: number[];
}

const MAX_SNIPPETS_PER_PAGE = 5;
const SNIPPET_CONTEXT = 60;

function normalizeForSearch(s: string): string {
  return s.replace(/\s+/g, ' ');
}

function findAllMatches(
  haystack: string,
  needle: string,
  wholeWord: boolean
): { start: number; end: number }[] {
  if (!needle) return [];
  const hits: { start: number; end: number }[] = [];
  const hayLower = haystack.toLowerCase();
  const needleLower = needle.toLowerCase();

  let idx = 0;
  while (true) {
    const found = hayLower.indexOf(needleLower, idx);
    if (found < 0) break;
    const end = found + needleLower.length;

    if (wholeWord) {
      const before = found === 0 ? '' : haystack[found - 1];
      const after = end >= haystack.length ? '' : haystack[end];
      const isBoundary = (c: string) => c === '' || !/[\p{L}\p{N}_]/u.test(c);
      if (!(isBoundary(before) && isBoundary(after))) {
        idx = found + 1;
        continue;
      }
    }

    hits.push({ start: found, end });
    idx = end;
    if (hits.length > 2000) break;
  }
  return hits;
}

function buildSnippets(
  pageText: string,
  matches: { start: number; end: number }[]
): Snippet[] {
  const snippets: Snippet[] = [];
  for (let i = 0; i < Math.min(MAX_SNIPPETS_PER_PAGE, matches.length); i++) {
    const m = matches[i];
    const beforeStart = Math.max(0, m.start - SNIPPET_CONTEXT);
    const afterEnd = Math.min(pageText.length, m.end + SNIPPET_CONTEXT);
    let before = pageText.slice(beforeStart, m.start);
    let after = pageText.slice(m.end, afterEnd);

    if (beforeStart > 0) before = '…' + before.replace(/^\S*\s/, '');
    if (afterEnd < pageText.length) after = after.replace(/\s\S*$/, '') + '…';

    snippets.push({
      before: before.replace(/\s+/g, ' '),
      match: pageText.slice(m.start, m.end),
      after: after.replace(/\s+/g, ' '),
      absoluteIndex: m.start,
    });
  }
  return snippets;
}

const measureCanvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
const measureCtx = measureCanvas ? measureCanvas.getContext('2d') : null;
if (measureCtx) {
  measureCtx.font = 'bold 100px sans-serif';
}

function measureTextWidth(text: string) {
  if (!measureCtx) return text.length;
  return measureCtx.measureText(text).width;
}

function calculateHighlights(hit: PageHit, query: string): HighlightBox[] {
  if (!hit.pageImageDataUrl || hit.matches.length === 0) return [];
  
  if (hit.words && hit.words.length > 0) {
     const normQuery = query.trim().toLowerCase();
     const queryWords = normQuery.split(/\s+/).filter(Boolean);
     
     const boxes: HighlightBox[] = [];
     let i = 0;
     while (i < hit.words.length) {
        const w = hit.words[i];
        if (!w.text) { i++; continue; }
        const wText = w.text.toLowerCase();
        
        // 1. Full phrase is contained within this single bounding box
        if (wText.includes(normQuery)) {
            let startIdx = wText.indexOf(normQuery);
            while (startIdx !== -1) {
                const beforeText = w.text.substring(0, startIdx);
                const matchText = w.text.substring(startIdx, startIdx + normQuery.length);

                const totalW = measureTextWidth(w.text);
                const beforeW = measureTextWidth(beforeText);
                const matchW = measureTextWidth(matchText);
                const scale = totalW > 0 ? w.w / totalW : 0;

                boxes.push({
                  xFrac: w.x + (beforeW * scale),
                  yFrac: w.y,
                  wFrac: matchW * scale,
                  hFrac: w.h,
                  wordIndices: [i]
                });
                startIdx = wText.indexOf(normQuery, startIdx + normQuery.length);
            }
            i++;
            continue;
        }

        // 2. Single word substring (fallback)
        if (queryWords.length === 1 && wText.includes(queryWords[0])) {
            let startIdx = wText.indexOf(queryWords[0]);
            while (startIdx !== -1) {
                const beforeText = w.text.substring(0, startIdx);
                const matchText = w.text.substring(startIdx, startIdx + queryWords[0].length);

                const totalW = measureTextWidth(w.text);
                const beforeW = measureTextWidth(beforeText);
                const matchW = measureTextWidth(matchText);
                const scale = totalW > 0 ? w.w / totalW : 0;

                boxes.push({
                  xFrac: w.x + (beforeW * scale),
                  yFrac: w.y,
                  wFrac: matchW * scale,
                  hFrac: w.h,
                  wordIndices: [i]
                });
                startIdx = wText.indexOf(queryWords[0], startIdx + queryWords[0].length);
            }
            i++;
            continue;
        }

        // 3. Phrase spans across multiple separate word boxes
        if (queryWords.length > 1 && wText.includes(queryWords[0])) {
            let matched = true;
            let endIdx = i;
            for (let j = 1; j < queryWords.length; j++) {
               if (i+j >= hit.words.length || !hit.words[i+j].text.toLowerCase().includes(queryWords[j])) {
                   matched = false;
                   break;
               }
               endIdx = i+j;
            }
            if (matched) {
               let minX = w.x, minY = w.y, maxX = w.x + w.w, maxY = w.y + w.h;
               const indices = [];
               for (let k = i; k <= endIdx; k++) {
                   const kw = hit.words[k];
                   minX = Math.min(minX, kw.x);
                   minY = Math.min(minY, kw.y);
                   maxX = Math.max(maxX, kw.x + kw.w);
                   maxY = Math.max(maxY, kw.y + kw.h);
                   indices.push(k);
               }
               boxes.push({ 
                 xFrac: minX, 
                 yFrac: minY, 
                 wFrac: maxX - minX, 
                 hFrac: maxY - minY,
                 wordIndices: indices
               });
               i = endIdx + 1;
               continue;
            }
        }
        i++;
     }
     if (boxes.length > 0) return boxes;
  }

  // Fallback to proportional mapping for older un-processed docs
  const textLen = Math.max(1, hit.pageText.length);
  return hit.matches.slice(0, 30).map((m) => {
    const position = m.start / textLen;
    const approxWidth = Math.max(0.05, Math.min(0.3, (m.end - m.start) / textLen * 8));
    const yFrac = 0.06 + position * 0.88;
    const xBase = 0.08 + ((m.start % 53) / 53) * 0.7;
    const xFrac = Math.max(0.05, Math.min(0.95 - approxWidth, xBase));
    return { xFrac, yFrac, wFrac: approxWidth, hFrac: 0.022, wordIndices: [] };
  });
}

type ViewMode = 'list' | 'grid';

const TextSearch: React.FC = () => {
  const { toasts, push, remove } = useToasts();
  const [docs, setDocs] = useState<OcrDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState(searchTextState.query);
  const [wholeWord, setWholeWord] = useState(searchTextState.wholeWord);
  const [viewMode, setViewMode] = useState<ViewMode>((searchTextState as any).viewMode || 'list');
  const [selectedDocIds, setSelectedDocIds] = useState<Set<string>>(
    new Set((searchTextState as any).selectedDocIds || [])
  );
  const [selectedHit, setSelectedHit] = useState<PageHit | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const queryInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { searchTextState.query = query; }, [query]);
  useEffect(() => { searchTextState.wholeWord = wholeWord; }, [wholeWord]);
  useEffect(() => { (searchTextState as any).viewMode = viewMode; }, [viewMode]);
  useEffect(() => { (searchTextState as any).selectedDocIds = Array.from(selectedDocIds); }, [selectedDocIds]);

  const refresh = async () => {
    setLoading(true);
    try {
      const list = await getAllOcrDocs();
      list.sort((a, b) => b.createdAt - a.createdAt);
      setDocs(list);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    setTimeout(() => queryInputRef.current?.focus(), 100);
  }, []);

  const docPages = useMemo(() => {
    const map = new Map<string, OcrPageRecord[]>();
    for (const d of docs) {
      if (d.pageRecords && d.pageRecords.length > 0) {
        map.set(d.id, d.pageRecords);
      } else {
        const parts = d.text.split(/\n\n/);
        const records: OcrPageRecord[] = [];
        const pageCount = Math.max(1, d.pages);
        if (parts.length === pageCount) {
          for (let i = 0; i < pageCount; i++) {
            records.push({ pageNumber: i + 1, text: parts[i] || '', imageDataUrl: '' });
          }
        } else {
          records.push({ pageNumber: 1, text: d.text, imageDataUrl: '' });
        }
        map.set(d.id, records);
      }
    }
    return map;
  }, [docs]);

  const documentHits: DocumentHits[] = useMemo(() => {
    const q = query.trim();
    if (!q) return [];
    const normQuery = normalizeForSearch(q);

    const docsToSearch = selectedDocIds.size > 0
      ? docs.filter((d) => selectedDocIds.has(d.id))
      : docs;

    const results: DocumentHits[] = [];
    for (const d of docsToSearch) {
      const pages = docPages.get(d.id) || [];
      const pageHits: PageHit[] = [];
      let totalMatches = 0;
      for (const p of pages) {
        const normalizedPage = normalizeForSearch(p.text || '');
        const matches = findAllMatches(normalizedPage, normQuery, wholeWord);
        if (matches.length === 0) continue;
        totalMatches += matches.length;
        pageHits.push({
          docId: d.id,
          docName: d.name,
          language: d.language,
          pageNumber: p.pageNumber,
          pageText: normalizedPage,
          pageImageDataUrl: p.imageDataUrl || null,
          matchCount: matches.length,
          snippets: buildSnippets(normalizedPage, matches),
          matches,
          words: p.words,
          createdAt: d.createdAt,
          hasSearchablePdf: !!d.searchablePdfDataUrl,
        });
      }
      if (pageHits.length > 0) {
        results.push({ doc: d, pageHits, totalMatches });
      }
    }
    results.sort((a, b) => b.totalMatches - a.totalMatches);
    return results;
  }, [query, wholeWord, selectedDocIds, docs, docPages]);

  const flatResults: PageHit[] = useMemo(() => {
    const all: PageHit[] = [];
    for (const dh of documentHits) {
      all.push(...dh.pageHits);
    }
    all.sort((a, b) => {
      if (b.matchCount !== a.matchCount) return b.matchCount - a.matchCount;
      return b.createdAt - a.createdAt;
    });
    return all;
  }, [documentHits]);

  const totalMatches = flatResults.reduce((acc, r) => acc + r.matchCount, 0);
  const uniqueDocs = new Set(flatResults.map((r) => r.docId)).size;

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setHasSearched(true);
    if (!query.trim()) {
      push('info', 'Enter a word or phrase to search.');
      return;
    }
    if (docs.length === 0) {
      push('error', 'No OCR documents yet. Run OCR on a PDF first.');
      return;
    }
  };

  const clearSearch = () => {
    setQuery('');
    setHasSearched(false);
    setSelectedHit(null);
    setTimeout(() => queryInputRef.current?.focus(), 50);
  };

  const handlePageEdit = async (hit: PageHit, newText: string) => {
    const updated = await updateOcrDocPage(hit.docId, hit.pageNumber, newText);
    if (updated) {
      await refresh();
      push('success', `Page ${hit.pageNumber} updated in database.`);
    }
  };

  const downloadPdf = (hit: PageHit) => {
    const doc = docs.find((d) => d.id === hit.docId);
    if (!doc || !doc.searchablePdfDataUrl) {
      push('error', 'No searchable PDF on file for this document.');
      return;
    }
    downloadPdfFromDataUrl(doc.searchablePdfDataUrl, doc.name);
    push('success', 'PDF download started.');
  };

  const toggleDocSelect = (docId: string) => {
    setSelectedDocIds((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) next.delete(docId);
      else next.add(docId);
      return next;
    });
  };

  return (
    <div className="text-search">
      <style>{`
        .edit-highlight-box {
          position: absolute;
          border: 2px dashed #10b981;
          background: rgba(16, 185, 129, 0.25);
          cursor: move;
          touch-action: none;
        }
        .edit-highlight-box:hover {
          background: rgba(16, 185, 129, 0.4);
        }
        .edit-handle {
          position: absolute;
          width: 14px;
          height: 14px;
          background: #10b981;
          border: 2px solid white;
          border-radius: 50%;
          transform: translate(-50%, -50%);
          touch-action: none;
          box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }
        .edit-handle.nw { top: 0; left: 0; cursor: nwse-resize; }
        .edit-handle.ne { top: 0; left: 100%; cursor: nesw-resize; }
        .edit-handle.sw { top: 100%; left: 0; cursor: nesw-resize; }
        .edit-handle.se { top: 100%; left: 100%; cursor: nwse-resize; }
      `}</style>
      
      <Toast toasts={toasts} onRemove={remove} />

      <header className="page-head">
        <div>
          <h1>Text Search</h1>
          <p>
            Search <strong>every word</strong> across all OCR documents. Select <strong>multiple PDFs</strong> to compare matches side-by-side in grid view — each tile shows the exact page with your search term <strong>highlighted in its original position</strong>.
          </p>
        </div>
        <Button variant="ghost" icon="↻" onClick={refresh} disabled={loading}>
          Refresh Index
        </Button>
      </header>

      <Card>
        <form className="search-bar" onSubmit={onSubmit}>
          <div className="search-input-wrap">
            <span className="search-icon">🔍</span>
            <input
              ref={queryInputRef}
              className="search-input"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder='Type a word or phrase — e.g. "certified true copy"'
              autoComplete="off"
              spellCheck={false}
            />
            {query && (
              <button
                type="button"
                className="search-clear"
                onClick={clearSearch}
                title="Clear"
              >
                ×
              </button>
            )}
          </div>

          <div className="search-controls">
            <label className="search-toggle">
              <input
                type="checkbox"
                checked={wholeWord}
                onChange={(e) => setWholeWord(e.target.checked)}
              />
              <span>Whole words only</span>
            </label>

            <div className="view-mode-toggle">
              <button
                type="button"
                className={`vm-btn ${viewMode === 'list' ? 'active' : ''}`}
                onClick={() => setViewMode('list')}
              >
                ☰ List
              </button>
              <button
                type="button"
                className={`vm-btn ${viewMode === 'grid' ? 'active' : ''}`}
                onClick={() => setViewMode('grid')}
              >
                ⊞ Grid
              </button>
            </div>

            <Button variant="primary" icon="⚡" type="submit">
              Search
            </Button>
          </div>
        </form>

        {query.trim() && (
          <div className="search-summary">
            {totalMatches > 0 ? (
              <>
                <span className="summary-pill success">
                  ✓ {totalMatches} match{totalMatches === 1 ? '' : 'es'}
                </span>
                <span className="summary-pill">
                  📄 {uniqueDocs} document{uniqueDocs === 1 ? '' : 's'}
                </span>
                <span className="summary-pill">
                  📋 {flatResults.length} page{flatResults.length === 1 ? '' : 's'}
                </span>
                {selectedDocIds.size > 0 && (
                  <span className="summary-pill filter">
                    🎯 Filtered to {selectedDocIds.size} selected
                  </span>
                )}
              </>
            ) : hasSearched || query.trim().length >= 2 ? (
              <span className="summary-pill warn">
                ⓘ No matches found
              </span>
            ) : null}
          </div>
        )}
      </Card>

      {!loading && docs.length > 0 && (
        <Card
          title="📚 Documents"
          subtitle={
            selectedDocIds.size === 0
              ? `Searching across all ${docs.length} documents — tick specific ones to narrow down`
              : `Searching in ${selectedDocIds.size} selected document${selectedDocIds.size === 1 ? '' : 's'}`
          }
          right={
            <div style={{ display: 'flex', gap: 8 }}>
              <Button variant="ghost" onClick={() => setSelectedDocIds(new Set(docs.map((d) => d.id)))} disabled={selectedDocIds.size === docs.length}>
                Select all
              </Button>
              <Button variant="ghost" onClick={() => setSelectedDocIds(new Set())} disabled={selectedDocIds.size === 0}>
                Clear
              </Button>
            </div>
          }
        >
          <div className="doc-picker">
            {docs.map((d) => {
              const selected = selectedDocIds.has(d.id);
              const pages = docPages.get(d.id) || [];
              const hasImages = pages.some((p) => !!p.imageDataUrl);
              const thisDocHits = query.trim()
                ? documentHits.find((dh) => dh.doc.id === d.id)
                : null;
              return (
                <button
                  key={d.id}
                  type="button"
                  className={`doc-pick-card ${selected ? 'selected' : ''}`}
                  onClick={() => toggleDocSelect(d.id)}
                >
                  <div className={`doc-pick-check ${selected ? 'checked' : ''}`}>
                    {selected && '✓'}
                  </div>
                  <div className="doc-pick-body">
                    <div className="doc-pick-name" title={d.name}>{d.name}</div>
                    <div className="doc-pick-meta">
                      {d.pages} pg · {d.language}
                      {hasImages && <span className="doc-pick-img"> · 🖼</span>}
                      {d.searchablePdfDataUrl && <span className="doc-pick-pdf"> · 📄</span>}
                    </div>
                    {thisDocHits && (
                      <div className="doc-pick-hits">
                        🎯 {thisDocHits.totalMatches} match{thisDocHits.totalMatches === 1 ? '' : 'es'} in {thisDocHits.pageHits.length} page{thisDocHits.pageHits.length === 1 ? '' : 's'}
                      </div>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </Card>
      )}

      {!loading && query.trim() && viewMode === 'grid' && documentHits.length > 0 && (
        <div className="grid-results">
          <div className="grid-results-title">
            📊 Grid Comparison
          </div>
          <div className="pdf-grid">
            {documentHits.map((dh) => (
              <PdfGridTile
                key={dh.doc.id}
                documentHits={dh}
                query={query.trim()}
                wholeWord={wholeWord}
                onOpenPage={(hit) => setSelectedHit(hit)}
                onDownloadPdf={() => downloadPdf(dh.pageHits[0])}
              />
            ))}
          </div>
        </div>
      )}

      {!loading && query.trim() && viewMode === 'list' && flatResults.length > 0 && (
        <div className="results-list">
          {flatResults.map((hit, i) => (
            <ResultCard
              key={`${hit.docId}-${hit.pageNumber}-${i}`}
              hit={hit}
              onOpen={() => setSelectedHit(hit)}
            />
          ))}
        </div>
      )}

      {selectedHit && (
        <PageViewerModal
          hit={selectedHit}
          query={query.trim()}
          wholeWord={wholeWord}
          onClose={() => setSelectedHit(null)}
          onDownloadPdf={() => downloadPdf(selectedHit)}
          onEditSave={(newText) => handlePageEdit(selectedHit, newText)}
          onRefreshRequired={refresh}
        />
      )}
    </div>
  );
};

const PdfGridTile: React.FC<any> = ({ documentHits, query, onOpenPage, onDownloadPdf }) => {
  const { doc, pageHits, totalMatches } = documentHits;
  const [activePageIdx, setActivePageIdx] = useState(0);
  const activeHit = pageHits[activePageIdx];

  return (
    <div className="pdf-tile">
      <div className="pdf-tile-head">
        <div className="pdf-tile-title-wrap">
          <div className="pdf-tile-icon">📄</div>
          <div className="pdf-tile-title-info">
            <div className="pdf-tile-title" title={doc.name}>{doc.name}</div>
            <div className="pdf-tile-meta">
              <span className="tile-match-badge">🎯 {totalMatches} match</span>
            </div>
          </div>
        </div>
      </div>
      {activeHit && (
        <>
          <HighlightedPagePreview hit={activeHit} query={query} onClick={() => onOpenPage(activeHit)} />
          <button className="pdf-tile-open-btn" onClick={() => onOpenPage(activeHit)}>
            🔍 Open page {activeHit.pageNumber}
          </button>
        </>
      )}
    </div>
  );
};

const HighlightedPagePreview: React.FC<any> = ({ hit, query, onClick }) => {
  const highlights = useMemo(() => calculateHighlights(hit, query), [hit, query]);
  if (!hit.pageImageDataUrl) return <div className="tile-preview" onClick={onClick}>No image</div>;

  return (
    <div className="tile-preview" onClick={onClick} style={{ position: 'relative', display: 'inline-block', width: 'fit-content', margin: '0 auto' }}>
      <img src={hit.pageImageDataUrl} alt={`page`} className="tile-preview-img" style={{ display: 'block', maxWidth: '100%', height: 'auto' }} />
      {highlights.map((h: any, i: number) => (
        <div key={i} className="tile-highlight-box" style={{ left: `${h.xFrac * 100}%`, top: `${h.yFrac * 100}%`, width: `${h.wFrac * 100}%`, height: `${h.hFrac * 100}%` }} />
      ))}
    </div>
  );
};

const ResultCard: React.FC<any> = ({ hit, onOpen }) => {
  return (
    <div className="result-card" onClick={onOpen}>
      <div className="result-card-main">
        {hit.pageImageDataUrl && (
          <div className="result-thumb"><img src={hit.pageImageDataUrl} alt={`page`} /><span className="result-thumb-page">p.{hit.pageNumber}</span></div>
        )}
        <div className="result-card-body">
          <div className="result-head">
            <div className="result-title-wrap"><div className="result-doc-icon">📄</div><div className="result-title-info"><div className="result-doc-name">{hit.docName}</div></div></div>
            <button className="result-open" onClick={(e) => { e.stopPropagation(); onOpen(); }}>Open page →</button>
          </div>
          <div className="result-snippets">
            {hit.snippets.map((s: any, i: number) => (
              <div key={i} className="snippet"><span className="snippet-text">{s.before}<mark className="snippet-match">{s.match}</mark>{s.after}</span></div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const PageViewerModal: React.FC<{
  hit: PageHit;
  query: string;
  wholeWord: boolean;
  onClose: () => void;
  onDownloadPdf: () => void;
  onEditSave: (newText: string) => Promise<void>;
  onRefreshRequired: () => Promise<void>;
}> = ({ hit, query, wholeWord, onClose, onDownloadPdf, onEditSave, onRefreshRequired }) => {
  const [currentMatch, setCurrentMatch] = useState(0);
  const [view, setView] = useState<'image' | 'text' | 'split'>(hit.pageImageDataUrl ? 'split' : 'text');
  const [editing, setEditing] = useState(false);
  const [editDraft, setEditDraft] = useState(hit.pageText);
  const [saving, setSaving] = useState(false);
  const [zoom, setZoom] = useState(1);
  const textRef = useRef<HTMLDivElement>(null);
  const imageFrameRef = useRef<HTMLDivElement>(null);

  // === Highlight Adjustment State ===
  const imageHighlights = useMemo(() => calculateHighlights(hit, query), [hit, query]);
  const [isAdjusting, setIsAdjusting] = useState(false);
  const [adjustedBoxes, setAdjustedBoxes] = useState<HighlightBox[] | null>(null);
  
  // Drag interaction refs & state
  const interactionRef = useRef<{
    idx: number;
    type: string;
    startX: number;
    startY: number;
    startBox: HighlightBox;
  } | null>(null);

  useEffect(() => {
    setEditDraft(hit.pageText);
    setEditing(false);
    setCurrentMatch(0);
    setIsAdjusting(false);
    setAdjustedBoxes(null);
  }, [hit.pageText, hit.docId, hit.pageNumber]);

  useEffect(() => {
    if (isAdjusting && !adjustedBoxes) {
      setAdjustedBoxes(JSON.parse(JSON.stringify(imageHighlights)));
    }
  }, [isAdjusting, adjustedBoxes, imageHighlights]);

  // Global mouse handlers for box resizing
  useEffect(() => {
    const handlePointerMove = (e: PointerEvent) => {
      const int = interactionRef.current;
      if (!int || !isAdjusting || !adjustedBoxes || !imageFrameRef.current) return;
      
      const rect = imageFrameRef.current.getBoundingClientRect();
      // Adjust pointer delta by zoom scale
      const dxFrac = (e.clientX - int.startX) / (rect.width);
      const dyFrac = (e.clientY - int.startY) / (rect.height);

      setAdjustedBoxes(prev => {
        if (!prev) return prev;
        const newBoxes = [...prev];
        const box = { ...int.startBox };

        if (int.type === 'move') {
          box.xFrac += dxFrac;
          box.yFrac += dyFrac;
        } else {
          if (int.type.includes('n')) { box.yFrac += dyFrac; box.hFrac -= dyFrac; }
          if (int.type.includes('s')) { box.hFrac += dyFrac; }
          if (int.type.includes('w')) { box.xFrac += dxFrac; box.wFrac -= dxFrac; }
          if (int.type.includes('e')) { box.wFrac += dxFrac; }
        }

        // Enforce minimum dimension boundary
        if (box.wFrac < 0.005) box.wFrac = 0.005;
        if (box.hFrac < 0.005) box.hFrac = 0.005;

        newBoxes[int.idx] = box;
        return newBoxes;
      });
    };

    const handlePointerUp = () => { interactionRef.current = null; };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [isAdjusting, adjustedBoxes]);

  const startInteraction = (e: React.PointerEvent, idx: number, type: string) => {
    if (!adjustedBoxes) return;
    e.preventDefault();
    e.stopPropagation();
    interactionRef.current = {
      idx,
      type,
      startX: e.clientX,
      startY: e.clientY,
      startBox: { ...adjustedBoxes[idx] }
    };
  };

  const saveHighlights = async () => {
    if (!adjustedBoxes || !hit.words) return;
    setSaving(true);
    try {
      const newWords = JSON.parse(JSON.stringify(hit.words));
      
      for (let i = 0; i < adjustedBoxes.length; i++) {
        const box = adjustedBoxes[i];
        const origBox = imageHighlights[i];
        
        // Skip if box was not changed
        if (Math.abs(box.xFrac - origBox.xFrac) < 0.0001 && Math.abs(box.wFrac - origBox.wFrac) < 0.0001) {
          continue;
        }

        // Apply proportional visual transformation to underlying word blocks
        const wRatio = box.wFrac / origBox.wFrac;
        const hRatio = box.hFrac / origBox.hFrac;

        for (const wordIdx of box.wordIndices) {
            const word = newWords[wordIdx];
            const relX = word.x - origBox.xFrac;
            const relY = word.y - origBox.yFrac;
            word.x = box.xFrac + (relX * wRatio);
            word.y = box.yFrac + (relY * hRatio);
            word.w = word.w * wRatio;
            word.h = word.h * hRatio;
        }
      }

      await updateOcrDocPageWords(hit.docId, hit.pageNumber, newWords);
      push('success', 'Highlight bounds permanently saved! PDF export will now use these bounds.');
      setIsAdjusting(false);
      await onRefreshRequired();
    } finally {
      setSaving(false);
    }
  };
  
  const activeBoxes = isAdjusting && adjustedBoxes ? adjustedBoxes : imageHighlights;

  return (
    <div className="pv-backdrop" onClick={onClose}>
      <div className="pv-modal" onClick={(e) => e.stopPropagation()}>
        <div className="pv-head">
          <div className="pv-head-info">
            <div className="pv-title">📄 <div><div className="pv-doc-name">{hit.docName}</div><div className="pv-doc-sub">Page {hit.pageNumber} · {hit.matchCount} matches</div></div></div>
          </div>
          <div className="pv-head-actions">
            {!isAdjusting && (
              <Button variant="ghost" icon="🛠️" onClick={() => setIsAdjusting(true)} disabled={imageHighlights.length === 0}>
                Adjust Highlight
              </Button>
            )}
            {isAdjusting && (
              <>
                <Button variant="ghost" onClick={() => { setIsAdjusting(false); setAdjustedBoxes(null); }}>Cancel</Button>
                <Button variant="primary" icon="💾" onClick={saveHighlights} loading={saving}>Save Alignment</Button>
              </>
            )}
            <Button variant="secondary" icon="⬇" onClick={onDownloadPdf}>PDF</Button>
            <button className="pv-close" onClick={onClose}>×</button>
          </div>
        </div>

        <div className={`pv-body pv-view-${view}`}>
          {hit.pageImageDataUrl && (
            <div className="pv-image-pane">
              <div className="pv-image-toolbar">
                <button className="pv-zoom-btn" onClick={() => setZoom((z) => Math.max(0.3, z - 0.2))}>−</button>
                <span className="pv-zoom-label">{(zoom * 100).toFixed(0)}%</span>
                <button className="pv-zoom-btn" onClick={() => setZoom((z) => Math.min(4, z + 0.2))}>+</button>
              </div>
              <div className="pv-image-scroll" style={{ textAlign: 'center' }}>
                <div
                  ref={imageFrameRef}
                  className="pv-image-frame"
                  style={{ transform: `scale(${zoom})`, transformOrigin: 'top center', position: 'relative', display: 'inline-block', margin: '0 auto' }}
                >
                  <img src={hit.pageImageDataUrl} alt={`page`} style={{ display: 'block', maxWidth: '100%', height: 'auto' }} />
                  
                  {activeBoxes.map((h, i) => (
                    <div
                      key={i}
                      className={isAdjusting ? 'edit-highlight-box' : `pv-image-highlight ${i === currentMatch ? 'active' : ''}`}
                      onPointerDown={isAdjusting ? (e) => startInteraction(e, i, 'move') : undefined}
                      style={{
                        left: `${h.xFrac * 100}%`,
                        top: `${h.yFrac * 100}%`,
                        width: `${h.wFrac * 100}%`,
                        height: `${h.hFrac * 100}%`,
                      }}
                    >
                      {!isAdjusting && <span className="pv-image-highlight-num">{i + 1}</span>}
                      
                      {isAdjusting && (
                        <>
                          <div className="edit-handle nw" onPointerDown={(e) => startInteraction(e, i, 'nw')} />
                          <div className="edit-handle ne" onPointerDown={(e) => startInteraction(e, i, 'ne')} />
                          <div className="edit-handle sw" onPointerDown={(e) => startInteraction(e, i, 'sw')} />
                          <div className="edit-handle se" onPointerDown={(e) => startInteraction(e, i, 'se')} />
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TextSearch;