
// Online knowledge assist — Multi-Source Web Intelligence
// ========================================================
// Queries MULTIPLE public web sources in parallel to build the richest
// possible context for a label. No API keys required — all endpoints are
// public and CORS-friendly.
//
// Sources (all free, all public):
//   1. Wikipedia REST Summary API     — encyclopedia entries
//   2. Wikipedia OpenSearch API       — fuzzy title search
//   3. Wikidata Search API            — structured entity data
//   4. DuckDuckGo Instant Answer API  — general web Q&A (abstract, entity type)
//   5. DBpedia Lookup API             — linked-data entity resolution
//   6. Open Library Search API        — book/publication context
//   7. MusicBrainz API                — music/band context
//
// The results are fused into a single, coherent reasoning paragraph with
// cross-referenced sources. Fails gracefully to a local heuristic offline.

import type { SampleCategory } from './database';

export interface KnowledgeResult {
  reasoning: string;
  suggestedLabel?: string;
  suggestedCategory?: SampleCategory;
  confidence?: number;
  source: string;
  fetchedAt: number;
  // NEW — rich multi-source payload
  sources?: KnowledgeSource[];
  entityType?: string;
  relatedLinks?: { title: string; url: string }[];
}

export interface KnowledgeSource {
  name: string;           // e.g. "Wikipedia", "Wikidata", "DuckDuckGo"
  title?: string;         // resolved title of the entity
  excerpt: string;        // short summary/excerpt
  url?: string;           // canonical URL when available
  confidence: number;     // 0..1 heuristic confidence of this source
}

const TIMEOUT_MS = 7000;

/* ============================================================
 * Source endpoints
 * ============================================================ */

const WIKI_SUMMARY = 'https://en.wikipedia.org/api/rest_v1/page/summary/';
const WIKI_SEARCH = 'https://en.wikipedia.org/w/api.php';
const WIKIDATA_SEARCH = 'https://www.wikidata.org/w/api.php';
const DDG_INSTANT = 'https://api.duckduckgo.com/';
const DBPEDIA_LOOKUP = 'https://lookup.dbpedia.org/api/search';
const OPENLIBRARY_SEARCH = 'https://openlibrary.org/search.json';
const MUSICBRAINZ_SEARCH = 'https://musicbrainz.org/ws/2/artist/';

/* ============================================================
 * Public API
 * ============================================================ */

/**
 * Query MULTIPLE web sources in parallel and fuse the results.
 * This is the heart of the knowledge-assist system.
 */
export async function lookupKnowledge(
  label: string,
  category: SampleCategory,
  userComment?: string
): Promise<KnowledgeResult> {
  const trimmed = (label || '').trim();
  if (!trimmed) {
    return localHeuristic(label, category, userComment);
  }

  // Fire all sources in parallel — each one is independently fail-safe.
  const [
    wiki,
    wikidata,
    ddg,
    dbpedia,
    openLibrary,
    musicBrainz,
  ] = await Promise.all([
    safeCall(() => queryWikipedia(trimmed)),
    safeCall(() => queryWikidata(trimmed)),
    safeCall(() => queryDuckDuckGo(trimmed)),
    safeCall(() => queryDBpedia(trimmed)),
    safeCall(() => queryOpenLibrary(trimmed)),
    safeCall(() => queryMusicBrainz(trimmed)),
  ]);

  const sources: KnowledgeSource[] = [
    wiki, wikidata, ddg, dbpedia, openLibrary, musicBrainz,
  ].filter((s): s is KnowledgeSource => !!s);

  if (sources.length === 0) {
    // Everything failed — network error or offline. Use local heuristic but
    // keep the label the user gave.
    return localHeuristic(trimmed, category, userComment);
  }

  // Rank sources by confidence.
  sources.sort((a, b) => b.confidence - a.confidence);

  // Fuse into a coherent reasoning block.
  const reasoning = fuseReasoning(sources, category, userComment);

  // Determine the best suggested label (prefer Wikipedia/Wikidata titles).
  const topTitled = sources.find((s) => s.title && s.title.trim().length > 0);
  const suggestedLabel = topTitled?.title || trimmed;

  // Entity-type hint drawn from Wikidata/DBpedia/DDG.
  const entityType =
    (wikidata as any)?.__entityType ||
    (dbpedia as any)?.__entityType ||
    (ddg as any)?.__entityType ||
    undefined;

  const relatedLinks = sources
    .filter((s) => s.url)
    .map((s) => ({ title: `${s.name}: ${s.title || trimmed}`, url: s.url! }));

  // Aggregate confidence: mean of top-3 sources, capped at 0.92.
  const top = sources.slice(0, 3);
  const avgConf = top.reduce((a, b) => a + b.confidence, 0) / top.length;
  const confidence = Math.min(0.92, avgConf);

  const sourceLabel =
    sources.length === 1
      ? `${sources[0].name} · ${sources[0].title || trimmed}`
      : `Multi-source (${sources.length}): ${sources.map((s) => s.name).join(' + ')}`;

  return {
    reasoning,
    suggestedLabel,
    suggestedCategory: category,
    confidence,
    source: sourceLabel,
    fetchedAt: Date.now(),
    sources,
    entityType,
    relatedLinks,
  };
}

/* ============================================================
 * Fusion — merges multi-source excerpts into one paragraph
 * ============================================================ */

function fuseReasoning(
  sources: KnowledgeSource[],
  category: SampleCategory,
  userComment?: string
): string {
  const parts: string[] = [];

  // Primary excerpt from the highest-confidence source.
  const primary = sources[0];
  parts.push(`${primary.excerpt}`);

  // Cross-reference corroborating sources that add NEW information.
  const used = new Set<string>();
  used.add(normalize(primary.excerpt));

  for (let i = 1; i < sources.length; i++) {
    const s = sources[i];
    const norm = normalize(s.excerpt);
    // Only add if it contributes meaningfully new content (not a near-duplicate).
    if (norm.length < 30) continue;
    let overlapping = false;
    for (const u of used) {
      if (jaccard(norm, u) > 0.55) {
        overlapping = true;
        break;
      }
    }
    if (!overlapping) {
      parts.push(`According to ${s.name}: ${s.excerpt}`);
      used.add(norm);
      if (parts.length >= 3) break; // cap at 3 fused paragraphs
    }
  }

  // Category hint
  const categoryHint =
    category === 'logo'
      ? `This context will be stored with the logo annotation so the system can learn the brand's identity alongside its visual signature.`
      : category === 'signature'
      ? `This context will be stored with the signature annotation for provenance tracking.`
      : `This context will be stored with the stamp annotation for historical/organizational reference.`;
  parts.push(categoryHint);

  if (userComment && userComment.trim()) {
    parts.push(`User note: "${userComment.trim()}"`);
  }

  return parts.join(' ');
}

/* ============================================================
 * Individual source queries
 * ============================================================ */

async function queryWikipedia(q: string): Promise<KnowledgeSource | null> {
  // Try direct summary first
  try {
    const r = await fetchJson(WIKI_SUMMARY + encodeURIComponent(q));
    if (r?.extract && typeof r.extract === 'string' && !r.type?.includes('disambiguation')) {
      return {
        name: 'Wikipedia',
        title: r.title || q,
        excerpt: shorten(r.extract, 380),
        url: r.content_urls?.desktop?.page,
        confidence: 0.85,
      };
    }
  } catch { /* fall through */ }

  // Fallback: opensearch
  try {
    const params = new URLSearchParams({
      action: 'opensearch',
      search: q,
      limit: '1',
      namespace: '0',
      format: 'json',
      origin: '*',
    });
    const data = await fetchJson(`${WIKI_SEARCH}?${params.toString()}`);
    if (Array.isArray(data) && data.length >= 4) {
      const titles: string[] = data[1] || [];
      const descs: string[] = data[2] || [];
      const urls: string[] = data[3] || [];
      if (titles.length && descs[0]) {
        return {
          name: 'Wikipedia',
          title: titles[0],
          excerpt: shorten(descs[0], 380),
          url: urls[0],
          confidence: 0.6,
        };
      }
    }
  } catch { /* noop */ }

  return null;
}

async function queryWikidata(q: string): Promise<KnowledgeSource | null> {
  try {
    const params = new URLSearchParams({
      action: 'wbsearchentities',
      search: q,
      language: 'en',
      format: 'json',
      limit: '1',
      origin: '*',
    });
    const data = await fetchJson(`${WIKIDATA_SEARCH}?${params.toString()}`);
    const first = data?.search?.[0];
    if (first) {
      const desc = first.description || 'Known entity in Wikidata.';
      const res: KnowledgeSource & { __entityType?: string } = {
        name: 'Wikidata',
        title: first.label || q,
        excerpt: shorten(`${first.label || q} — ${desc}.`, 280),
        url: first.concepturi || (first.id ? `https://www.wikidata.org/wiki/${first.id}` : undefined),
        confidence: 0.7,
      };
      res.__entityType = first.description;
      return res;
    }
  } catch { /* noop */ }
  return null;
}

async function queryDuckDuckGo(q: string): Promise<KnowledgeSource | null> {
  try {
    const params = new URLSearchParams({
      q,
      format: 'json',
      no_html: '1',
      skip_disambig: '1',
      t: 'ocr-ai-studio',
    });
    const data = await fetchJson(`${DDG_INSTANT}?${params.toString()}`);
    if (data?.AbstractText && typeof data.AbstractText === 'string' && data.AbstractText.length > 10) {
      const res: KnowledgeSource & { __entityType?: string } = {
        name: 'DuckDuckGo',
        title: data.Heading || q,
        excerpt: shorten(data.AbstractText, 380),
        url: data.AbstractURL || undefined,
        confidence: 0.72,
      };
      if (data.Entity) res.__entityType = data.Entity;
      return res;
    }
    // RelatedTopics fallback
    if (Array.isArray(data?.RelatedTopics) && data.RelatedTopics.length > 0) {
      const top = data.RelatedTopics.find(
        (t: any) => t?.Text && typeof t.Text === 'string' && t.Text.length > 20
      );
      if (top) {
        return {
          name: 'DuckDuckGo',
          title: q,
          excerpt: shorten(top.Text, 280),
          url: top.FirstURL,
          confidence: 0.5,
        };
      }
    }
  } catch { /* noop */ }
  return null;
}

async function queryDBpedia(q: string): Promise<KnowledgeSource | null> {
  try {
    const params = new URLSearchParams({
      query: q,
      format: 'json',
      maxResults: '1',
    });
    const data = await fetchJson(`${DBPEDIA_LOOKUP}?${params.toString()}`, {
      Accept: 'application/json',
    });
    const first = data?.docs?.[0] || data?.results?.[0];
    if (first) {
      const comment =
        (Array.isArray(first.comment) ? first.comment[0] : first.comment) ||
        (Array.isArray(first.description) ? first.description[0] : first.description) ||
        '';
      const label =
        (Array.isArray(first.label) ? first.label[0] : first.label) || q;
      const url =
        (Array.isArray(first.resource) ? first.resource[0] : first.resource) ||
        first.uri;
      const type =
        (Array.isArray(first.typeName) ? first.typeName[0] : first.typeName) ||
        (Array.isArray(first.type) ? first.type[0] : first.type);
      if (comment && comment.length > 20) {
        const stripped = comment.replace(/<[^>]+>/g, '').trim();
        const res: KnowledgeSource & { __entityType?: string } = {
          name: 'DBpedia',
          title: label,
          excerpt: shorten(stripped, 340),
          url,
          confidence: 0.6,
        };
        if (type) res.__entityType = String(type);
        return res;
      }
    }
  } catch { /* noop */ }
  return null;
}

async function queryOpenLibrary(q: string): Promise<KnowledgeSource | null> {
  try {
    const params = new URLSearchParams({
      q,
      limit: '1',
    });
    const data = await fetchJson(`${OPENLIBRARY_SEARCH}?${params.toString()}`);
    const doc = data?.docs?.[0];
    if (doc && doc.title) {
      const authors = Array.isArray(doc.author_name) ? doc.author_name.slice(0, 3).join(', ') : '';
      const year = doc.first_publish_year ? ` (${doc.first_publish_year})` : '';
      const excerpt = `"${doc.title}"${year}${authors ? ` by ${authors}` : ''} — found in Open Library's catalog of books and publications.`;
      return {
        name: 'Open Library',
        title: doc.title,
        excerpt: shorten(excerpt, 260),
        url: doc.key ? `https://openlibrary.org${doc.key}` : undefined,
        confidence: 0.45,
      };
    }
  } catch { /* noop */ }
  return null;
}

async function queryMusicBrainz(q: string): Promise<KnowledgeSource | null> {
  try {
    const params = new URLSearchParams({
      query: q,
      fmt: 'json',
      limit: '1',
    });
    const data = await fetchJson(`${MUSICBRAINZ_SEARCH}?${params.toString()}`);
    const artist = data?.artists?.[0];
    if (artist && artist.name && artist.score >= 90) {
      const type = artist.type ? ` (${artist.type})` : '';
      const country = artist.country ? ` from ${artist.country}` : '';
      const tags = Array.isArray(artist.tags)
        ? artist.tags.slice(0, 3).map((t: any) => t.name).join(', ')
        : '';
      const excerpt = `${artist.name}${type}${country}${tags ? ` — associated with: ${tags}` : ''}. Indexed by MusicBrainz.`;
      return {
        name: 'MusicBrainz',
        title: artist.name,
        excerpt: shorten(excerpt, 260),
        url: artist.id ? `https://musicbrainz.org/artist/${artist.id}` : undefined,
        confidence: 0.5,
      };
    }
  } catch { /* noop */ }
  return null;
}

/* ============================================================
 * Utilities
 * ============================================================ */

async function fetchJson(
  url: string,
  extraHeaders: Record<string, string> = {}
): Promise<any> {
  const signal = typeof AbortSignal !== 'undefined' && (AbortSignal as any).timeout
    ? (AbortSignal as any).timeout(TIMEOUT_MS)
    : undefined;
  const res = await fetch(url, {
    headers: { Accept: 'application/json', ...extraHeaders },
    signal,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function safeCall<T>(fn: () => Promise<T | null>): Promise<T | null> {
  try {
    return await fn();
  } catch {
    return null;
  }
}

function shorten(text: string, max: number): string {
  const t = text.replace(/\s+/g, ' ').trim();
  if (t.length <= max) return t;
  const cut = t.slice(0, max);
  const lastDot = cut.lastIndexOf('.');
  if (lastDot > max * 0.6) return cut.slice(0, lastDot + 1);
  return cut + '…';
}

function normalize(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
}

function jaccard(a: string, b: string): number {
  const as = new Set(a.split(' ').filter((w) => w.length > 3));
  const bs = new Set(b.split(' ').filter((w) => w.length > 3));
  if (as.size === 0 || bs.size === 0) return 0;
  let inter = 0;
  for (const w of as) if (bs.has(w)) inter++;
  return inter / (as.size + bs.size - inter);
}

/* ============================================================
 * Local fallback
 * ============================================================ */

function localHeuristic(
  label: string,
  category: SampleCategory,
  userComment?: string
): KnowledgeResult {
  const hints: string[] = [];
  if (category === 'logo') {
    hints.push(
      'Logos typically show a brand mark, wordmark, or emblem with consistent colors and proportions.'
    );
  } else if (category === 'signature') {
    hints.push(
      'Signatures are handwritten strokes, often stylized, with variable baseline and looping curves unique to a signer.'
    );
  } else if (category === 'stamp') {
    hints.push(
      'Stamps are rubber-inked impressions — usually circular/rectangular with bold outlines and monochrome ink (blue, red, or black).'
    );
  }

  if (userComment && userComment.trim()) {
    hints.push(`User noted: "${userComment.trim()}".`);
  }

  hints.push(
    `Label "${label}" will be stored as ground-truth so future scans of visually similar elements are classified correctly.`
  );
  hints.push(
    'Note: online knowledge sources were unreachable — used offline heuristic.'
  );

  return {
    reasoning: hints.join(' '),
    suggestedLabel: label,
    suggestedCategory: category,
    confidence: 0.4,
    source: 'Local heuristic (offline)',
    fetchedAt: Date.now(),
    sources: [],
  };
}
