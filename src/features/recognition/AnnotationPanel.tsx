import React, { useEffect, useState } from 'react';
import './AnnotationPanel.css';
import Button from '../../components/Button/Button';
import type { RegionMatch } from '../../services/recognitionService';
import type { PredictImageState } from '../../state/workspaceState';
import { getUniqueLabels, type SampleCategory } from '../../services/database';
import { cropRegion, addAnnotation, getAnnotationsForSource, type RegionAnnotation, type AnnotationVerdict } from '../../services/annotationsService';

interface Props {
  item: PredictImageState;
  regionIndex: number;
  onClose: () => void;
  onPromoteToTraining: (annotation: RegionAnnotation) => Promise<void>;
  onRejected: (annotation: RegionAnnotation) => Promise<void>;
}

const CAT_META: Record<SampleCategory, { icon: string; color: string; label: string }> = {
  word: { icon: '🔤', color: '#06b6d4', label: 'Word / Text' },
  logo: { icon: '🏷️', color: '#8b5cf6', label: 'Logo' },
  signature: { icon: '✍️', color: '#ec4899', label: 'Signature' },
  stamp: { icon: '🔖', color: '#f59e0b', label: 'Stamp' },
};

const CATEGORY_ORDER: SampleCategory[] = ['word', 'logo', 'signature', 'stamp'];

const AnnotationPanel: React.FC<Props> = ({
  item,
  regionIndex,
  onClose,
  onPromoteToTraining,
  onRejected,
}) => {
  const region: RegionMatch | undefined = item.prediction?.regions?.[regionIndex];

  const [cropUrl, setCropUrl] = useState<string | null>(null);
  const [comment, setComment] = useState('');
  const [verdict, setVerdict] = useState<AnnotationVerdict>('confirmed');
  const [correctedLabel, setCorrectedLabel] = useState('');
  const [correctedCategory, setCorrectedCategory] = useState<SampleCategory>('word');
  const [knownLabels, setKnownLabels] = useState<{ category: SampleCategory; label: string; count: number }[]>([]);
  const [existingNotes, setExistingNotes] = useState<RegionAnnotation[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!region) return;
    (async () => {
      try {
        const url = await cropRegion(item.dataUrl, region);
        setCropUrl(url);
      } catch (e) {
        console.warn('crop failed', e);
      }
    })();
  }, [item.dataUrl, region]);

  useEffect(() => {
    (async () => {
      const labels = await getUniqueLabels();
      setKnownLabels(labels);
      const notes = await getAnnotationsForSource(item.id);
      setExistingNotes(notes);
    })();
  }, [item.id]);

  useEffect(() => {
    if (region) {
      setCorrectedLabel(region.label);
      setCorrectedCategory(region.category as SampleCategory);
    }
  }, [regionIndex, region?.label, region?.category]);

  if (!region) return null;

  const meta = CAT_META[region.category as SampleCategory] || CAT_META.logo;

  const save = async () => {
    if (!region) return;
    setSaving(true);
    try {
      const ann: RegionAnnotation = {
        id: `ann_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        sourceImageId: item.id,
        sourceImageName: item.sourceName,
        sourceImageDataUrl: item.dataUrl,
        region,
        regionCropDataUrl: cropUrl,
        originalLabel: region.label,
        originalCategory: region.category as SampleCategory,
        originalConfidence: region.confidence,
        verdict,
        correctedLabel: verdict === 'relabeled' ? correctedLabel.trim() : undefined,
        correctedCategory: verdict === 'relabeled' ? correctedCategory : undefined,
        comment: comment.trim(),
      };
      await addAnnotation(ann);

      if (verdict === 'confirmed' || verdict === 'relabeled') {
        await onPromoteToTraining(ann);
      } else if (verdict === 'rejected') {
        await onRejected(ann);
      }

      onClose();
    } finally {
      setSaving(false);
    }
  };

  const suggestionsFiltered = knownLabels.filter((k) =>
    verdict === 'relabeled'
      ? k.category === correctedCategory
      : true
  );

  return (
    <div className="ann-backdrop" onClick={onClose}>
      <div className="ann-panel" onClick={(e) => e.stopPropagation()}>
        <div className="ann-head">
          <div className="ann-head-left">
            <div className="ann-head-badge" style={{ background: meta.color }}>
              {meta.icon}
            </div>
            <div>
              <div className="ann-title">Annotate Region #{regionIndex + 1}</div>
              <div className="ann-sub">
                Original: <strong>{region.label}</strong> ({region.category}) ·{' '}
                {(region.confidence * 100).toFixed(0)}% confidence
                {region.textMatch && (
                  <span className="ann-sub-text-match"> · 🔤 OCR: "{region.textMatch.ocrText.slice(0, 30)}{region.textMatch.ocrText.length > 30 ? '…' : ''}"</span>
                )}
              </div>
            </div>
          </div>
          <button className="ann-close" onClick={onClose}>×</button>
        </div>

        <div className="ann-body">
          <div className="ann-col-left">
            <div className="ann-section-title">Region Preview</div>
            <div className="ann-crop-box">
              {cropUrl ? (
                <img src={cropUrl} alt="region crop" />
              ) : (
                <div className="ann-crop-loading">
                  <div className="mini-spinner-lg" />
                  Extracting crop…
                </div>
              )}
            </div>

            <div className="ann-context-box">
              <div className="ann-section-title">Context</div>
              <div className="ann-context-img-wrap">
                <img src={item.dataUrl} alt={item.sourceName} className="ann-context-img" />
                <div
                  className="ann-context-marker"
                  style={{
                    left: `${region.x * 100}%`,
                    top: `${region.y * 100}%`,
                    width: `${region.width * 100}%`,
                    height: `${region.height * 100}%`,
                    borderColor: meta.color,
                  }}
                />
              </div>
            </div>

            {existingNotes.length > 0 && (
              <div className="ann-existing">
                <div className="ann-section-title">
                  📝 Existing Notes on This Image ({existingNotes.length})
                </div>
                <div className="ann-existing-list">
                  {existingNotes.slice(0, 4).map((n) => (
                    <div key={n.id} className="ann-existing-item">
                      <div className="ann-existing-verdict" data-v={n.verdict}>
                        {n.verdict}
                      </div>
                      <div className="ann-existing-body">
                        <strong>{n.correctedLabel || n.originalLabel}</strong>
                        {n.comment && <> — {n.comment}</>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="ann-col-right">
            <div className="ann-section-title">Your Verdict</div>
            <div className="verdict-group">
              <VerdictButton
                active={verdict === 'confirmed'}
                onClick={() => setVerdict('confirmed')}
                icon="✓"
                color="#10b981"
                title="Confirm"
                desc="This is correct — promote to training"
              />
              <VerdictButton
                active={verdict === 'relabeled'}
                onClick={() => setVerdict('relabeled')}
                icon="✎"
                color="#f59e0b"
                title="Relabel"
                desc="Wrong label — I'll provide the correct one"
              />
              <VerdictButton
                active={verdict === 'rejected'}
                onClick={() => setVerdict('rejected')}
                icon="✕"
                color="#ef4444"
                title="Reject"
                desc="Not a real match — remember as negative"
              />
              <VerdictButton
                active={verdict === 'note_only'}
                onClick={() => setVerdict('note_only')}
                icon="💬"
                color="#6366f1"
                title="Note"
                desc="Just leave a comment, don't change training"
              />
            </div>

            {verdict === 'relabeled' && (
              <div className="ann-correction">
                <label className="field-label">Correct Category</label>
                <div className="cat-multi">
                  {CATEGORY_ORDER.map((c) => {
                    const m = CAT_META[c];
                    const active = correctedCategory === c;
                    const isPriority = c === 'word';
                    return (
                      <button
                        key={c}
                        type="button"
                        className={`cat-multi-pill ${active ? 'active' : ''} ${isPriority ? 'priority' : ''}`}
                        style={{ ['--pill' as any]: m.color }}
                        onClick={() => setCorrectedCategory(c)}
                      >
                        <span className="pill-icon">{m.icon}</span>
                        <span>{m.label}</span>
                      </button>
                    );
                  })}
                </div>

                <label className="field-label" style={{ marginTop: 14 }}>
                  Correct Label
                  {correctedCategory === 'word' && (
                    <span className="field-hint"> — the exact text phrase</span>
                  )}
                </label>
                <input
                  className="ann-input"
                  value={correctedLabel}
                  onChange={(e) => setCorrectedLabel(e.target.value)}
                  placeholder={correctedCategory === 'word' ? 'e.g. PAID, INVOICE' : 'e.g. Acme Corp'}
                  list="corr-labels"
                />
                <datalist id="corr-labels">
                  {suggestionsFiltered.map((k) => (
                    <option key={`${k.category}::${k.label}`} value={k.label} />
                  ))}
                </datalist>

                {suggestionsFiltered.length > 0 && (
                  <div className="ann-chip-row">
                    <span className="ann-chip-title">Or pick an existing label:</span>
                    {suggestionsFiltered.slice(0, 8).map((k) => (
                      <button
                        key={`${k.category}::${k.label}`}
                        type="button"
                        className={`ann-chip ${correctedLabel === k.label ? 'active' : ''}`}
                        onClick={() => {
                          setCorrectedLabel(k.label);
                          setCorrectedCategory(k.category);
                        }}
                      >
                        <span>{CAT_META[k.category].icon}</span>
                        {k.label}
                        <span className="ann-chip-count">×{k.count}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            <label className="field-label" style={{ marginTop: 16 }}>
              Your Comment / Justification
            </label>
            <textarea
              className="ann-textarea"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder={
                verdict === 'confirmed'
                  ? 'Why is this a correct match? e.g. "Matches the PAID stamp, red ink."'
                  : verdict === 'relabeled'
                  ? 'Explain the correction. e.g. "This is actually RECEIVED, not PAID."'
                  : verdict === 'rejected'
                  ? 'Why is this NOT a match? e.g. "Just a table border, no actual stamp here."'
                  : 'Any note worth remembering…'
              }
              rows={3}
            />

          </div>
        </div>

        <div className="ann-foot">
          <div className="ann-foot-info">
            {verdict === 'confirmed' && '✅ This region will be added as a new training sample for this label.'}
            {verdict === 'relabeled' && `🎯 A new training sample will be added for "${correctedLabel || '…'}".`}
            {verdict === 'rejected' && '🚫 The region crop will be remembered as a negative example to avoid future false matches.'}
            {verdict === 'note_only' && '💬 Comment saved to the knowledge base, no training changes.'}
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <Button variant="ghost" onClick={onClose} disabled={saving}>Cancel</Button>
            <Button variant="primary" icon="💾" onClick={save} loading={saving} disabled={
              saving ||
              (verdict === 'relabeled' && !correctedLabel.trim())
            }>
              Save Annotation
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

const VerdictButton: React.FC<{
  active: boolean;
  onClick: () => void;
  icon: string;
  color: string;
  title: string;
  desc: string;
}> = ({ active, onClick, icon, color, title, desc }) => (
  <button
    type="button"
    className={`verdict-btn ${active ? 'active' : ''}`}
    style={{ ['--vc' as any]: color }}
    onClick={onClick}
  >
    <span className="verdict-icon">{icon}</span>
    <div>
      <div className="verdict-title">{title}</div>
      <div className="verdict-desc">{desc}</div>
    </div>
  </button>
);

export default AnnotationPanel;