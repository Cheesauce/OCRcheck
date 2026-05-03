
import React, { useEffect, useState } from 'react';
import './CompositeGallery.css';
import Card from '../../components/Card/Card';
import Button from '../../components/Button/Button';
import { getAllSamples, type TrainingSample, type SampleCategory } from '../../services/database';
import { buildCompositeImage } from '../../services/compositeBuilder';

interface ClassGroup {
  category: SampleCategory;
  label: string;
  samples: TrainingSample[];
  compositeUrl: string | null;
}

const CAT_META: Record<SampleCategory, { icon: string; color: string }> = {
  logo: { icon: '🏷️', color: '#8b5cf6' },
  signature: { icon: '✍️', color: '#ec4899' },
  stamp: { icon: '🔖', color: '#f59e0b' },
};

const CompositeGallery: React.FC<{ onRefreshStats?: () => void }> = ({ onRefreshStats }) => {
  const [groups, setGroups] = useState<ClassGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | SampleCategory>('all');
  const [building, setBuilding] = useState<string | null>(null);
  const [selectedGroup, setSelectedGroup] = useState<ClassGroup | null>(null);

  const loadGroups = async () => {
    setLoading(true);
    const all = await getAllSamples();
    const map = new Map<string, TrainingSample[]>();
    for (const s of all) {
      const key = `${s.category}::${s.label}`;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(s);
    }
    const next: ClassGroup[] = [];
    for (const [key, samples] of map) {
      const [category, label] = key.split('::');
      next.push({
        category: category as SampleCategory,
        label,
        samples,
        compositeUrl: null,
      });
    }
    next.sort((a, b) => b.samples.length - a.samples.length);
    setGroups(next);
    setLoading(false);
  };

  useEffect(() => {
    loadGroups();
  }, []);

  const buildComposite = async (group: ClassGroup) => {
    const key = `${group.category}::${group.label}`;
    setBuilding(key);
    try {
      const composite = await buildCompositeImage(
        group.samples.map((s) => s.imageData),
        { size: 320, autoAlign: true }
      );
      setGroups((prev) =>
        prev.map((g) =>
          g.category === group.category && g.label === group.label
            ? { ...g, compositeUrl: composite }
            : g
        )
      );
      if (selectedGroup && selectedGroup.category === group.category && selectedGroup.label === group.label) {
        setSelectedGroup({ ...group, compositeUrl: composite });
      }
    } catch (e) {
      console.error('Composite build failed:', e);
    } finally {
      setBuilding(null);
    }
  };

  const buildAllComposites = async () => {
    for (const g of groups) {
      if (g.samples.length > 1 && !g.compositeUrl) {
        await buildComposite(g);
      }
    }
  };

  const filtered = groups.filter((g) => filter === 'all' || g.category === filter);

  return (
    <div className="composite-gallery">
      <div className="cg-header">
        <div>
          <h3 className="cg-title">Composite Prototypes</h3>
          <p className="cg-sub">
            Blend all samples of the same name into an idealized "average image" — exactly what the AI uses internally to match.
          </p>
        </div>
        <div className="cg-actions">
          <div className="filter-chips">
            {(['all', 'logo', 'signature', 'stamp'] as const).map((f) => (
              <button
                key={f}
                className={`chip-btn ${filter === f ? 'active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
          <Button
            variant="secondary"
            icon="✨"
            onClick={buildAllComposites}
            disabled={!!building || groups.every((g) => g.samples.length <= 1 || g.compositeUrl)}
          >
            Build All Composites
          </Button>
          <Button variant="ghost" icon="↻" onClick={loadGroups} disabled={loading}>
            Refresh
          </Button>
        </div>
      </div>

      {loading && (
        <div className="cg-empty">
          <div className="mini-spinner-lg" />
          <div>Loading groups…</div>
        </div>
      )}

      {!loading && filtered.length === 0 && (
        <div className="cg-empty">
          <div className="cg-empty-icon">🎨</div>
          <div className="cg-empty-title">No training groups yet</div>
          <div className="cg-empty-sub">
            Train samples under the same name (e.g. "Crizil") to see them blended here.
          </div>
        </div>
      )}

      <div className="cg-grid">
        {filtered.map((g) => {
          const key = `${g.category}::${g.label}`;
          const isBuilding = building === key;
          const meta = CAT_META[g.category];
          return (
            <div
              key={key}
              className="cg-card"
              style={{ ['--cg-color' as any]: meta.color }}
              onClick={() => setSelectedGroup(g)}
            >
              <div className="cg-card-head">
                <span className="cg-card-icon">{meta.icon}</span>
                <div className="cg-card-label-wrap">
                  <div className="cg-card-label">{g.label}</div>
                  <div className="cg-card-meta">
                    {g.samples.length} sample{g.samples.length === 1 ? '' : 's'}
                  </div>
                </div>
              </div>

              <div className="cg-preview-area">
                {g.compositeUrl ? (
                  <div className="cg-composite-wrap">
                    <img src={g.compositeUrl} alt={`composite of ${g.label}`} />
                    <span className="cg-badge">Composite</span>
                  </div>
                ) : g.samples.length === 1 ? (
                  <div className="cg-composite-wrap">
                    <img src={g.samples[0].imageData} alt={g.label} />
                    <span className="cg-badge single">Single</span>
                  </div>
                ) : (
                  <div className="cg-stack">
                    {g.samples.slice(0, 4).map((s, i) => (
                      <div
                        key={s.id}
                        className="cg-stack-item"
                        style={{
                          backgroundImage: `url(${s.imageData})`,
                          transform: `translate(${i * 6}px, ${i * 6}px) rotate(${(i - 1.5) * 2}deg)`,
                          zIndex: 10 - i,
                        }}
                      />
                    ))}
                  </div>
                )}
              </div>

              <div className="cg-card-foot">
                {g.samples.length > 1 && !g.compositeUrl && (
                  <Button
                    variant="primary"
                    icon={isBuilding ? '' : '✨'}
                    loading={isBuilding}
                    onClick={(e) => {
                      e.stopPropagation();
                      buildComposite(g);
                    }}
                    className="cg-build-btn"
                  >
                    {isBuilding ? 'Blending…' : 'Build Composite'}
                  </Button>
                )}
                {g.compositeUrl && (
                  <Button
                    variant="ghost"
                    icon="↻"
                    onClick={(e) => {
                      e.stopPropagation();
                      buildComposite(g);
                    }}
                    className="cg-build-btn"
                  >
                    Rebuild
                  </Button>
                )}
                {g.samples.length === 1 && (
                  <div className="cg-hint-small">Add more samples for a blended prototype</div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {selectedGroup && (
        <CompositeModal
          group={selectedGroup}
          onClose={() => setSelectedGroup(null)}
          onBuildComposite={buildComposite}
          isBuilding={building === `${selectedGroup.category}::${selectedGroup.label}`}
        />
      )}
    </div>
  );
};

const CompositeModal: React.FC<{
  group: ClassGroup;
  onClose: () => void;
  onBuildComposite: (g: ClassGroup) => void;
  isBuilding: boolean;
}> = ({ group, onClose, onBuildComposite, isBuilding }) => {
  const meta = CAT_META[group.category];

  const downloadComposite = () => {
    if (!group.compositeUrl) return;
    const a = document.createElement('a');
    a.href = group.compositeUrl;
    a.download = `${group.label}_composite.jpg`;
    a.click();
  };

  return (
    <div className="cg-modal-backdrop" onClick={onClose}>
      <div className="cg-modal" onClick={(e) => e.stopPropagation()}>
        <div className="cg-modal-head">
          <div>
            <div className="cg-modal-title">
              <span style={{ fontSize: 22 }}>{meta.icon}</span>
              {group.label}
            </div>
            <div className="cg-modal-sub">
              {group.samples.length} training sample{group.samples.length === 1 ? '' : 's'} · category: {group.category}
            </div>
          </div>
          <button className="cg-modal-close" onClick={onClose}>×</button>
        </div>

        <div className="cg-modal-body">
          <div className="cg-modal-composite-section">
            <div className="cg-section-title">🎨 Composite Prototype</div>
            {group.compositeUrl ? (
              <>
                <div className="cg-modal-composite">
                  <img src={group.compositeUrl} alt="composite" />
                </div>
                <div className="cg-modal-composite-actions">
                  <Button variant="primary" icon="⬇️" onClick={downloadComposite}>
                    Download Composite
                  </Button>
                  <Button
                    variant="secondary"
                    icon="↻"
                    onClick={() => onBuildComposite(group)}
                    loading={isBuilding}
                  >
                    Rebuild
                  </Button>
                </div>
                <p className="cg-modal-hint">
                  This image is a pixel-wise average of all aligned samples. Areas where all samples agree appear sharp; areas where they differ appear softer. The AI uses this same averaging technique internally on its feature embeddings.
                </p>
              </>
            ) : group.samples.length > 1 ? (
              <>
                <div className="cg-modal-composite empty">
                  <div>No composite built yet</div>
                </div>
                <Button
                  variant="primary"
                  icon="✨"
                  onClick={() => onBuildComposite(group)}
                  loading={isBuilding}
                >
                  {isBuilding ? 'Blending…' : 'Build Composite'}
                </Button>
              </>
            ) : (
              <div className="cg-modal-composite empty">
                <div>Only 1 sample — no blending possible.<br/>Add more samples to generate a composite.</div>
              </div>
            )}
          </div>

          <div className="cg-modal-samples-section">
            <div className="cg-section-title">
              📸 Individual Samples ({group.samples.length})
            </div>
            <div className="cg-modal-samples">
              {group.samples.map((s) => (
                <div key={s.id} className="cg-modal-sample">
                  <img src={s.imageData} alt={s.name} />
                  <div className="cg-modal-sample-name" title={s.name}>{s.name}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CompositeGallery;
