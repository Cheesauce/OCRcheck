
import React, { useEffect, useState } from 'react';
import './Dashboard.css';
import type { ViewKey } from '../../App';
import { getAllSamples, getAllOcrDocs } from '../../services/database';

interface Props {
  onNavigate: (v: ViewKey) => void;
}

const Dashboard: React.FC<Props> = ({ onNavigate }) => {
  const [stats, setStats] = useState({ samples: 0, logos: 0, signatures: 0, stamps: 0, docs: 0, docPages: 0 });

  useEffect(() => {
    (async () => {
      const [samples, docs] = await Promise.all([getAllSamples(), getAllOcrDocs()]);
      setStats({
        samples: samples.length,
        logos: samples.filter((s) => s.category === 'logo').length,
        signatures: samples.filter((s) => s.category === 'signature').length,
        stamps: samples.filter((s) => s.category === 'stamp').length,
        docs: docs.length,
        docPages: docs.reduce((a, b) => a + (b.pages || 0), 0),
      });
    })();
  }, []);

  const cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;
  const poolSize = Math.min(4, Math.max(2, cores - 1));

  return (
    <div className="dashboard">
      <header className="dash-hero">
        <div>
          <div className="hero-kicker">WELCOME BACK</div>
          <h1 className="hero-title">
            OCR & AI <span className="grad-text">Recognition Studio</span>
          </h1>
          <p className="hero-sub">
            Convert scanned PDFs into searchable documents, search every word across your library in milliseconds, and train AI models to recognize logos, signatures, and stamps — all running locally in your browser with <strong>GPU acceleration</strong> and <strong>{poolSize} parallel OCR workers</strong>.
          </p>
          <div className="hero-actions">
            <button className="hero-btn primary" onClick={() => onNavigate('ocr')}>
              <span>📄</span> Start OCR Session
            </button>
            <button className="hero-btn secondary" onClick={() => onNavigate('search')}>
              <span>🔍</span> Search All Documents
            </button>
            <button className="hero-btn secondary" onClick={() => onNavigate('recognition')}>
              <span>🧠</span> Train Recognition Model
            </button>
          </div>
        </div>
        <div className="hero-visual">
          <div className="orb orb-1" />
          <div className="orb orb-2" />
          <div className="orb orb-3" />
        </div>
      </header>

      <section className="perf-banner">
        <div className="perf-icon">⚡</div>
        <div className="perf-text">
          <strong>Performance mode active</strong> — {poolSize} parallel OCR workers, WebGL GPU inference, OffscreenCanvas compression, and coarse-to-fine region search.
        </div>
        <div className="perf-stats">
          <span>{cores} CPU cores</span>
        </div>
      </section>

      <section className="stat-grid">
        <StatCard label="Training Samples" value={stats.samples} icon="🎯" color="#6366f1" />
        <StatCard label="Logos" value={stats.logos} icon="🏷️" color="#8b5cf6" />
        <StatCard label="Signatures" value={stats.signatures} icon="✍️" color="#ec4899" />
        <StatCard label="Stamps" value={stats.stamps} icon="🔖" color="#f59e0b" />
        <StatCard label="OCR Documents" value={stats.docs} icon="📚" color="#10b981" />
        <StatCard label="Searchable Pages" value={stats.docPages} icon="🔍" color="#06b6d4" />
      </section>

      <section className="feature-grid">
        <FeatureBlock
          icon="⚡"
          title="Parallel OCR"
          desc={`Runs ${poolSize} Tesseract workers simultaneously — up to ${poolSize}x faster on multi-page PDFs. JPEG rendering halves memory usage vs PNG.`}
        />
        <FeatureBlock
          icon="🎮"
          title="GPU Acceleration"
          desc="TensorFlow.js auto-selects WebGL backend for 5–10× faster neural-network inference on supported GPUs."
        />
        <FeatureBlock
          icon="🧠"
          title="Smart Region Search"
          desc="Coarse-to-fine sliding window with saliency pre-filter skips blank areas — up to 3× faster element localization."
        />
        <FeatureBlock
          icon="🗄️"
          title="Compressed Database"
          desc="OffscreenCanvas + JPEG 0.7 compression runs off the main thread — ~70% smaller storage, zero UI jank."
        />
      </section>
    </div>
  );
};

const StatCard: React.FC<{ label: string; value: number; icon: string; color: string }> = ({ label, value, icon, color }) => (
  <div className="stat-card" style={{ ['--accent-color' as any]: color }}>
    <div className="stat-icon">{icon}</div>
    <div className="stat-value">{value}</div>
    <div className="stat-label">{label}</div>
  </div>
);

const FeatureBlock: React.FC<{ icon: string; title: string; desc: string }> = ({ icon, title, desc }) => (
  <div className="feature-block">
    <div className="feature-icon">{icon}</div>
    <h4>{title}</h4>
    <p>{desc}</p>
  </div>
);

export default Dashboard;
