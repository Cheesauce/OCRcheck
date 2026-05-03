
import React from 'react';
import './Sidebar.css';
import type { ViewKey } from '../../App';

interface Props {
  current: ViewKey;
  onNavigate: (v: ViewKey) => void;
}

const items: { key: ViewKey; label: string; icon: string; hint: string }[] = [
  { key: 'dashboard', label: 'Dashboard', icon: '📊', hint: 'Overview' },
  { key: 'ocr', label: 'OCR Workspace', icon: '📄', hint: 'Scan PDFs & Images' },
  { key: 'search', label: 'Text Search', icon: '🔍', hint: 'Search all OCR text' },
  { key: 'recognition', label: 'AI Recognition', icon: '🧠', hint: 'Train & Identify' },
  { key: 'database', label: 'Database', icon: '🗄️', hint: 'Manage Samples' },
];

const Sidebar: React.FC<Props> = ({ current, onNavigate }) => {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-logo">◉</div>
        <div>
          <div className="brand-title">OCR Studio</div>
          <div className="brand-sub">AI Recognition Suite</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {items.map((item) => (
          <button
            key={item.key}
            className={`nav-item ${current === item.key ? 'active' : ''}`}
            onClick={() => onNavigate(item.key)}
          >
            <span className="nav-icon">{item.icon}</span>
            <div className="nav-text">
              <span className="nav-label">{item.label}</span>
              <span className="nav-hint">{item.hint}</span>
            </div>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-dot" />
        <div>
          <div className="footer-title">Local Engine</div>
          <div className="footer-sub">Tesseract.js 5 · TF.js</div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
