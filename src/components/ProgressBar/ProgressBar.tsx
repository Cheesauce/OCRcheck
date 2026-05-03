
import React from 'react';
import './ProgressBar.css';

interface Props {
  value: number;
  label?: string;
}

const ProgressBar: React.FC<Props> = ({ value, label }) => {
  const pct = Math.round(Math.min(100, Math.max(0, value)));
  return (
    <div className="progress-wrap">
      {label && (
        <div className="progress-label">
          <span>{label}</span>
          <span className="progress-pct">{pct}%</span>
        </div>
      )}
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
};

export default ProgressBar;
