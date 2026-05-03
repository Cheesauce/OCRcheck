
import React, { useState } from 'react';
import './RegionOverlay.css';
import type { RegionMatch } from '../../services/recognitionService';

interface Props {
  imageUrl: string;
  regions: RegionMatch[];
  onRegionClick?: (index: number) => void;
  onBackgroundClick?: () => void;
}

const CAT_COLORS: Record<string, string> = {
  word: '#06b6d4',
  logo: '#8b5cf6',
  signature: '#ec4899',
  stamp: '#f59e0b',
};

const RegionOverlay: React.FC<Props> = ({ imageUrl, regions, onRegionClick, onBackgroundClick }) => {
  const [hovered, setHovered] = useState<number | null>(null);

  return (
    <div
      className="region-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget && onBackgroundClick) onBackgroundClick();
      }}
    >
      <img
        src={imageUrl}
        alt="scanned"
        className="region-overlay-img"
        onClick={(e) => {
          if (onBackgroundClick) onBackgroundClick();
          e.stopPropagation();
        }}
      />

      {regions.map((r, i) => {
        const color = CAT_COLORS[r.category] || '#6366f1';
        const isHovered = hovered === i;
        const showRotation = r.rotation !== undefined && r.rotation !== 0;
        const isWord = r.category === 'word';
        const clickable = !!onRegionClick;
        return (
          <div
            key={i}
            className={`region-box ${isHovered ? 'hovered' : ''} ${clickable ? 'clickable' : ''} ${isWord ? 'is-word' : ''}`}
            style={{
              left: `${r.x * 100}%`,
              top: `${r.y * 100}%`,
              width: `${r.width * 100}%`,
              height: `${r.height * 100}%`,
              borderColor: color,
              boxShadow: isHovered ? `0 0 0 2px ${color}, 0 8px 24px ${color}66` : (isWord ? `0 0 0 1px ${color}88, 0 2px 10px ${color}33` : undefined),
              background: isHovered ? `${color}33` : `${color}14`,
            }}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            onClick={(e) => {
              e.stopPropagation();
              if (onRegionClick) onRegionClick(i);
            }}
            title={clickable ? `Click to annotate "${r.label}"` : r.label}
          >
            <div className="region-num" style={{ background: color }}>
              {i + 1}
            </div>
            <div
              className="region-label"
              style={{ background: color }}
            >
              {isWord && <span className="region-priority-star">★</span>}
              <span className="region-label-text">{r.label}</span>
              <span className="region-label-conf">{(r.confidence * 100).toFixed(0)}%</span>
              {r.textMatch && r.textMatch.similarity > 0 && (
                <span className="region-label-text-match" title={`OCR: "${r.textMatch.ocrText}" · ${(r.textMatch.similarity * 100).toFixed(0)}% match`}>
                  🔤
                </span>
              )}
              {showRotation && (
                <span className="region-label-rot" title={`rotation: ${r.rotation === -1 ? 'mirrored' : r.rotation + '°'}`}>
                  {r.rotation === -1 ? '⇄' : `↻${r.rotation}°`}
                </span>
              )}
              {clickable && isHovered && (
                <span className="region-label-edit">✎ Annotate</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default RegionOverlay;
