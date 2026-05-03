
import React from 'react';
import './Card.css';

interface Props {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
}

const Card: React.FC<Props> = ({ title, subtitle, children, right, className }) => {
  return (
    <div className={`card ${className || ''}`}>
      {(title || right) && (
        <div className="card-head">
          <div>
            {title && <h3 className="card-title">{title}</h3>}
            {subtitle && <p className="card-sub">{subtitle}</p>}
          </div>
          {right && <div>{right}</div>}
        </div>
      )}
      <div className="card-body">{children}</div>
    </div>
  );
};

export default Card;
