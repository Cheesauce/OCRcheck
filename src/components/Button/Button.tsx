
import React from 'react';
import './Button.css';

interface Props extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  loading?: boolean;
  icon?: string;
}

const Button: React.FC<Props> = ({ variant = 'primary', loading, icon, children, className, disabled, ...rest }) => {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={`btn btn-${variant} ${loading ? 'loading' : ''} ${className || ''}`}
    >
      {loading ? (
        <span className="spinner" />
      ) : (
        icon && <span className="btn-icon">{icon}</span>
      )}
      <span>{children}</span>
    </button>
  );
};

export default Button;
