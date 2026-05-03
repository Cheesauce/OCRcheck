
import React, { useEffect } from 'react';
import './Toast.css';

export interface ToastMsg {
  id: number;
  type: 'success' | 'error' | 'info';
  text: string;
}

interface Props {
  toasts: ToastMsg[];
  onRemove: (id: number) => void;
}

const Toast: React.FC<Props> = ({ toasts, onRemove }) => {
  return (
    <div className="toast-stack">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onRemove={onRemove} />
      ))}
    </div>
  );
};

const ToastItem: React.FC<{ toast: ToastMsg; onRemove: (id: number) => void }> = ({ toast, onRemove }) => {
  useEffect(() => {
    const t = setTimeout(() => onRemove(toast.id), 3500);
    return () => clearTimeout(t);
  }, [toast.id, onRemove]);

  const icon = toast.type === 'success' ? '✓' : toast.type === 'error' ? '✕' : 'ℹ';
  return (
    <div className={`toast toast-${toast.type}`}>
      <span className="toast-icon">{icon}</span>
      <span>{toast.text}</span>
    </div>
  );
};

export default Toast;
