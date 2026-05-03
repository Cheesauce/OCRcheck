
import { useState, useCallback } from 'react';
import type { ToastMsg } from '../components/Toast/Toast';

let idCounter = 0;

export function useToasts() {
  const [toasts, setToasts] = useState<ToastMsg[]>([]);

  const push = useCallback((type: ToastMsg['type'], text: string) => {
    idCounter += 1;
    setToasts((prev) => [...prev, { id: idCounter, type, text }]);
  }, []);

  const remove = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return { toasts, push, remove };
}
