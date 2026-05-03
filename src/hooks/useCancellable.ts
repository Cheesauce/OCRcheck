
import { useRef, useCallback } from 'react';

export interface CancelToken {
  cancelled: boolean;
  throwIfCancelled: () => void;
}

export function useCancellable() {
  const tokenRef = useRef<CancelToken | null>(null);

  const start = useCallback((): CancelToken => {
    const token: CancelToken = {
      cancelled: false,
      throwIfCancelled() {
        if (token.cancelled) {
          const err = new Error('Operation cancelled by user.');
          (err as any).cancelled = true;
          throw err;
        }
      },
    };
    tokenRef.current = token;
    return token;
  }, []);

  const cancel = useCallback(() => {
    if (tokenRef.current) {
      tokenRef.current.cancelled = true;
    }
  }, []);

  const isRunning = useCallback(() => {
    return tokenRef.current !== null && !tokenRef.current.cancelled;
  }, []);

  const clear = useCallback(() => {
    tokenRef.current = null;
  }, []);

  return { start, cancel, isRunning, clear };
}

export function isCancelledError(e: any): boolean {
  return !!(e && (e.cancelled || (e.message && /cancel/i.test(e.message))));
}
