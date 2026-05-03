// Persistent storage wrapper. Upgraded to use IndexedDB natively to support
// large, uncompressed image blobs and avoid QuotaExceeded errors associated
// with localStorage and small persistentStorage limits.

type PersistenceAPI = {
  setItem(key: string, value: string): Promise<void>;
  getItem(key: string): Promise<string | null>;
  removeItem(key: string): Promise<void>;
  clear(): Promise<void>;
};

const DB_NAME = 'ocr_ai_studio_idb';
const STORE_NAME = 'keyval_store';

function getDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE_NAME);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

const idbPersistence: PersistenceAPI = {
  async setItem(key, value) {
    const db = await getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  },
  async getItem(key) {
    const db = await getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(key);
      tx.oncomplete = () => resolve(req.result || null);
      tx.onerror = () => reject(tx.error);
    });
  },
  async removeItem(key) {
    const db = await getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  },
  async clear() {
    const db = await getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
};

const memoryStore = new Map<string, string>();
const memoryPersistence: PersistenceAPI = {
  async setItem(key, value) { memoryStore.set(key, value); },
  async getItem(key) { return memoryStore.has(key) ? memoryStore.get(key)! : null; },
  async removeItem(key) { memoryStore.delete(key); },
  async clear() { memoryStore.clear(); },
};

function hasIDB() {
  try { return typeof window !== 'undefined' && !!window.indexedDB; } catch { return false; }
}

export const persistence: PersistenceAPI = hasIDB() ? idbPersistence : memoryPersistence;
export const persistenceBackend = hasIDB() ? 'indexedDB' : 'memory';