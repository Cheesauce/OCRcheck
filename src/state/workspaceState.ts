// Global, in-memory persistence of workspace state so that switching
// sidebar tabs does NOT lose the user's work.

import type { OcrResult, OcrQuality } from '../services/ocrService';
import type { SampleCategory } from '../services/database';
import type { RecognitionQuality } from '../services/recognitionService';

export interface OcrWorkspaceState {
  file: File | null;
  fileName: string;
  lang: string;
  quality: OcrQuality;
  result: OcrResult | null;
  activePage: number;
  edited: boolean;
}

export interface StagedImageState {
  id: string;
  sourceName: string;
  pageNumber?: number;
  previewUrl: string;
  dataUrl: string;
  originalSize: number;
  compressedSize: number;
  categories: SampleCategory[];
  labelOverride?: string;
}

export interface PredictImageState {
  id: string;
  sourceName: string;
  pageNumber?: number;
  previewUrl: string;
  dataUrl: string;
  prediction: any | null;
  status: 'pending' | 'running' | 'done' | 'error' | 'no_match';
  error?: string;
  scannedAt?: number;
  sourceFileId?: string;
  qualityUsed?: RecognitionQuality;
  embeddedText?: { text: string; words: any[] };
}

export interface RecognitionWorkspaceState {
  mode: 'train' | 'predict' | 'gallery';
  defaultCategories: SampleCategory[];
  label: string;
  staged: StagedImageState[];
  predictItems: PredictImageState[];
  ocrTrainingEnabled?: boolean;
  ocrLang?: string;
  recognitionQuality?: RecognitionQuality;
}

export interface SearchTextState {
  query: string;
  wholeWord: boolean;
  filterDocId: string;
  viewMode?: 'list' | 'grid';
  selectedDocIds?: string[];
}

export const ocrWorkspaceState: OcrWorkspaceState = {
  file: null,
  fileName: '',
  lang: 'eng',
  quality: 'precise',
  result: null,
  activePage: 0,
  edited: false,
};

export const recognitionWorkspaceState: RecognitionWorkspaceState = {
  mode: 'train',
  defaultCategories: ['logo'],
  label: '',
  staged: [],
  predictItems: [],
  ocrTrainingEnabled: true,
  ocrLang: 'eng',
  recognitionQuality: 'balanced',
};

export const searchTextState: SearchTextState = {
  query: '',
  wholeWord: false,
  filterDocId: 'all',
  viewMode: 'list',
  selectedDocIds: [],
};

export function resetOcrState() {
  ocrWorkspaceState.file = null;
  ocrWorkspaceState.fileName = '';
  ocrWorkspaceState.result = null;
  ocrWorkspaceState.activePage = 0;
  ocrWorkspaceState.edited = false;
}