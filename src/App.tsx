
import React, { useState } from 'react';
import './App.css';
import Sidebar from './components/Sidebar/Sidebar';
import OcrWorkspace from './features/ocr/OcrWorkspace';
import RecognitionStudio from './features/recognition/RecognitionStudio';
import DatabaseExplorer from './features/database/DatabaseExplorer';
import Dashboard from './features/dashboard/Dashboard';
import TextSearch from './features/search/TextSearch';

export type ViewKey = 'dashboard' | 'ocr' | 'search' | 'recognition' | 'database';

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('React error:', error, info);
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{
          padding: 40,
          maxWidth: 800,
          margin: '40px auto',
          background: '#1a2138',
          border: '1px solid #ef4444',
          borderRadius: 12,
          color: '#e6e9f2'
        }}>
          <h2 style={{ color: '#ef4444', margin: '0 0 12px' }}>⚠️ Runtime Error</h2>
          <pre style={{
            background: '#0a0e1a',
            padding: 14,
            borderRadius: 8,
            overflow: 'auto',
            fontSize: 12.5,
            lineHeight: 1.6
          }}>
            {this.state.error.stack || this.state.error.message}
          </pre>
          <button
            onClick={() => this.setState({ error: null })}
            style={{
              marginTop: 14,
              padding: '10px 20px',
              background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
              color: 'white',
              border: 'none',
              borderRadius: 8,
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [view, setView] = useState<ViewKey>('dashboard');

  return (
    <ErrorBoundary>
      <div className="app-shell">
        <Sidebar current={view} onNavigate={setView} />
        <main className="app-main">
          {view === 'dashboard' && <Dashboard onNavigate={setView} />}
          {view === 'ocr' && <OcrWorkspace />}
          {view === 'search' && <TextSearch />}
          {view === 'recognition' && <RecognitionStudio />}
          {view === 'database' && <DatabaseExplorer />}
        </main>
      </div>
    </ErrorBoundary>
  );
}

export default App;
