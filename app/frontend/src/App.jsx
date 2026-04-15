import { useState, useRef } from 'react'
import axios from 'axios'

// Class metadata
const CLASSES = [
  { name: 'NonDemented',      severity: 0 },
  { name: 'VeryMildDemented', severity: 1 },
  { name: 'MildDemented',     severity: 2 },
  { name: 'ModerateDemented', severity: 3 },
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function classColor(severity) {
  return ['#22c55e', '#eab308', '#f97316', '#ef4444'][severity]
}

// ── Components ────────────────────────────────────────────────────────────────

function InfoTooltip({ text }) {
  return (
    <span className="tooltip-container">
      <div className="tooltip-icon">
        <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
      </div>
      <span className="tooltip-text">{text}</span>
    </span>
  )
}


function UploadPanel({ patient, setPatient, onResult, onLoading, loading }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  function handleFile(f) {
    if (!f) return
    setFile(f)
    setError(null)
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  async function analyze() {
    if (!file) return
    setError(null)
    onLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const { data } = await axios.post('/predict', form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      onResult({ ...data, filename: file.name })
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Is the backend running?')
    } finally {
      onLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="card-header">
        <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3 16V8a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"></path><path strokeLinecap="round" strokeLinejoin="round" d="M12 11v4m-3-2h6"></path></svg>
        <span className="card-title">Acquire Medical Imagery</span>
      </div>

      <div
        id="upload-zone"
        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload MRI image"
      >
        <input
          ref={inputRef}
          type="file"
          id="mri-file-input"
          accept="image/png,image/jpeg,image/bmp"
          style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files[0])}
        />

        {preview ? (
          <div className="upload-preview">
            <img src={preview} alt="Uploaded MRI scan preview" />
          </div>
        ) : (
          <>
            <div className="upload-icon">
              <svg fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"></path></svg>
            </div>
            <div className="upload-title">Select or drag MRI scan</div>
            <div className="upload-subtitle">DICOM, PNG, JPEG &nbsp;|&nbsp; Max 50 MB</div>
          </>
        )}

        {file && (
          <div className="file-name">
            <svg style={{width:'14px', marginRight:'4px', verticalAlign:'middle'}} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"></path></svg>
            {file.name}
          </div>
        )}
      </div>

      {error && (
        <div className="error-banner" role="alert">
          ⚠️ {error}
        </div>
      )}

      <div className="patient-form">
        <div className="form-group">
          <label>Patient ID</label>
          <input type="text" placeholder="e.g. MRN-10293" value={patient.id} onChange={e => setPatient({...patient, id: e.target.value})} disabled={loading} />
        </div>
        <div className="form-group">
          <label>Patient Name</label>
          <input type="text" placeholder="Last, First" value={patient.name} onChange={e => setPatient({...patient, name: e.target.value})} disabled={loading} />
        </div>
        <div className="form-group">
          <label>Date of Birth</label>
          <input type="date" value={patient.dob} onChange={e => setPatient({...patient, dob: e.target.value})} disabled={loading} />
        </div>
        <div className="form-group">
          <label>Sex</label>
          <select value={patient.sex} onChange={e => setPatient({...patient, sex: e.target.value})} disabled={loading}>
            <option value="">Select...</option>
            <option value="M">Male</option>
            <option value="F">Female</option>
            <option value="O">Other</option>
          </select>
        </div>
      </div>

      <button
        id="analyze-btn"
        className="btn-primary"
        onClick={analyze}
        disabled={!file || loading}
      >
        {loading ? (
          <><span className="spinner" aria-hidden="true" /> Processing...</>
        ) : (
          'Execute Analysis'
        )}
      </button>

      <div className="model-status">
        <div className="status-dot live" />
        CNN + Grad-CAM · 95.26% accuracy · 4 classes
      </div>
    </div>
  )
}


function DiagnosisCard({ result }) {
  const cls = CLASSES[result.class_id]
  const conf = (result.confidence * 100).toFixed(1)
  const color = classColor(cls.severity)

  return (
    <div className="card fade-in" style={{ marginBottom: 24 }}>
      <div className="card-header">
        <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
        <span className="card-title">Diagnostic Confidence</span>
        <InfoTooltip text="AI model confidence based on Softmax distribution. Higher value correlates with stronger feature activation." />
      </div>

      <div className={`diagnosis-badge severity-${cls.severity}`}>
        {cls.name}
      </div>

      <div className="confidence-label">Confidence Score</div>
      <div className="confidence-bar-track">
        <div
          className="confidence-bar-fill"
          style={{ width: `${conf}%`, background: color }}
          role="progressbar"
          aria-valuenow={conf}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
      <div className="confidence-value">{conf}%</div>

      {/* Probability breakdown */}
      <div className="prob-table" role="table" aria-label="Class probabilities">
        {result.probabilities.map((p, i) => (
          <div
            key={i}
            className={`prob-row ${i === result.class_id ? 'active' : ''}`}
            role="row"
          >
            <span className="prob-class-name">{CLASSES[i].name}</span>
            <span className="prob-pct">{(p * 100).toFixed(1)}%</span>
            <div className="prob-mini-bar">
              <div
                className="prob-mini-fill"
                style={{
                  width: `${p * 100}%`,
                  background: classColor(CLASSES[i].severity)
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}


function HeatmapViewer({ result }) {
  const [opacity, setOpacity] = useState(60)

  return (
    <div className="card fade-in" style={{ marginBottom: 24 }}>
      <div className="card-header">
        <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg>
        <span className="card-title">Visualization Overlay</span>
        <InfoTooltip text="Gradient-weighted Class Activation Mapping (Grad-CAM). Highlights regions the CNN found most discriminative." />
      </div>

      <div className="heatmap-overlay-container">
        <img
          src={`data:image/png;base64,${result.original_png}`}
          alt="Original MRI"
          className="heatmap-layer"
          style={{ zIndex: 1 }}
        />
        <img
          src={`data:image/png;base64,${result.gradcam_png}`}
          alt="Grad-CAM Overlay"
          className="heatmap-layer"
          style={{ zIndex: 2, opacity: opacity / 100 }}
        />
      </div>

      <div className="heatmap-controls">
        <div className="slider-header">
          <span>MRI Base</span>
          <span>Heatmap Overlay ({opacity}%)</span>
        </div>
        <input 
          type="range" 
          min="0" max="100" 
          value={opacity} 
          onChange={(e) => setOpacity(e.target.value)}
          className="heatmap-slider"
          aria-label="Adjust heatmap overlay opacity"
        />
      </div>
    </div>
  )
}


function ReportButton({ result, patient, loading, setLoading }) {
  const [genLoading, setGenLoading] = useState(false)
  const [error, setError] = useState(null)

  async function downloadReport() {
    setGenLoading(true)
    setError(null)
    try {
      const res = await axios.post('/report', {
        filename: result.filename,
        class_id: result.class_id,
        probabilities: result.probabilities,
        original_png: result.original_png,
        gradcam_png: result.gradcam_png,
        patient_id: patient.id || null,
        patient_name: patient.name || null,
        scan_date: patient.dob ? new Date().toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
      }, { responseType: 'blob' })

      const url = URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }))
      const a = document.createElement('a')
      a.href = url
      a.download = `NeuroGAN_Report_${result.filename || 'report'}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      setError('PDF generation failed. Is the backend running?')
    } finally {
      setGenLoading(false)
    }
  }

  return (
    <div className="card fade-in">
      <div className="card-header">
        <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
        <span className="card-title">Clinical Report</span>
      </div>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
        Download a professional PDF report containing the scan, Grad-CAM heatmap,
        diagnosis, and probability breakdown.
      </p>

      {error && (
        <div className="error-banner" role="alert" style={{ marginBottom: 8 }}>
          ⚠️ {error}
        </div>
      )}

      <button
        id="download-report-btn"
        className="btn-secondary"
        onClick={downloadReport}
        disabled={genLoading}
      >
        {genLoading ? (
          <><span className="spinner" /> Generating PDF...</>
        ) : (
          <> Download PDF Report</>
        )}
      </button>

      <div style={{ marginTop: 10, fontSize: '0.72rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
        For research use only. Not a substitute for clinical diagnosis.
      </div>
    </div>
  )
}


function EmptyResults() {
  return (
    <div className="card fade-in">
      <div className="empty-state">
        <img src="/clinical_ai_brain.png" alt="AI Neural Brain" className="clinical-graphic" />
        <div className="empty-title">Ready for Analysis</div>
        <div className="empty-subtitle">
          Acquire images and execute analysis<br />to view results and overlays.
        </div>
      </div>
    </div>
  )
}


// ── App Root ──────────────────────────────────────────────────────────────────

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [patient, setPatient] = useState({ id: '', name: '', dob: '', sex: '' })
  const [history, setHistory] = useState([])

  function handleNewResult(res) {
    const newRes = { ...res, internalId: Date.now() }
    setResult(newRes)
    setHistory(prev => [newRes, ...prev])
  }

  return (
    <div className="app-wrapper">
      {/* Navbar */}
      <nav className="navbar" aria-label="Main navigation">
        <div className="navbar-logo">
          <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
          <span className="logo-text">NeuroGAN Clinical</span>
        </div>
        <div className="navbar-badge">STABLE · v2.1.0</div>
      </nav>

      {/* Hero */}
      <header className="hero">
        <h1>Radiological Image Analysis<br />Diagnostic Support System</h1>
        <p>
          GAN-augmented CNN classifier with Grad-CAM explainability.
          Acquire MRI data for clinical-grade diagnostic probabilities.
        </p>
        <div className="hero-stats">
          <div className="stat-item">
            <div className="stat-value">95.26%</div>
            <div className="stat-label">Accuracy</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">0.9528</div>
            <div className="stat-label">F1 Score</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">4</div>
            <div className="stat-label">Classes</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">GAN</div>
            <div className="stat-label">Augmented</div>
          </div>
        </div>
      </header>

      {/* Main Layout wrapper for sidebar */}
      <div className="app-layout">
        
        {/* Left Sidebar: Session History */}
        <aside className="history-sidebar" aria-label="Recent scans">
          <div className="history-header">
            Recent Analyses
            <span style={{ fontSize: '0.8rem', fontWeight: 'normal', color: 'var(--text-muted)' }}>
              {history.length} {history.length === 1 ? 'scan' : 'scans'}
            </span>
          </div>
          {history.length === 0 ? (
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'center', padding: '20px 0' }}>
              No scans analyzed yet.
            </div>
          ) : (
            history.map(item => (
              <div 
                key={item.internalId} 
                className={`history-item ${result?.internalId === item.internalId ? 'active' : ''}`}
                onClick={() => setResult(item)}
              >
                <img src={`data:image/png;base64,${item.original_png}`} className="history-thumb" alt="thumb" />
                <div className="history-info">
                  <div className="history-name">{item.filename}</div>
                  <div className="history-date">
                    {CLASSES[item.class_id].name} ({(item.confidence * 100).toFixed(0)}%)
                  </div>
                </div>
              </div>
            ))
          )}
        </aside>

        {/* Right Content: Upload & Results */}
        <main className="main-grid">
          {/* Main Left: Upload */}
          <section aria-label="Upload MRI scan">
            <UploadPanel
              patient={patient}
              setPatient={setPatient}
              onResult={handleNewResult}
              onLoading={setLoading}
              loading={loading}
            />
          </section>

          {/* Main Right: Results */}
          <section aria-label="Analysis results">
            {result ? (
              <>
                <DiagnosisCard result={result} />
                <HeatmapViewer result={result} />
                <ReportButton result={result} patient={patient} loading={loading} setLoading={setLoading} />
              </>
            ) : (
              <EmptyResults />
            )}
          </section>
        </main>
      </div>
    </div>
  )
}
