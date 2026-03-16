import { useState, useRef, useCallback } from 'react'

const API = 'http://localhost:8000'

const SEVERITY_STYLE = {
  High:    { bg: '#1f0808', border: '#6b1a1a', text: '#ff8a8a', dot: '#ef4444' },
  Medium:  { bg: '#1f1408', border: '#6b4a10', text: '#ffc875', dot: '#f59e0b' },
  Low:     { bg: '#081a0e', border: '#135228', text: '#7ee8a2', dot: '#22c55e' },
  Unknown: { bg: '#0f151c', border: '#1c2a38', text: '#7a9ab5', dot: '#3d5a72' },
}

const CLASS_META = {
  'pothole':             { icon: '⚫', severity: 'High'   },
  'Alligator':           { icon: '🔶', severity: 'High'   },
  'Rutting':             { icon: '〰️', severity: 'High'   },
  'Longitudinal-Crack':  { icon: '➖', severity: 'Medium' },
  'Lateral-Crack':       { icon: '↔️', severity: 'Medium' },
  'Edge Cracking':       { icon: '📐', severity: 'Medium' },
  'Ravelling':           { icon: '🔘', severity: 'Low'    },
  'Striping':            { icon: '🟡', severity: 'Low'    },
}

function getSeverityStyle(s) { return SEVERITY_STYLE[s] || SEVERITY_STYLE.Unknown }

export default function App() {
  const [file, setFile]           = useState(null)
  const [preview, setPreview]     = useState(null)
  const [result, setResult]       = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [drag, setDrag]           = useState(false)
  const [conf, setConf]           = useState(25)
  const [tab, setTab]             = useState('annotated')
  const inputRef = useRef()

  const reset = () => {
    setFile(null); setPreview(null); setResult(null)
    setError(null); setTab('annotated')
    if (inputRef.current) inputRef.current.value = ''
  }

  const loadFile = (f) => {
    if (!f || !f.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPG, PNG, WEBP).'); return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null); setError(null)
  }

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDrag(false)
    loadFile(e.dataTransfer.files[0])
  }, [])

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch(`${API}/predict/both?conf=${conf / 100}`, {
        method: 'POST', body: fd,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Server error ${res.status}`)
      }
      const data = await res.json()
      setResult(data); setTab('annotated')
    } catch (e) {
      setError(e.message.includes('fetch')
        ? 'Cannot reach backend. Make sure uvicorn is running on port 8000.'
        : e.message)
    } finally {
      setLoading(false)
    }
  }

  const severityCounts = result?.detections.reduce((acc, d) => {
    acc[d.severity] = (acc[d.severity] || 0) + 1; return acc
  }, {}) ?? {}

  return (
    <div style={css.page}>

      {/* ── Scanline effect ── */}
      <div style={css.scanline} />

      {/* ── Header ── */}
      <header style={css.header}>
        <div style={css.headerInner}>
          <div style={css.logo}>
            <div style={css.logoBox}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M3 9h18M3 15h18M9 3v18M15 3v18" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <div>
              <div style={css.logoTitle}>RoadScan AI</div>
              <div style={css.logoSub}>Urban Infrastructure Damage Detector</div>
            </div>
          </div>
          <div style={css.headerBadges}>
            <div style={css.badge}>
              <span style={css.liveDot} />
              Model Active
            </div>
            <div style={{ ...css.badge, color: 'var(--muted2)' }}>YOLOv8 · 8 classes</div>
          </div>
        </div>
      </header>

      {/* ── Main layout ── */}
      <div style={css.layout}>

        {/* ════ LEFT SIDEBAR ════ */}
        <aside style={css.sidebar}>

          {/* Upload */}
          <div style={css.block}>
            <div style={css.blockLabel}>INPUT IMAGE</div>
            <div
              style={{ ...css.dropzone, ...(drag ? css.dropActive : {}), ...(preview ? { cursor: 'default', minHeight: 'auto' } : {}) }}
              onDragOver={e => { e.preventDefault(); setDrag(true) }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
              onClick={() => !preview && inputRef.current?.click()}
            >
              <input ref={inputRef} type="file" accept="image/*"
                style={{ display: 'none' }}
                onChange={e => loadFile(e.target.files[0])} />

              {preview ? (
                <div style={{ position: 'relative' }}>
                  <img src={preview} alt="preview" style={css.previewImg} />
                  <button style={css.changeBtn}
                    onClick={e => { e.stopPropagation(); inputRef.current?.click() }}>
                    ↺ Change
                  </button>
                </div>
              ) : (
                <div style={css.dropInner}>
                  <div style={css.dropIcon}>
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <div style={css.dropTitle}>Drop image here</div>
                  <div style={css.dropSub}>or click to browse · JPG PNG WEBP</div>
                </div>
              )}
            </div>
          </div>

          {/* Confidence */}
          <div style={css.block}>
            <div style={{ ...css.blockLabel, display: 'flex', justifyContent: 'space-between' }}>
              <span>CONFIDENCE THRESHOLD</span>
              <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-head)', fontSize: 13 }}>{conf}%</span>
            </div>
            <input type="range" min="5" max="95" value={conf}
              onChange={e => setConf(+e.target.value)}
              style={css.slider} />
            <div style={css.sliderHints}>
              <span>More detections</span><span>Fewer, surer</span>
            </div>
          </div>

          {/* Analyze */}
          <button
            style={{ ...css.analyzeBtn, ...(loading || !file ? css.analyzeDim : {}) }}
            onClick={analyze} disabled={loading || !file}
          >
            {loading
              ? <><span style={css.spin} />Analyzing...</>
              : <><ScanIcon />Detect Damage</>
            }
          </button>

          {file && !loading && (
            <button style={css.clearBtn} onClick={reset}>✕ Clear</button>
          )}

          {error && (
            <div style={css.errorBox}>
              <span style={{ color: '#ff6b6b' }}>⚠ </span>{error}
            </div>
          )}

          {/* Legend */}
          <div style={css.block}>
            <div style={css.blockLabel}>DAMAGE CLASSES</div>
            <div style={css.legend}>
              {Object.entries(CLASS_META).map(([label, { icon, severity }]) => {
                const sc = getSeverityStyle(severity)
                return (
                  <div key={label} style={css.legendRow}>
                    <span style={{ fontSize: 13 }}>{icon}</span>
                    <span style={css.legendName}>{label}</span>
                    <span style={{ ...css.legendSev, color: sc.text, background: sc.bg, border: `1px solid ${sc.border}` }}>
                      {severity}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

        </aside>

        {/* ════ MAIN CONTENT ════ */}
        <main style={css.content}>

          {/* Stat cards — only when result exists */}
          {result && (
            <div style={{ ...css.statsRow, animation: 'fadeUp 0.4s ease' }}>
              <StatCard label="Total Detections" value={result.total_detections} color="var(--accent)" />
              <StatCard label="Inference Time"   value={`${result.inference_time_ms} ms`} color="#a78bfa" />
              <StatCard label="High Risk"        value={severityCounts.High   ?? 0} color="#ef4444" />
              <StatCard label="Medium Risk"      value={severityCounts.Medium ?? 0} color="#f59e0b" />
              <StatCard label="Low Risk"         value={severityCounts.Low    ?? 0} color="#22c55e" />
            </div>
          )}

          {/* Image viewer */}
          <div style={css.imageCard}>
            {result && (
              <div style={css.tabs}>
                {['annotated', 'original'].map(t => (
                  <button key={t} style={{ ...css.tabBtn, ...(tab === t ? css.tabActive : {}) }}
                    onClick={() => setTab(t)}>
                    {t === 'annotated' ? '🔍 Annotated' : '🖼 Original'}
                  </button>
                ))}
                <div style={css.tabRight}>
                  <span style={css.dimText}>{result.image_size.width} × {result.image_size.height}px</span>
                </div>
              </div>
            )}

            <div style={css.imageBox}>
              {/* Loading overlay */}
              {loading && (
                <div style={css.loadOverlay}>
                  <div style={css.loadBox}>
                    <div style={css.bigSpin} />
                    <div style={css.loadTitle}>Running Inference</div>
                    <div style={css.loadSub}>YOLOv8 is scanning the image...</div>
                    <div style={css.loadBar}><div style={css.loadBarFill} /></div>
                  </div>
                </div>
              )}

              {/* Result images */}
              {result && tab === 'annotated' && !loading && (
                <img
                  src={`data:image/png;base64,${result.image_base64}`}
                  alt="Annotated result"
                  style={{ ...css.displayImg, animation: 'fadeUp 0.35s ease' }}
                />
              )}
              {preview && (!result || tab === 'original') && !loading && (
                <img src={preview} alt="Original"
                  style={{ ...css.displayImg, animation: 'fadeUp 0.35s ease' }} />
              )}
              {!preview && !loading && (
                <div style={css.emptyState}>
                  <div style={css.emptyIcon}>🛣️</div>
                  <div style={css.emptyTitle}>No image loaded</div>
                  <div style={css.emptySub}>Upload a road photo to begin detection</div>
                </div>
              )}
            </div>
          </div>

          {/* Detections list */}
          {result && (
            <div style={{ ...css.detCard, animation: 'fadeUp 0.5s ease' }}>
              <div style={css.detCardHeader}>
                <span style={css.blockLabel}>DETECTION RESULTS</span>
                <span style={css.totalBadge}>{result.total_detections} found</span>
              </div>

              {result.total_detections === 0 ? (
                <div style={css.noDetect}>
                  <span style={{ fontSize: 24 }}>✅</span>
                  <span>No road damage detected at {conf}% confidence threshold.</span>
                </div>
              ) : (
                <>
                  {/* Summary pills */}
                  <div style={css.summaryRow}>
                    {Object.entries(result.summary).map(([label, count]) => (
                      <div key={label} style={css.summaryPill}>
                        {CLASS_META[label]?.icon ?? '●'}&nbsp;{label}
                        <span style={css.pillCount}>{count}</span>
                      </div>
                    ))}
                  </div>

                  {/* Detection rows */}
                  <div style={css.detRows}>
                    {result.detections.map((d, i) => {
                      const sc = getSeverityStyle(d.severity)
                      const icon = CLASS_META[d.label]?.icon ?? '●'
                      return (
                        <div key={i} style={css.detRow}>
                          <div style={css.detRowLeft}>
                            <span style={css.detIcon}>{icon}</span>
                            <div>
                              <div style={css.detName}>{d.label}</div>
                              <div style={css.detBbox}>
                                [{d.bbox.map(v => Math.round(v)).join(', ')}]
                              </div>
                            </div>
                          </div>
                          <div style={css.detRowRight}>
                            <span style={{ ...css.sevBadge, background: sc.bg, border: `1px solid ${sc.border}`, color: sc.text }}>
                              <span style={{ ...css.sevDot, background: sc.dot }} />
                              {d.severity}
                            </span>
                            <div style={css.confWrap}>
                              <div style={css.confTrack}>
                                <div style={{
                                  ...css.confFill,
                                  width: `${d.confidence}%`,
                                  background: d.confidence > 75 ? '#22c55e'
                                            : d.confidence > 50 ? '#f59e0b' : '#ef4444'
                                }} />
                              </div>
                              <span style={css.confNum}>{d.confidence}%</span>
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </>
              )}
            </div>
          )}

        </main>
      </div>
    </div>
  )
}

/* ── Sub-components ── */
function StatCard({ label, value, color }) {
  return (
    <div style={{ ...css.statCard, borderColor: color + '33' }}>
      <div style={{ ...css.statVal, color }}>{value}</div>
      <div style={css.statLbl}>{label}</div>
    </div>
  )
}

function ScanIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
      <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
      <path d="m21 21-4.35-4.35" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
      <path d="M11 8v6M8 11h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
  )
}

/* ── All styles ── */
const css = {
  page: { minHeight: '100vh', display: 'flex', flexDirection: 'column', position: 'relative', background: 'var(--bg)' },

  scanline: {
    position: 'fixed', top: 0, left: 0, right: 0, height: '30%',
    background: 'linear-gradient(transparent, rgba(0,229,255,0.015), transparent)',
    pointerEvents: 'none', zIndex: 0,
    animation: 'scanline 6s linear infinite',
  },

  header: {
    borderBottom: '1px solid var(--border)',
    background: 'rgba(7,9,13,0.92)',
    backdropFilter: 'blur(16px)',
    position: 'sticky', top: 0, zIndex: 100,
  },
  headerInner: {
    maxWidth: 1400, margin: '0 auto', padding: '12px 28px',
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  },
  logo: { display: 'flex', alignItems: 'center', gap: 14 },
  logoBox: {
    width: 42, height: 42, borderRadius: 10,
    background: 'var(--surface)', border: '1px solid var(--border2)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    boxShadow: '0 0 20px rgba(0,229,255,0.08)',
  },
  logoTitle: { fontFamily: 'var(--font-head)', fontSize: 17, fontWeight: 800, letterSpacing: '-0.03em', color: 'var(--text)' },
  logoSub: { fontSize: 11, color: 'var(--muted)', marginTop: 1 },
  headerBadges: { display: 'flex', gap: 10 },
  badge: {
    display: 'flex', alignItems: 'center', gap: 7,
    padding: '5px 13px', borderRadius: 20,
    background: 'var(--surface)', border: '1px solid var(--border)',
    fontSize: 11, color: 'var(--muted2)', fontFamily: 'var(--font-head)',
  },
  liveDot: {
    width: 7, height: 7, borderRadius: '50%', background: '#22c55e',
    boxShadow: '0 0 8px #22c55e', animation: 'pulse 2s infinite',
  },

  layout: {
    maxWidth: 1400, margin: '0 auto', padding: '24px 28px',
    display: 'grid', gridTemplateColumns: '320px 1fr', gap: 24,
    flex: 1, position: 'relative', zIndex: 1,
  },

  /* Sidebar */
  sidebar: { display: 'flex', flexDirection: 'column', gap: 18 },
  block: { display: 'flex', flexDirection: 'column', gap: 9 },
  blockLabel: {
    fontSize: 10, fontFamily: 'var(--font-head)', fontWeight: 700,
    letterSpacing: '0.14em', color: 'var(--muted)', textTransform: 'uppercase',
  },

  dropzone: {
    border: '1.5px dashed var(--border2)', borderRadius: 12,
    minHeight: 190, display: 'flex', alignItems: 'center', justifyContent: 'center',
    cursor: 'pointer', background: 'var(--surface)',
    transition: 'border-color .2s, background .2s', overflow: 'hidden',
  },
  dropActive: { borderColor: 'var(--accent)', background: 'rgba(0,229,255,0.04)' },
  dropInner: { textAlign: 'center', padding: '28px 20px' },
  dropIcon: { marginBottom: 12, opacity: 0.65 },
  dropTitle: { fontFamily: 'var(--font-head)', fontSize: 14, fontWeight: 600, color: 'var(--muted2)', marginBottom: 5 },
  dropSub: { fontSize: 11, color: 'var(--muted)' },
  previewImg: { width: '100%', display: 'block', maxHeight: 240, objectFit: 'cover', borderRadius: 10 },
  changeBtn: {
    position: 'absolute', bottom: 8, right: 8,
    padding: '4px 10px', borderRadius: 6, fontSize: 11,
    background: 'rgba(7,9,13,0.88)', border: '1px solid var(--border2)',
    color: 'var(--muted2)', cursor: 'pointer', fontFamily: 'var(--font-body)',
  },

  slider: { width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' },
  sliderHints: { display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--muted)' },

  analyzeBtn: {
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
    width: '100%', padding: '12px 0', borderRadius: 10, border: 'none',
    background: 'linear-gradient(135deg, #009bbd 0%, #00e5ff 100%)',
    color: '#07090d', fontFamily: 'var(--font-head)', fontSize: 14, fontWeight: 800,
    cursor: 'pointer', letterSpacing: '0.03em',
    boxShadow: '0 0 28px rgba(0,229,255,0.22)',
    transition: 'opacity .2s, box-shadow .2s',
  },
  analyzeDim: { opacity: 0.28, cursor: 'not-allowed', boxShadow: 'none' },
  spin: {
    display: 'inline-block', width: 13, height: 13, borderRadius: '50%',
    border: '2px solid rgba(7,9,13,0.3)', borderTopColor: '#07090d',
    animation: 'spin 0.65s linear infinite',
  },
  clearBtn: {
    width: '100%', padding: '8px 0', borderRadius: 10,
    background: 'transparent', border: '1px solid var(--border)',
    color: 'var(--muted)', fontSize: 12, cursor: 'pointer', fontFamily: 'var(--font-body)',
    transition: 'border-color .2s',
  },
  errorBox: {
    padding: '10px 13px', borderRadius: 8,
    background: '#180808', border: '1px solid #5a1212',
    fontSize: 12, color: '#ffaaaa', lineHeight: 1.55,
  },

  legend: { display: 'flex', flexDirection: 'column', gap: 5 },
  legendRow: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '6px 10px', borderRadius: 7,
    background: 'var(--surface)', border: '1px solid var(--border)',
  },
  legendName: { flex: 1, fontSize: 12, color: 'var(--muted2)' },
  legendSev: { fontSize: 10, padding: '2px 7px', borderRadius: 5, fontFamily: 'var(--font-head)', fontWeight: 600 },

  /* Main content */
  content: { display: 'flex', flexDirection: 'column', gap: 20 },

  statsRow: { display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 },
  statCard: {
    padding: '14px 16px', borderRadius: 10,
    background: 'var(--surface)', border: '1px solid',
    textAlign: 'center',
  },
  statVal: { fontFamily: 'var(--font-head)', fontSize: 24, fontWeight: 800, lineHeight: 1 },
  statLbl: { fontSize: 10, color: 'var(--muted)', marginTop: 5, letterSpacing: '0.08em', fontFamily: 'var(--font-head)' },

  imageCard: {
    background: 'var(--surface)', borderRadius: 12,
    border: '1px solid var(--border)', overflow: 'hidden',
  },
  tabs: {
    display: 'flex', alignItems: 'center',
    borderBottom: '1px solid var(--border)',
    padding: '0 4px',
  },
  tabBtn: {
    padding: '10px 18px', background: 'none', border: 'none',
    borderBottom: '2px solid transparent', marginBottom: -1,
    color: 'var(--muted)', fontSize: 12, cursor: 'pointer',
    fontFamily: 'var(--font-head)', fontWeight: 600, transition: 'color .2s',
  },
  tabActive: { color: 'var(--accent)', borderBottomColor: 'var(--accent)' },
  tabRight: { marginLeft: 'auto', padding: '0 14px' },
  dimText: { fontSize: 11, color: 'var(--muted)' },

  imageBox: {
    position: 'relative', minHeight: 340,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'var(--surface2)',
  },
  displayImg: { width: '100%', maxHeight: 520, objectFit: 'contain', display: 'block' },
  emptyState: { textAlign: 'center', padding: 56 },
  emptyIcon: { fontSize: 52, marginBottom: 14 },
  emptyTitle: { fontFamily: 'var(--font-head)', fontSize: 16, fontWeight: 700, color: 'var(--muted2)', marginBottom: 6 },
  emptySub: { fontSize: 13, color: 'var(--muted)' },

  loadOverlay: {
    position: 'absolute', inset: 0, zIndex: 10,
    background: 'rgba(7,9,13,0.88)', backdropFilter: 'blur(6px)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  loadBox: { textAlign: 'center' },
  bigSpin: {
    width: 48, height: 48, borderRadius: '50%',
    border: '3px solid var(--border2)', borderTopColor: 'var(--accent)',
    animation: 'spin 0.8s linear infinite', margin: '0 auto 18px',
  },
  loadTitle: { fontFamily: 'var(--font-head)', fontSize: 16, fontWeight: 700, color: 'var(--text)', marginBottom: 5 },
  loadSub: { fontSize: 12, color: 'var(--muted)', marginBottom: 18 },
  loadBar: { width: 160, height: 3, background: 'var(--border2)', borderRadius: 2, margin: '0 auto', overflow: 'hidden' },
  loadBarFill: { height: '100%', width: '60%', background: 'var(--accent)', borderRadius: 2, animation: 'pulse 1.2s infinite' },

  detCard: {
    background: 'var(--surface)', borderRadius: 12,
    border: '1px solid var(--border)', padding: 20,
    display: 'flex', flexDirection: 'column', gap: 14,
  },
  detCardHeader: { display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
  totalBadge: {
    padding: '3px 11px', borderRadius: 20, fontSize: 11, fontWeight: 700,
    background: 'rgba(0,229,255,0.08)', border: '1px solid rgba(0,229,255,0.2)',
    color: 'var(--accent)', fontFamily: 'var(--font-head)',
  },
  noDetect: {
    display: 'flex', alignItems: 'center', gap: 12,
    padding: '14px 0', fontSize: 13, color: 'var(--muted2)',
  },

  summaryRow: { display: 'flex', flexWrap: 'wrap', gap: 8 },
  summaryPill: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '4px 11px', borderRadius: 20,
    background: 'var(--surface2)', border: '1px solid var(--border)',
    fontSize: 12, color: 'var(--muted2)',
  },
  pillCount: {
    marginLeft: 4, padding: '0 6px', borderRadius: 10,
    background: 'var(--border2)', color: 'var(--accent)',
    fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-head)',
  },

  detRows: { display: 'flex', flexDirection: 'column', gap: 8 },
  detRow: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '10px 14px', borderRadius: 9,
    background: 'var(--surface2)', border: '1px solid var(--border)',
    gap: 12,
  },
  detRowLeft: { display: 'flex', alignItems: 'center', gap: 10 },
  detIcon: { fontSize: 18, flexShrink: 0 },
  detName: { fontSize: 13, fontWeight: 500, color: 'var(--text)', marginBottom: 2 },
  detBbox: { fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace', letterSpacing: '0.02em' },
  detRowRight: { display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0 },
  sevBadge: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '3px 9px', borderRadius: 6,
    fontSize: 11, fontWeight: 600, fontFamily: 'var(--font-head)',
  },
  sevDot: { width: 6, height: 6, borderRadius: '50%' },
  confWrap: { display: 'flex', alignItems: 'center', gap: 8 },
  confTrack: { width: 70, height: 4, borderRadius: 2, background: 'var(--border2)', overflow: 'hidden' },
  confFill: { height: '100%', borderRadius: 2, transition: 'width 0.4s ease' },
  confNum: { fontSize: 12, fontWeight: 700, color: 'var(--text)', minWidth: 38, textAlign: 'right', fontFamily: 'var(--font-head)' },
}