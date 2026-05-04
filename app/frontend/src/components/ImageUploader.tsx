import { useEffect, useMemo, useState, useRef } from "react";
import axios from "axios";
import { Download, Image as ImageIcon, RefreshCw, Upload, Wand2, X, Zap, Eye, Brain, Layers } from "lucide-react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8001";

// ── Types ──────────────────────────────────────────────────────────────────
interface StageResult {
  id: string | number;
  name: string;
  paper: string;
  image: string;   // base64 PNG
  active: boolean;
}

interface EnhanceResponse {
  status: string;
  processing_time_ms: number;
  darkness_factor: number;
  darkness_interpretation: string;
  blur_severity: number;
  blur_corrected: boolean;
  input_image: string;
  deblurred_image: string;
  enhanced_image: string;
  lcr_output: string;
  final_image: string;
  input_size: number[];
  output_size: number[];
}

// ── Helper ─────────────────────────────────────────────────────────────────
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function downloadB64(b64: string, filename: string) {
  const a = document.createElement("a");
  a.href = `data:image/png;base64,${b64}`;
  a.download = filename;
  a.click();
}

// ── Darkness Gauge ─────────────────────────────────────────────────────────
function DarknessGauge({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value * 100);
  const color =
    value < 0.25 ? "#22c55e" : value < 0.55 ? "#f59e0b" : value < 0.78 ? "#f97316" : "#ef4444";
  return (
    <div className="gauge-wrap" data-testid="darkness-gauge">
      <div className="gauge-label">Darkness Factor (ANFIS)</div>
      <div className="gauge-bar-bg">
        <div className="gauge-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="gauge-values">
        <span style={{ color }}>{(value).toFixed(3)}</span>
        <span className="gauge-interp">{label}</span>
      </div>
    </div>
  );
}

// ── Pipeline Flow ──────────────────────────────────────────────────────────
const PIPELINE_STAGES = [
  { id: 1,   name: "ANFIS Darkness Estimation", paper: "Paper 3", icon: Brain },
  { id: 2,   name: "Blur Correction",           paper: "Paper 5", icon: Eye },
  { id: "3a",name: "Zero-DCE Enhancement",      paper: "Paper 2", icon: Zap },
  { id: "3b",name: "ANFIS-LCR Hallucination",   paper: "Paper 1", icon: Layers },
  { id: 4,   name: "Regression Blending",       paper: "Paper 4", icon: Brain },
  { id: 5,   name: "RRDB Refinement",           paper: "ESRGAN",  icon: Wand2 },
];

function PipelineVisualizer({ activeStage }: { activeStage: number }) {
  return (
    <div className="pipeline-vis" data-testid="pipeline-visualizer">
      {PIPELINE_STAGES.map((s, i) => {
        const Icon = s.icon;
        const done = i < activeStage;
        const active = i === activeStage;
        return (
          <div key={String(s.id)} className="pipeline-step-wrap">
            <div className={`pipeline-step ${done ? "done" : ""} ${active ? "active" : ""}`}>
              <div className="pipeline-icon"><Icon size={14} /></div>
              <div className="pipeline-text">
                <div className="pipeline-name">{s.name}</div>
                <div className="pipeline-paper">{s.paper}</div>
              </div>
              {active && <div className="pipeline-pulse" />}
            </div>
            {i < PIPELINE_STAGES.length - 1 && (
              <div className={`pipeline-arrow ${done ? "done" : ""}`}>→</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Stage Image Grid ────────────────────────────────────────────────────────
function StageGrid({ stages }: { stages: StageResult[] }) {
  const [selected, setSelected] = useState(0);
  return (
    <div className="stage-grid" data-testid="stage-grid">
      <div className="stage-tabs">
        {stages.map((s, i) => (
          <button
            key={String(s.id)}
            className={`stage-tab ${i === selected ? "active" : ""}`}
            onClick={() => setSelected(i)}
            data-testid={`stage-tab-${s.id}`}
          >
            Stage {s.id}
          </button>
        ))}
      </div>
      {stages[selected] && (
        <div className="stage-detail">
          <div className="stage-detail-header">
            <div>
              <div className="stage-detail-name">{stages[selected].name}</div>
              <div className="stage-detail-paper">{stages[selected].paper}</div>
            </div>
            <button
              className="btn-dl"
              onClick={() => downloadB64(stages[selected].image, `stage_${stages[selected].id}.png`)}
              data-testid={`download-stage-${stages[selected].id}`}
            >
              <Download size={14} /> Save
            </button>
          </div>
          <img
            src={`data:image/png;base64,${stages[selected].image}`}
            alt={stages[selected].name}
            className="stage-img"
            data-testid="stage-image"
          />
        </div>
      )}
    </div>
  );
}

// ── Before/After Slider ────────────────────────────────────────────────────
function BeforeAfterSlider({ before, after }: { before: string; after: string }) {
  const [pct, setPct] = useState(50);
  const ref = useRef<HTMLDivElement>(null);

  const onMove = (clientX: number) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const p = Math.min(Math.max(((clientX - rect.left) / rect.width) * 100, 0), 100);
    setPct(p);
  };

  return (
    <div
      ref={ref}
      className="slider-wrap"
      onMouseMove={(e) => e.buttons === 1 && onMove(e.clientX)}
      onTouchMove={(e) => onMove(e.touches[0].clientX)}
      data-testid="before-after-slider"
    >
      <img src={`data:image/png;base64,${before}`} alt="Before" className="slider-img" />
      <div className="slider-after-wrap" style={{ clipPath: `inset(0 ${100 - pct}% 0 0)` }}>
        <img src={`data:image/png;base64,${after}`} alt="After" className="slider-img" />
      </div>
      <div className="slider-handle" style={{ left: `${pct}%` }}>
        <div className="slider-line" />
        <div className="slider-knob">‹ ›</div>
      </div>
      <div className="slider-label left">Input</div>
      <div className="slider-label right">Output</div>
    </div>
  );
}

// ── Metrics Bar ─────────────────────────────────────────────────────────────
function MetricsBar({ df, blur, time, inputSize, outputSize, blurCorrected }: {
  df: number; blur: number; time: number;
  inputSize: number[]; outputSize: number[];
  blurCorrected: boolean;
}) {
  return (
    <div className="metrics-bar" data-testid="metrics-bar">
      <div className="metric-chip">
        <span className="metric-label">Darkness Factor</span>
        <span className="metric-val">{df.toFixed(3)}</span>
      </div>
      <div className="metric-chip">
        <span className="metric-label">Blur Severity</span>
        <span className="metric-val">{blur.toFixed(3)}{blurCorrected ? " ✓" : ""}</span>
      </div>
      <div className="metric-chip">
        <span className="metric-label">Input Size</span>
        <span className="metric-val">{inputSize[0]}×{inputSize[1]}</span>
      </div>
      <div className="metric-chip">
        <span className="metric-label">Output Size</span>
        <span className="metric-val">{outputSize[0]}×{outputSize[1]}</span>
      </div>
      <div className="metric-chip">
        <span className="metric-label">Processing</span>
        <span className="metric-val">{time.toFixed(0)} ms</span>
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────
export default function ImageUploader() {
  const [file, setFileState] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeStage, setActiveStage] = useState(-1);
  const [result, setResult] = useState<EnhanceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const metadata = useMemo(() =>
    file ? `${file.name} — ${formatSize(file.size)}` : null, [file]);

  const setFile = (f: File | null) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) { setError("Please select an image file."); return; }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFileState(f);
    setPreviewUrl(URL.createObjectURL(f));
    setError(null);
    setResult(null);
    setActiveStage(-1);
  };

  const handleEnhance = async () => {
    if (!file) { setError("Upload an image first."); return; }
    setLoading(true);
    setError(null);
    setActiveStage(0);

    const formData = new FormData();
    formData.append("file", file);

    // Simulate stage progression
    const stageTimer = setInterval(() => {
      setActiveStage(prev => (prev < PIPELINE_STAGES.length - 1 ? prev + 1 : prev));
    }, 400);

    try {
      const resp = await axios.post<EnhanceResponse>(`${BACKEND_URL}/api/enhance`, formData);
      clearInterval(stageTimer);
      setActiveStage(PIPELINE_STAGES.length);
      setResult(resp.data);
    } catch (err) {
      clearInterval(stageTimer);
      setActiveStage(-1);
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(typeof detail === "string" ? detail : "Backend error. Is the server running?");
      } else {
        setError("Request failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null); setFileState(null);
    setResult(null); setError(null); setActiveStage(-1);
  };

  // Build stage outputs for the grid
  const stages: StageResult[] = result ? [
    { id: "In", name: "Input (Low-light LR)", paper: "—",
      image: result.input_image, active: true },
    { id: 2,   name: "Motion Blur Corrected", paper: "Paper 5",
      image: result.deblurred_image, active: true },
    { id: "3a",name: "Zero-DCE Enhanced",     paper: "Paper 2",
      image: result.enhanced_image, active: true },
    { id: "3b",name: "ANFIS-LCR Hallucinated",paper: "Paper 1",
      image: result.lcr_output, active: true },
    { id: "Out",name: "Final HR Output",       paper: "Papers 1–5",
      image: result.final_image, active: true },
  ] : [];

  return (
    <div className="uploader-root">
      {/* Upload zone */}
      {!result && (
        <div
          className={`dropzone ${previewUrl ? "has-preview" : ""}`}
          onDrop={(e) => { e.preventDefault(); setFile(e.dataTransfer.files?.[0] || null); }}
          onDragOver={(e) => e.preventDefault()}
          data-testid="upload-dropzone"
        >
          {!previewUrl ? (
            <>
              <Upload size={32} className="drop-icon" />
              <h2 className="drop-title">Drop a dark face image here</h2>
              <p className="drop-sub">or choose a local file</p>
              <label htmlFor="file-input" className="btn btn-primary" data-testid="upload-button">
                <ImageIcon size={16} /> Browse files
              </label>
              <input id="file-input" type="file" accept="image/*"
                className="hidden-input" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            </>
          ) : (
            <>
              <img src={previewUrl} alt="Preview" className="preview" data-testid="preview-image" />
              <p className="file-meta">{metadata}</p>
              <div className="actions">
                <button className="btn btn-ghost" onClick={reset} data-testid="reset-button">
                  <X size={15} /> Remove
                </button>
                <button className="btn btn-primary" onClick={handleEnhance}
                  disabled={loading} data-testid="enhance-button">
                  {loading ? <RefreshCw size={15} className="spin" /> : <Wand2 size={15} />}
                  {loading ? "Processing…" : "Enhance with ANFIS"}
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="message error" data-testid="error-message">{error}</div>
      )}

      {/* Pipeline progress */}
      {loading && (
        <div className="pipeline-progress-wrap">
          <p className="pipeline-progress-title">Running ANFIS pipeline…</p>
          <PipelineVisualizer activeStage={activeStage} />
        </div>
      )}

      {/* Results */}
      {result && (
        <section className="results-section" data-testid="results-section">
          <div className="results-header">
            <h3 className="results-title">Enhancement Results</h3>
            <button className="btn btn-ghost" onClick={reset} data-testid="new-image-button">
              <Upload size={14} /> New Image
            </button>
          </div>

          {/* Metrics */}
          <MetricsBar
            df={result.darkness_factor}
            blur={result.blur_severity}
            time={result.processing_time_ms}
            inputSize={result.input_size}
            outputSize={result.output_size}
            blurCorrected={result.blur_corrected}
          />

          {/* Darkness gauge */}
          <DarknessGauge value={result.darkness_factor} label={result.darkness_interpretation} />

          {/* Before/after slider */}
          <h4 className="section-label">Before / After</h4>
          <BeforeAfterSlider before={result.input_image} after={result.final_image} />

          {/* Stage-by-stage grid */}
          <h4 className="section-label">Pipeline Stage Outputs</h4>
          <StageGrid stages={stages} />

          {/* Download final */}
          <div className="download-row">
            <button
              className="btn btn-primary"
              onClick={() => downloadB64(result.final_image, "anfis_sr_output.png")}
              data-testid="download-final-button"
            >
              <Download size={15} /> Download Final (4×SR)
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
