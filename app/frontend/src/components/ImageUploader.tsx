import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { Download, Image as ImageIcon, RefreshCw, Upload, Wand2, X } from "lucide-react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8001";

type Dimensions = { width: number; height: number };

interface EnhancementResult {
  success: boolean;
  message: string;
  sr_image_base64?: string;
  enhanced_image_base64?: string;
  input_dimensions?: Dimensions;
  output_dimensions?: Dimensions;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ImageUploader() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EnhancementResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const metadata = useMemo(() => {
    if (!selectedFile) return null;
    return `${selectedFile.name} (${formatSize(selectedFile.size)})`;
  }, [selectedFile]);

  const setFile = (file: File | null) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please select an image file.");
      return;
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
    setResult(null);
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFile(event.target.files?.[0] || null);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setFile(event.dataTransfer.files?.[0] || null);
  };

  const handleEnhance = async () => {
    if (!selectedFile) {
      setError("Upload an image first.");
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post<EnhancementResult>(`${BACKEND_URL}/api/enhance`, formData);
      setResult(response.data);
      if (!response.data.success) {
        setError(response.data.message || "Enhancement failed.");
      }
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(typeof detail === "string" ? detail : "Request failed. Check backend status.");
      } else {
        setError("Request failed. Check backend status.");
      }
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  const downloadImage = (base64?: string, filename = "output.png") => {
    if (!base64) return;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${base64}`;
    link.download = filename;
    link.click();
  };

  return (
    <div className="uploader">
      <div
        className={`dropzone ${previewUrl ? "has-preview" : ""}`}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        data-testid="upload-dropzone"
      >
        {!previewUrl && (
          <>
            <Upload size={28} />
            <h2>Drop an image here</h2>
            <p>or choose a local file</p>
            <label htmlFor="file-input" className="btn btn-primary" data-testid="upload-button">
              <ImageIcon size={16} />
              Browse files
            </label>
            <input
              id="file-input"
              type="file"
              accept="image/*"
              className="hidden-input"
              onChange={handleFileSelect}
            />
          </>
        )}

        {previewUrl && (
          <>
            <img src={previewUrl} alt="Input preview" className="preview" data-testid="preview-image" />
            <p className="file-meta">{metadata}</p>
            <div className="actions">
              <button className="btn btn-ghost" onClick={reset} data-testid="reset-button">
                <X size={16} />
                Remove
              </button>
              <button
                className="btn btn-primary"
                onClick={handleEnhance}
                disabled={loading}
                data-testid="enhance-button"
              >
                {loading ? <RefreshCw size={16} className="spin" /> : <Wand2 size={16} />}
                {loading ? "Processing..." : "Enhance image"}
              </button>
            </div>
          </>
        )}
      </div>

      {error && (
        <div className="message error" data-testid="error-message">
          {error}
        </div>
      )}

      {result?.success && (
        <section className="results">
          <div className="results-head">
            <h3>Results</h3>
            <button className="btn btn-ghost" onClick={reset} data-testid="new-image-button">
              Upload another
            </button>
          </div>

          <div className="result-grid">
            <article className="card">
              <h4>Input</h4>
              {previewUrl ? <img src={previewUrl} alt="Input" data-testid="result-input-image" /> : null}
              {result.input_dimensions ? (
                <span>{result.input_dimensions.width} x {result.input_dimensions.height}</span>
              ) : null}
            </article>

            <article className="card">
              <h4>Enhanced</h4>
              {result.enhanced_image_base64 ? (
                <img
                  src={`data:image/png;base64,${result.enhanced_image_base64}`}
                  alt="Enhanced"
                  data-testid="result-enhanced-image"
                />
              ) : null}
              <button
                className="btn btn-ghost"
                onClick={() => downloadImage(result.enhanced_image_base64, "enhanced.png")}
                data-testid="download-enhanced-button"
              >
                <Download size={16} />
                Download
              </button>
            </article>

            <article className="card">
              <h4>Super-resolved (4x)</h4>
              {result.sr_image_base64 ? (
                <img
                  src={`data:image/png;base64,${result.sr_image_base64}`}
                  alt="Super resolved"
                  data-testid="result-sr-image"
                />
              ) : null}
              {result.output_dimensions ? (
                <span>{result.output_dimensions.width} x {result.output_dimensions.height}</span>
              ) : null}
              <button
                className="btn btn-primary"
                onClick={() => downloadImage(result.sr_image_base64, "super_resolved.png")}
                data-testid="download-sr-button"
              >
                <Download size={16} />
                Download
              </button>
            </article>
          </div>
        </section>
      )}
    </div>
  );
}
