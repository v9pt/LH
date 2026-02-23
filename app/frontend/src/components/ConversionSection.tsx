import { useState, useCallback, useRef } from "react";
import { Upload, ImagePlus, ArrowRight, Download, X, Loader2, ZoomIn } from "lucide-react";
import { Button } from "@/components/ui/button";
import ScrollReveal from "@/components/ScrollReveal";
import { motion, AnimatePresence } from "framer-motion";

interface ConvertedImage {
  id: string;
  originalUrl: string;
  originalName: string;
  enhancedUrl: string;
  status: "processing" | "done";
}

const ConversionSection = () => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [images, setImages] = useState<ConvertedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<ConvertedImage | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const simulateConversion = (file: File) => {
    const id = crypto.randomUUID();
    const originalUrl = URL.createObjectURL(file);
    const entry: ConvertedImage = { id, originalUrl, originalName: file.name, enhancedUrl: "", status: "processing" };
    setImages((prev) => [entry, ...prev]);

    setTimeout(() => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const scale = 4;
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        const ctx = canvas.getContext("2d")!;
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        ctx.globalCompositeOperation = "screen";
        ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.globalCompositeOperation = "overlay";
        ctx.fillStyle = "rgba(128, 128, 128, 0.15)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.globalCompositeOperation = "source-over";
        const enhancedUrl = canvas.toDataURL("image/png");
        setImages((prev) => prev.map((item) => (item.id === id ? { ...item, enhancedUrl, status: "done" } : item)));
      };
      img.src = originalUrl;
    }, 1500 + Math.random() * 2000);
  };

  const handleFiles = useCallback((files: FileList | File[]) => {
    Array.from(files).filter((f) => f.type.startsWith("image/")).forEach(simulateConversion);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragOver(false); handleFiles(e.dataTransfer.files); }, [handleFiles]);
  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragOver(true); };
  const onDragLeave = () => setIsDragOver(false);
  const removeImage = (id: string) => { setImages((prev) => prev.filter((img) => img.id !== id)); if (selectedImage?.id === id) setSelectedImage(null); };
  const downloadImage = (img: ConvertedImage) => { const a = document.createElement("a"); a.href = img.enhancedUrl; a.download = `enhanced_${img.originalName}`; a.click(); };

  return (
    <section id="convert" className="py-24 px-6 relative">
      <div className="absolute inset-0 bg-grid opacity-[0.03]" />
      <div className="max-w-6xl mx-auto relative z-10">
        <ScrollReveal>
          <div className="text-center mb-16">
            <span className="font-mono text-sm tracking-widest text-primary/80 uppercase">Interactive Demo</span>
            <h2 className="text-4xl md:text-5xl font-bold mt-3 mb-4 text-foreground">
              LR → HR <span className="text-gradient-primary">Conversion</span>
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto text-lg">
              Drop your low-resolution or low-light face images below. Our pipeline simulates enhancement, ×4 super-resolution, and identity-preserving hallucination.
            </p>
          </div>
        </ScrollReveal>

        <ScrollReveal variant="scale" delay={0.15}>
          <div
            onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={`relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center py-16 px-8 mb-12 ${isDragOver ? "border-primary bg-primary/5 scale-[1.01] shadow-lg shadow-primary/10" : "border-border hover:border-primary/40 hover:bg-card/50"}`}
          >
            <div className={`p-4 rounded-full mb-4 transition-colors ${isDragOver ? "bg-primary/20" : "bg-muted"}`}>
              {isDragOver ? <ImagePlus className="w-8 h-8 text-primary" /> : <Upload className="w-8 h-8 text-muted-foreground" />}
            </div>
            <p className="text-foreground font-semibold text-lg mb-1">{isDragOver ? "Release to upload" : "Drag & drop face images here"}</p>
            <p className="text-muted-foreground text-sm">or click to browse · PNG, JPG, WebP supported</p>
            <input ref={fileInputRef} type="file" accept="image/*" multiple className="hidden" onChange={(e) => e.target.files && handleFiles(e.target.files)} />
          </div>
        </ScrollReveal>

        <AnimatePresence>
          {images.length > 0 && (
            <motion.div className="space-y-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
              <h3 className="text-xl font-semibold text-foreground font-mono">
                Results <span className="text-muted-foreground text-sm font-normal">({images.length} image{images.length !== 1 ? "s" : ""})</span>
              </h3>
              <div className="grid gap-6">
                <AnimatePresence>
                  {images.map((img) => (
                    <motion.div
                      key={img.id}
                      layout
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ duration: 0.35 }}
                      className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-4 md:p-6"
                    >
                      <div className="flex flex-col md:flex-row items-center gap-4 md:gap-6">
                        <div className="flex-1 w-full">
                          <p className="text-xs font-mono text-muted-foreground mb-2 uppercase tracking-wider">Input (LR / Low-Light)</p>
                          <div className="relative aspect-square rounded-lg overflow-hidden bg-muted/30 border border-border">
                            <img src={img.originalUrl} alt="Original" className="w-full h-full object-cover opacity-70 blur-[0.5px]" style={{ filter: "brightness(0.6) contrast(0.9) blur(0.5px)" }} />
                            <div className="absolute bottom-2 left-2 bg-background/80 backdrop-blur-sm text-xs font-mono px-2 py-1 rounded text-muted-foreground">{img.originalName}</div>
                          </div>
                        </div>
                        <div className="flex-shrink-0">
                          {img.status === "processing" ? <Loader2 className="w-6 h-6 text-primary animate-spin" /> : <ArrowRight className="w-6 h-6 text-primary" />}
                        </div>
                        <div className="flex-1 w-full">
                          <p className="text-xs font-mono text-muted-foreground mb-2 uppercase tracking-wider">Output (SR ×4 Enhanced)</p>
                          <div className="relative aspect-square rounded-lg overflow-hidden bg-muted/30 border border-border">
                            {img.status === "processing" ? (
                              <div className="w-full h-full flex flex-col items-center justify-center gap-3">
                                <div className="w-12 h-12 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
                                <p className="text-xs font-mono text-muted-foreground animate-pulse">Processing pipeline...</p>
                                <div className="flex gap-1">
                                  {["Zero-DCE", "RRDB ×4", "ArcFace"].map((step, i) => (
                                    <span key={step} className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-primary/10 text-primary/60" style={{ animationDelay: `${i * 0.5}s` }}>{step}</span>
                                  ))}
                                </div>
                              </div>
                            ) : (
                              <>
                                <img src={img.enhancedUrl} alt="Enhanced" className="w-full h-full object-cover" />
                                <div className="absolute bottom-2 left-2 bg-background/80 backdrop-blur-sm text-xs font-mono px-2 py-1 rounded text-primary">×4 SR · Enhanced</div>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                      {img.status === "done" && (
                        <div className="flex justify-end gap-2 mt-4 pt-4 border-t border-border/50">
                          <Button variant="ghost" size="sm" onClick={() => setSelectedImage(img)} className="text-muted-foreground hover:text-foreground"><ZoomIn className="w-4 h-4 mr-1" /> Compare</Button>
                          <Button variant="ghost" size="sm" onClick={() => downloadImage(img)} className="text-muted-foreground hover:text-foreground"><Download className="w-4 h-4 mr-1" /> Download</Button>
                          <Button variant="ghost" size="sm" onClick={() => removeImage(img.id)} className="text-muted-foreground hover:text-destructive"><X className="w-4 h-4 mr-1" /> Remove</Button>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {selectedImage && (
            <motion.div
              className="fixed inset-0 z-50 bg-background/90 backdrop-blur-sm flex items-center justify-center p-6"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              onClick={() => setSelectedImage(null)}
            >
              <motion.div
                className="max-w-5xl w-full bg-card border border-border rounded-2xl p-6"
                initial={{ scale: 0.92, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.92, opacity: 0 }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-mono font-semibold text-foreground">Side-by-Side Comparison</h3>
                  <Button variant="ghost" size="icon" onClick={() => setSelectedImage(null)}><X className="w-4 h-4" /></Button>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs font-mono text-muted-foreground mb-2 uppercase">Input</p>
                    <img src={selectedImage.originalUrl} alt="Original" className="w-full rounded-lg border border-border" style={{ filter: "brightness(0.6) contrast(0.9)" }} />
                  </div>
                  <div>
                    <p className="text-xs font-mono text-primary mb-2 uppercase">Enhanced SR ×4</p>
                    <img src={selectedImage.enhancedUrl} alt="Enhanced" className="w-full rounded-lg border border-primary/30" />
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        <p className="text-center text-xs text-muted-foreground mt-8 font-mono">
          ⚠ This is a browser-based simulation. Real inference requires the trained PyTorch model served via an API endpoint.
        </p>
      </div>
    </section>
  );
};

export default ConversionSection;
