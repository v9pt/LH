import { Badge } from "@/components/ui/badge";

const HeroSection = () => (
  <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
    <div className="absolute inset-0 bg-grid opacity-30" />
    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-primary/5 blur-[120px]" />
    <div className="absolute top-1/3 right-1/4 w-[300px] h-[300px] rounded-full bg-accent/5 blur-[100px]" />

    <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
      <Badge variant="outline" className="mb-6 border-glow text-primary font-mono text-xs tracking-wider px-4 py-1.5">
        2026 · AI RESEARCH
      </Badge>
      <h1 className="text-5xl md:text-7xl font-black tracking-tight leading-[1.05] mb-6">
        <span className="text-gradient-primary">Low-Light Face</span>
        <br />
        <span className="text-foreground">Hallucination</span>
      </h1>
      <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
        Identity-preserving super-resolution for degraded face images —
        combining Zero-DCE enhancement, GAN-based upscaling, and ArcFace constraints.
      </p>
      <div className="flex flex-wrap justify-center gap-3 text-sm font-mono text-muted-foreground">
        {["Zero-DCE", "RRDB / ESRGAN", "ArcFace ID Loss", "Landmark Consistency", "×4 SR"].map((t) => (
          <span key={t} className="px-3 py-1.5 rounded-md bg-secondary border border-border">
            {t}
          </span>
        ))}
      </div>
    </div>
  </section>
);

export default HeroSection;

