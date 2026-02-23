import ScrollReveal, { StaggerContainer, StaggerItem } from "@/components/ScrollReveal";

const blocks = [
  { label: "Dark LR Input", sub: "16×16 – 32×32", color: "border-destructive/40 bg-destructive/5" },
  { label: "Zero-DCE\nEnhancer", sub: "Light Curve Est.", color: "border-warning/40 bg-warning/5", accent: "warning" },
  { label: "Enhanced LR", sub: "Denoised + Bright", color: "border-primary/40 bg-primary/5" },
  { label: "RRDB Generator\n(GFPGAN-style)", sub: "×4 Upscale", color: "border-primary/40 bg-primary/5" },
  { label: "SR Output", sub: "128×128", color: "border-success/40 bg-success/5" },
];

const losses = [
  { name: "ℒ_pixel", desc: "L1 reconstruction", target: 3 },
  { name: "ℒ_perceptual", desc: "VGG feature matching", target: 3 },
  { name: "ℒ_adv", desc: "PatchGAN adversarial", target: 3 },
  { name: "ℒ_id", desc: "ArcFace cosine sim.", target: 4 },
  { name: "ℒ_lmk", desc: "68-pt landmark L2", target: 4 },
];

const Arrow = () => (
  <div className="flex items-center justify-center shrink-0">
    <svg width="32" height="16" viewBox="0 0 32 16" className="text-primary/50">
      <path d="M0 8h28m0 0l-6-6m6 6l-6 6" stroke="currentColor" strokeWidth="1.5" fill="none" />
    </svg>
  </div>
);

const ArchitectureSection = () => (
  <section className="py-24 px-6">
    <div className="max-w-6xl mx-auto">
      <ScrollReveal>
        <h2 className="text-3xl md:text-4xl font-bold mb-2">System Architecture</h2>
        <p className="text-muted-foreground mb-12 max-w-xl">
          End-to-end pipeline from degraded low-light input to identity-preserved high-resolution face.
        </p>
      </ScrollReveal>

      <ScrollReveal variant="scale" delay={0.1}>
        <div className="flex flex-wrap items-center justify-center gap-3 mb-16">
          {blocks.map((b, i) => (
            <div key={b.label} className="contents">
              {i > 0 && <Arrow />}
              <div className={`rounded-lg border p-4 w-40 text-center ${b.color}`}>
                <p className="font-semibold text-sm whitespace-pre-line text-foreground">{b.label}</p>
                <p className="text-xs text-muted-foreground mt-1">{b.sub}</p>
              </div>
            </div>
          ))}
        </div>
      </ScrollReveal>

      <StaggerContainer className="grid md:grid-cols-5 gap-3" staggerDelay={0.08}>
        {losses.map((l) => (
          <StaggerItem key={l.name}>
            <div className="rounded-lg border border-border bg-card p-4">
              <p className="font-mono text-primary text-sm font-bold">{l.name}</p>
              <p className="text-xs text-muted-foreground mt-1">{l.desc}</p>
            </div>
          </StaggerItem>
        ))}
      </StaggerContainer>
    </div>
  </section>
);

export default ArchitectureSection;
