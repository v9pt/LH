import ScrollReveal, { StaggerContainer, StaggerItem } from "@/components/ScrollReveal";

const datasets = [
  {
    name: "FFHQ",
    role: "Training",
    count: "70,000",
    desc: "High-quality 1024² faces. Degraded via ×4 bicubic ↓, Gaussian blur (σ∈[1,3]), noise (σ∈[5,25]), gamma dimming (γ∈[2.5,5]).",
  },
  {
    name: "CelebA-HQ",
    role: "Training + Test",
    count: "30,000",
    desc: "Diverse celebrity faces at 1024². Same degradation pipeline. 1K held out for evaluation.",
  },
  {
    name: "LFW",
    role: "ID Evaluation",
    count: "13,233",
    desc: "Labeled Faces in the Wild. Used to measure face-verification accuracy pre/post SR.",
  },
];

const DatasetSection = () => (
  <section className="py-24 px-6">
    <div className="max-w-5xl mx-auto">
      <ScrollReveal>
        <h2 className="text-3xl md:text-4xl font-bold mb-2">Datasets & Degradation</h2>
        <p className="text-muted-foreground mb-12">Realistic degradation simulation for training robustness.</p>
      </ScrollReveal>

      <StaggerContainer className="grid md:grid-cols-3 gap-4" staggerDelay={0.1}>
        {datasets.map((d) => (
          <StaggerItem key={d.name}>
            <div className="rounded-xl border border-border bg-card p-6 hover:border-glow transition-colors">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-bold text-lg">{d.name}</h3>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                  {d.role}
                </span>
              </div>
              <p className="text-2xl font-black text-gradient-primary mb-2">{d.count}</p>
              <p className="text-xs text-muted-foreground leading-relaxed">{d.desc}</p>
            </div>
          </StaggerItem>
        ))}
      </StaggerContainer>

      <ScrollReveal delay={0.3}>
        <div className="mt-8 rounded-xl border border-border bg-secondary/20 p-6">
          <h3 className="font-mono text-sm text-primary mb-3">Degradation Pipeline</h3>
          <pre className="font-mono text-xs text-muted-foreground leading-relaxed overflow-x-auto">
{`HR (1024×1024)
  → Bicubic ↓×4  → 256×256
  → Gaussian blur  (kernel=7, σ ~ U[1,3])
  → Additive noise (σ ~ U[5,25])
  → Gamma dimming  (γ ~ U[2.5, 5.0])
  → LR input       (256×256, dark & noisy)`}
          </pre>
        </div>
      </ScrollReveal>
    </div>
  </section>
);

export default DatasetSection;
