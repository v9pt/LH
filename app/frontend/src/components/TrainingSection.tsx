import ScrollReveal, { StaggerContainer, StaggerItem } from "@/components/ScrollReveal";

const phases = [
  {
    phase: "Phase A",
    title: "Low-Light Enhancement Pretraining",
    desc: "Train Zero-DCE on unlabeled dark face crops. Losses: spatial consistency, exposure control, color constancy, illumination smoothness.",
    details: ["Dataset: FFHQ darkened via gamma (γ ∈ [2.5, 5.0])", "Epochs: 200 · LR: 1e-4 · Adam"],
    status: "complete",
  },
  {
    phase: "Phase B",
    title: "SR GAN Training",
    desc: "Train RRDB generator + PatchGAN discriminator on paired enhanced-LR → HR data.",
    details: ["Losses: L1 + VGG-19 perceptual + adversarial", "Batch: 16 · LR: 2e-4 → 1e-4 cosine decay", "400K iterations"],
    status: "complete",
  },
  {
    phase: "Phase C",
    title: "Identity & Landmark Fine-tuning",
    desc: "End-to-end fine-tuning with frozen ArcFace & landmark detector. Lower LR to preserve texture quality.",
    details: ["Add ℒ_id (λ₄=0.1) + ℒ_lmk (λ₅=0.01)", "LR: 5e-5 · 100K iterations", "Freeze enhancer; tune generator only"],
    status: "active",
  },
];

const statusColors: Record<string, string> = {
  complete: "bg-success/20 text-success border-success/30",
  active: "bg-primary/20 text-primary border-primary/30 animate-pulse-glow",
};

const TrainingSection = () => (
  <section className="py-24 px-6">
    <div className="max-w-5xl mx-auto">
      <ScrollReveal>
        <h2 className="text-3xl md:text-4xl font-bold mb-2">Training Pipeline</h2>
        <p className="text-muted-foreground mb-12">Three-phase curriculum for stable convergence.</p>
      </ScrollReveal>

      <div className="relative">
        <div className="absolute left-[19px] top-4 bottom-4 w-px bg-border hidden md:block" />

        <StaggerContainer className="space-y-8" staggerDelay={0.15}>
          {phases.map((p) => (
            <StaggerItem key={p.phase}>
              <div className="flex gap-6">
                <div className="shrink-0 relative z-10 mt-1">
                  <div className={`w-10 h-10 rounded-full border flex items-center justify-center text-xs font-mono font-bold ${statusColors[p.status]}`}>
                    {p.phase.split(" ")[1]}
                  </div>
                </div>
                <div className="rounded-xl border border-border bg-card p-6 flex-1">
                  <p className="text-xs font-mono text-muted-foreground mb-1">{p.phase}</p>
                  <h3 className="text-lg font-semibold mb-2">{p.title}</h3>
                  <p className="text-sm text-muted-foreground mb-4">{p.desc}</p>
                  <div className="flex flex-wrap gap-2">
                    {p.details.map((d) => (
                      <span key={d} className="text-xs font-mono bg-secondary px-2.5 py-1 rounded border border-border text-secondary-foreground">
                        {d}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </StaggerItem>
          ))}
        </StaggerContainer>
      </div>
    </div>
  </section>
);

export default TrainingSection;
