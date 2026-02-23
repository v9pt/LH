import ScrollReveal, { StaggerContainer, StaggerItem } from "@/components/ScrollReveal";

const equations = [
  {
    title: "Total Loss",
    eq: "ℒ_total = λ₁ℒ_pixel + λ₂ℒ_percep + λ₃ℒ_adv + λ₄ℒ_id + λ₅ℒ_lmk",
  },
  {
    title: "Identity Loss (ArcFace)",
    eq: "ℒ_id = 1 − cos(F(I_SR), F(I_HR))",
    note: "F(·) is a frozen ArcFace encoder; cosine similarity between 512-d embeddings.",
  },
  {
    title: "Landmark Loss",
    eq: "ℒ_lmk = (1/68) Σᵢ ‖pᵢ(I_SR) − pᵢ(I_HR)‖₂",
    note: "68 facial keypoints detected via Dlib / RetinaFace.",
  },
  {
    title: "Zero-DCE Light-Enhancement Curve",
    eq: "Lₑ(x) = x + αₙ · x · (1 − x)   (iterated n times)",
    note: "Parameter-free enhancement via learned pixel-wise curves.",
  },
];

const LossSection = () => (
  <section className="py-24 px-6 bg-card/50">
    <div className="max-w-4xl mx-auto">
      <ScrollReveal>
        <h2 className="text-3xl md:text-4xl font-bold mb-2">Loss Formulations</h2>
        <p className="text-muted-foreground mb-12">
          Multi-objective optimization balancing fidelity, perceptual quality, and identity.
        </p>
      </ScrollReveal>
      <StaggerContainer className="space-y-6" staggerDelay={0.12}>
        {equations.map((eq) => (
          <StaggerItem key={eq.title}>
            <div className="rounded-xl border border-border bg-secondary/30 p-6">
              <h3 className="text-sm font-semibold text-primary mb-3">{eq.title}</h3>
              <pre className="font-mono text-base md:text-lg text-foreground overflow-x-auto">{eq.eq}</pre>
              {eq.note && <p className="text-xs text-muted-foreground mt-3">{eq.note}</p>}
            </div>
          </StaggerItem>
        ))}
      </StaggerContainer>
    </div>
  </section>
);

export default LossSection;
