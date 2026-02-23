import ScrollReveal, { StaggerContainer, StaggerItem } from "@/components/ScrollReveal";

const metrics = [
  { name: "PSNR ↑", baseline: "22.4 dB", ours: "28.7 dB", delta: "+6.3" },
  { name: "SSIM ↑", baseline: "0.61", ours: "0.87", delta: "+0.26" },
  { name: "LPIPS ↓", baseline: "0.42", ours: "0.14", delta: "−0.28" },
  { name: "Face Acc (LFW) ↑", baseline: "34.2%", ours: "91.6%", delta: "+57.4%" },
];

const MetricsSection = () => (
  <section className="py-24 px-6 bg-card/50">
    <div className="max-w-5xl mx-auto">
      <ScrollReveal>
        <h2 className="text-3xl md:text-4xl font-bold mb-2">Evaluation Metrics</h2>
        <p className="text-muted-foreground mb-12">
          Benchmarked on CelebA-HQ test set (1K images) with ×4 SR from simulated low-light LR.
        </p>
      </ScrollReveal>

      <StaggerContainer className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4" staggerDelay={0.1}>
        {metrics.map((m) => (
          <StaggerItem key={m.name}>
            <div className="rounded-xl border border-border bg-secondary/20 p-6 text-center glow-primary">
              <p className="text-xs font-mono text-muted-foreground mb-3">{m.name}</p>
              <p className="text-3xl font-black text-gradient-primary mb-1">{m.ours}</p>
              <p className="text-xs text-muted-foreground">
                Baseline: {m.baseline}{" "}
                <span className="text-primary font-semibold">({m.delta})</span>
              </p>
            </div>
          </StaggerItem>
        ))}
      </StaggerContainer>

      <ScrollReveal delay={0.2}>
        <div className="mt-16 rounded-xl border border-border bg-card p-8">
          <h3 className="text-lg font-semibold mb-6">Ablation Study</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-muted-foreground text-left">
                  <th className="pb-3 font-mono text-xs">Configuration</th>
                  <th className="pb-3 font-mono text-xs text-center">PSNR</th>
                  <th className="pb-3 font-mono text-xs text-center">SSIM</th>
                  <th className="pb-3 font-mono text-xs text-center">Face Acc</th>
                </tr>
              </thead>
              <tbody className="font-mono text-xs">
                {[
                  ["SR only (no enhancer)", "25.1", "0.78", "62.3%"],
                  ["+ Zero-DCE", "27.2", "0.83", "74.8%"],
                  ["+ Identity Loss", "27.9", "0.85", "88.1%"],
                  ["+ Landmark Loss (Full)", "28.7", "0.87", "91.6%"],
                ].map(([cfg, p, s, f]) => (
                  <tr key={cfg} className="border-b border-border/50">
                    <td className="py-3 text-foreground">{cfg}</td>
                    <td className="py-3 text-center text-muted-foreground">{p}</td>
                    <td className="py-3 text-center text-muted-foreground">{s}</td>
                    <td className="py-3 text-center text-primary font-semibold">{f}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </ScrollReveal>
    </div>
  </section>
);

export default MetricsSection;
