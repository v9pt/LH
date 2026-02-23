import { ImagePlus, Sparkles, Cpu } from "lucide-react";

const HeroSection = () => {
  return (
    <section className="relative overflow-hidden px-6 py-24">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_10%,rgba(99,102,241,0.18),transparent_45%),radial-gradient(circle_at_85%_10%,rgba(16,185,129,0.18),transparent_40%)]" />
      <div className="relative mx-auto max-w-5xl text-center">
        <p className="inline-block rounded-full border border-border bg-secondary/40 px-4 py-1 text-xs uppercase tracking-[0.14em] text-muted-foreground">
          Face Hallucination Pipeline
        </p>
        <h1 className="mt-6 text-4xl font-bold leading-tight md:text-6xl">
          Restore details in low-light faces with AI enhancement
        </h1>
        <p className="mx-auto mt-5 max-w-3xl text-muted-foreground md:text-lg">
          This frontend showcases your enhancement plus 4x super-resolution workflow in a single interface.
        </p>

        <div className="mt-10 grid gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-border bg-card p-4">
            <ImagePlus className="mx-auto mb-2 h-5 w-5 text-primary" />
            <p className="text-sm text-foreground">Low-light image input</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-4">
            <Sparkles className="mx-auto mb-2 h-5 w-5 text-primary" />
            <p className="text-sm text-foreground">Enhancement + denoise</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-4">
            <Cpu className="mx-auto mb-2 h-5 w-5 text-primary" />
            <p className="text-sm text-foreground">4x super-resolution output</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
