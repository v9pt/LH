import ScrollReveal from "@/components/ScrollReveal";

const FooterSection = () => (
  <footer className="py-16 px-6 border-t border-border">
    <ScrollReveal>
      <div className="max-w-5xl mx-auto text-center">
        <p className="text-sm text-muted-foreground mb-2">
          Low-Light Face Hallucination with Identity Preservation · 2026
        </p>
        <p className="text-xs text-muted-foreground/60 font-mono">
          Research prototype · FFHQ / CelebA-HQ / LFW · ArcFace + Zero-DCE + RRDB
        </p>
      </div>
    </ScrollReveal>
  </footer>
);

export default FooterSection;
