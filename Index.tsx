import HeroSection from "@/components/HeroSection";
import ArchitectureSection from "@/components/ArchitectureSection";
import LossSection from "@/components/LossSection";
import TrainingSection from "@/components/TrainingSection";
import DatasetSection from "@/components/DatasetSection";
import MetricsSection from "@/components/MetricsSection";
import FooterSection from "@/components/FooterSection";

const Index = () => (
  <div className="min-h-screen bg-background">
    <HeroSection />
    <ArchitectureSection />
    <LossSection />
    <TrainingSection />
    <DatasetSection />
    <MetricsSection />
    <FooterSection />
  </div>
);

export default Index;


