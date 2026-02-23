import HeroSection from "@/components/HeroSection"
import ArchitectureSection from "@/components/ArchitectureSection"
import LossSection from "@/components/LossSection"
import TrainingSection from "@/components/TrainingSection"
import DatasetSection from "@/components/DatasetSection"
import MetricsSection from "@/components/MetricsSection"
import FooterSection from "@/components/FooterSection"
import ImageUploader from "@/components/ImageUploader"

const Index = () => {
  return (
    <main className="min-h-screen bg-black text-white">
      <HeroSection />
      <ImageUploader />
      <ArchitectureSection />
      <LossSection />
      <TrainingSection />
      <DatasetSection />
      <MetricsSection />
      <FooterSection />
    </main>
  )
}

export default Index
