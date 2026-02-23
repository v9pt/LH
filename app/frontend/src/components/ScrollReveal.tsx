import { motion, type Variants } from "framer-motion";
import { type ReactNode } from "react";

interface ScrollRevealProps {
  children: ReactNode;
  className?: string;
  variant?: "fadeUp" | "fadeLeft" | "fadeRight" | "scale" | "stagger";
  delay?: number;
  duration?: number;
}

const variants: Record<string, Variants> = {
  fadeUp: {
    hidden: { opacity: 0, y: 40 },
    visible: { opacity: 1, y: 0 },
  },
  fadeLeft: {
    hidden: { opacity: 0, x: -40 },
    visible: { opacity: 1, x: 0 },
  },
  fadeRight: {
    hidden: { opacity: 0, x: 40 },
    visible: { opacity: 1, x: 0 },
  },
  scale: {
    hidden: { opacity: 0, scale: 0.92 },
    visible: { opacity: 1, scale: 1 },
  },
  stagger: {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0 },
  },
};

const ScrollReveal = ({
  children,
  className,
  variant = "fadeUp",
  delay = 0,
  duration = 0.6,
}: ScrollRevealProps) => (
  <motion.div
    className={className}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true, margin: "-80px" }}
    variants={variants[variant]}
    transition={{ duration, delay, ease: [0.25, 0.1, 0.25, 1] }}
  >
    {children}
  </motion.div>
);

export default ScrollReveal;

// For staggered children
export const StaggerContainer = ({
  children,
  className,
  staggerDelay = 0.1,
}: {
  children: ReactNode;
  className?: string;
  staggerDelay?: number;
}) => (
  <motion.div
    className={className}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true, margin: "-80px" }}
    transition={{ staggerChildren: staggerDelay }}
  >
    {children}
  </motion.div>
);

export const StaggerItem = ({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) => (
  <motion.div
    className={className}
    variants={{
      hidden: { opacity: 0, y: 24 },
      visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] } },
    }}
  >
    {children}
  </motion.div>
);
