export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--border))",
        ring: "hsl(var(--primary))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: "hsl(var(--card, var(--background)))",
        "card-foreground": "hsl(var(--foreground))",
        primary: "hsl(var(--primary))",
        "primary-foreground": "hsl(var(--primary-foreground, var(--foreground)))",
        secondary: "hsl(var(--muted))",
        "secondary-foreground": "hsl(var(--foreground))",
        accent: "hsl(var(--primary))",
        "accent-foreground": "hsl(var(--foreground))",
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
      },
    },
  },
  plugins: [],
}
