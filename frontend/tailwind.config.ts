import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        void: "#07070b",
        panel: "#111118",
        "panel-light": "#1a1a26",
        accent: "#6c5ce7",
        secondary: "#2d2d3f",
        success: "#00d4aa",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "glow": "glow 2s ease-in-out infinite alternate",
        "fade-in": "fadeIn 0.4s ease-out",
        "subtle-pulse": "subtlePulse 2s ease-in-out infinite",
        "slide-up": "slideUp 0.4s ease-out",
      },
      keyframes: {
        glow: {
          "0%": { boxShadow: "0 0 5px rgba(108, 92, 231, 0.3)" },
          "100%": { boxShadow: "0 0 20px rgba(108, 92, 231, 0.5)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        subtlePulse: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.6" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
