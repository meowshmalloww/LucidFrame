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
        void: "#09090B",
        surface: "#111113",
        elevated: "#18181B",
        accent: {
          DEFAULT: "#FFFFFF",
          hover: "#E4E4E7",
        },
        success: "#22C55E",
        warning: "#F59E0B",
        danger: "#EF4444",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      spacing: {
        "sidebar": "64px",
        "sidebar-expanded": "240px",
      },
      borderRadius: {
        "card": "12px",
        "btn": "6px",
        "dialog": "14px",
      },
      animation: {
        "fade-in": "fadeIn 0.2s ease-out",
        "fade-out": "fadeOut 0.15s ease-in",
        "slide-up": "slideUp 0.25s cubic-bezier(0.32, 0.72, 0, 1)",
        "slide-down": "slideDown 0.25s cubic-bezier(0.32, 0.72, 0, 1)",
        "slide-right": "slideRight 0.25s cubic-bezier(0.32, 0.72, 0, 1)",
        "scale-in": "scaleIn 0.2s cubic-bezier(0.32, 0.72, 0, 1)",
        "progress": "progress 1.5s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        fadeOut: {
          "0%": { opacity: "1" },
          "100%": { opacity: "0" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideDown: {
          "0%": { opacity: "0", transform: "translateY(-8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideRight: {
          "0%": { opacity: "0", transform: "translateX(-8px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        scaleIn: {
          "0%": { opacity: "0", transform: "scale(0.96)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        progress: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
