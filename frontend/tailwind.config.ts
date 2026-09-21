import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0b0f17",
          900: "#111725",
          800: "#1a2233",
          700: "#273248",
          600: "#3b4a66",
          400: "#8494b3",
          200: "#c9d3e6",
        },
        accent: {
          DEFAULT: "#5b8def",
          soft: "#7ba4f5",
          dim: "#31427a",
        },
        positive: "#3fb984",
        negative: "#e2685f",
        caution: "#e0a44b",
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "Segoe UI", "Inter", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
