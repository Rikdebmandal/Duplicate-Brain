import type { Config } from "tailwindcss";

function withOpacity(variableName: string) {
  return `color-mix(in srgb, var(${variableName}) calc(100% * <alpha-value>), transparent)`;
}

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: withOpacity("--color-ink-950"),
          900: withOpacity("--color-ink-900"),
          800: withOpacity("--color-ink-800"),
          700: withOpacity("--color-ink-700"),
          600: withOpacity("--color-ink-600"),
          400: withOpacity("--color-ink-400"),
          200: withOpacity("--color-ink-200"),
        },
        accent: {
          DEFAULT: withOpacity("--color-accent"),
          soft: withOpacity("--color-accent-soft"),
          dim: withOpacity("--color-accent-dim"),
        },
        positive: withOpacity("--color-positive"),
        negative: withOpacity("--color-negative"),
        caution: withOpacity("--color-caution"),
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
