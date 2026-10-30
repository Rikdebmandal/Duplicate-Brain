"use client";

import { useEffect, useState } from "react";

export type Theme = "system" | "light" | "dark";

export function ThemeSwitcher({ className = "" }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>("system");
  const [systemIsDark, setSystemIsDark] = useState<boolean>(true);
  const [mounted, setMounted] = useState<boolean>(false);

  useEffect(() => {
    setMounted(true);

    // Initialise from localStorage
    try {
      const saved = localStorage.getItem("color-scheme") as Theme | null;
      if (saved === "light" || saved === "dark") {
        setTheme(saved);
      } else {
        setTheme("system");
      }
    } catch {
      setTheme("system");
    }

    // Media query to detect and track system preference
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    setSystemIsDark(media.matches);

    const onMediaChange = (e: MediaQueryListEvent) => {
      setSystemIsDark(e.matches);
    };

    // Storage listener to sync across tabs
    const onStorage = (e: StorageEvent) => {
      if (e.key === "color-scheme") {
        const val = e.newValue as Theme | null;
        if (val === "light" || val === "dark") {
          setTheme(val);
          applyTheme(val);
        } else {
          setTheme("system");
          applyTheme("system");
        }
      }
    };

    media.addEventListener("change", onMediaChange);
    window.addEventListener("storage", onStorage);

    return () => {
      media.removeEventListener("change", onMediaChange);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  function applyTheme(target: Theme) {
    const meta = document.querySelector('meta[name="color-scheme"]') as HTMLMetaElement | null;
    if (target === "system") {
      if (meta) meta.content = "light dark";
      delete document.documentElement.dataset.theme;
      try {
        localStorage.removeItem("color-scheme");
      } catch {}
    } else {
      if (meta) meta.content = target;
      document.documentElement.dataset.theme = target;
      try {
        localStorage.setItem("color-scheme", target);
      } catch {}
    }
  }

  function handleSelect(nextTheme: Theme) {
    setTheme(nextTheme);
    applyTheme(nextTheme);
  }

  const resolved = theme === "system" ? (systemIsDark ? "dark" : "light") : theme;

  return (
    <div
      role="radiogroup"
      aria-label="Color scheme preference"
      className={`inline-flex items-center rounded-lg border border-ink-700 bg-ink-900/80 p-0.5 text-xs text-ink-400 ${className}`}
    >
      <button
        type="button"
        role="radio"
        aria-checked={mounted ? theme === "light" : false}
        onClick={() => handleSelect("light")}
        title="Light theme"
        className={`flex items-center gap-1 rounded-md px-2 py-1 transition ${
          mounted && theme === "light"
            ? "bg-ink-800 text-ink-200 font-medium shadow-xs"
            : "hover:text-ink-200"
        }`}
      >
        <svg
          viewBox="0 0 24 24"
          className="h-3.5 w-3.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
        </svg>
        <span className="sr-only sm:not-sr-only">Light</span>
      </button>

      <button
        type="button"
        role="radio"
        aria-checked={mounted ? theme === "dark" : false}
        onClick={() => handleSelect("dark")}
        title="Dark theme"
        className={`flex items-center gap-1 rounded-md px-2 py-1 transition ${
          mounted && theme === "dark"
            ? "bg-ink-800 text-ink-200 font-medium shadow-xs"
            : "hover:text-ink-200"
        }`}
      >
        <svg
          viewBox="0 0 24 24"
          className="h-3.5 w-3.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
        </svg>
        <span className="sr-only sm:not-sr-only">Dark</span>
      </button>

      <button
        type="button"
        role="radio"
        aria-checked={mounted ? theme === "system" : true}
        onClick={() => handleSelect("system")}
        title={`System preference (currently ${mounted ? resolved : "system"})`}
        className={`flex items-center gap-1 rounded-md px-2 py-1 transition ${
          mounted && theme === "system"
            ? "bg-ink-800 text-ink-200 font-medium shadow-xs"
            : "hover:text-ink-200"
        }`}
      >
        <svg
          viewBox="0 0 24 24"
          className="h-3.5 w-3.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <rect width="20" height="14" x="2" y="3" rx="2" />
          <line x1="8" x2="16" y1="21" y2="21" />
          <line x1="12" x2="12" y1="17" y2="21" />
        </svg>
        <span className="sr-only sm:not-sr-only">Auto</span>
      </button>
    </div>
  );
}

