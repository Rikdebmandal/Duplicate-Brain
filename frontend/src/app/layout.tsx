import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Personal Cognitive Digital Twin",
  description:
    "A behavioural decision-prediction model built from your own recorded decisions. Estimates, not certainties.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="color-scheme" content="light dark" />
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){
              try {
                var saved = localStorage.getItem("color-scheme");
                var scheme = saved === "light" || saved === "dark" ? saved : "light dark";
                var meta = document.querySelector('meta[name="color-scheme"]');
                if (meta) meta.content = scheme;
                if (saved === "light" || saved === "dark") {
                  document.documentElement.dataset.theme = saved;
                }
              } catch (e) {}
            })();`,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
