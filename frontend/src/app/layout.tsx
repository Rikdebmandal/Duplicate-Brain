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
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
