import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "The Prism · AI Behavioral Manifold Tooling",
  description: "Local-first AI behavioral diagnostic and LoRA fine-tuning suite",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased min-h-screen bg-[--background] text-[--foreground]">
        {children}
      </body>
    </html>
  );
}
