import Link from "next/link";
import { Nav } from "@/components/nav";

const features = [
  {
    href: "/resolve",
    icon: "⬡",
    title: "Auto-Resolver",
    subtitle: "Architecture Detection",
    desc: "Detect model family, count parameters, and surface LoRA target recommendations — downloading only config.json, never weights.",
    accent: "#7c6af5",
    soon: false,
  },
  {
    href: "/scan",
    icon: "◈",
    title: "Behavioral Scan",
    subtitle: "9-Dim Diagnostic",
    desc: "Probe sycophancy, hedging, calibration, depth, coherence, focus, specificity, verbosity and repetition with a radar chart + heatmap.",
    accent: "#22d3ee",
    soon: false,
  },
  {
    href: "/generate",
    icon: "⬟",
    title: "LoRA Generator",
    subtitle: "Config Synthesis",
    desc: "Generate PEFT-compatible LoRA YAML configs tuned to the detected architecture and training objective. (Phase 4)",
    accent: "#10b981",
    soon: true,
  },
  {
    href: "/monitor",
    icon: "◉",
    title: "Live Monitor",
    subtitle: "WebSocket Telemetry",
    desc: "Real-time training telemetry dashboard: loss curves, gradient norms, adapter convergence diagnostics. (Phase 5)",
    accent: "#f59e0b",
    soon: true,
  },
];

export default function HomePage() {
  return (
    <div className="min-h-screen">
      <Nav />

      {/* Hero */}
      <section className="mx-auto max-w-6xl px-6 py-20 text-center">
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[--brand]/30 bg-[--brand]/10 px-4 py-1.5 text-xs text-[--brand]">
          <span className="h-1.5 w-1.5 rounded-full bg-[--brand] animate-pulse" />
          Phase 3 · Next.js Frontend
        </div>
        <h1 className="mt-6 text-5xl font-bold tracking-tight">
          <span className="text-[--brand]">The Prism</span>
          <br />
          <span className="text-[--foreground]/80 text-3xl font-normal">
            AI Behavioral Manifold Tooling
          </span>
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-base text-[--muted-fg] leading-relaxed">
          Local-first diagnostics for large language models. Resolve architectures, probe
          behavioral signatures, generate fine-tuning configs — no weights loaded until you
          ask for them.
        </p>
        <div className="mt-8 flex items-center justify-center gap-4">
          <Link
            href="/resolve"
            className="inline-flex items-center gap-2 rounded-lg bg-[--brand] px-6 py-3 text-sm font-semibold text-white
                       shadow-lg shadow-[--brand]/20 hover:bg-[--brand]/90 transition-all duration-150"
          >
            ⬡ Resolve a model
          </Link>
          <Link
            href="/scan"
            className="inline-flex items-center gap-2 rounded-lg border border-[--border] bg-[--card] px-6 py-3 text-sm font-medium
                       text-[--foreground] hover:bg-[--card-hover] transition-all duration-150"
          >
            ◈ Run a scan
          </Link>
        </div>
      </section>

      {/* Feature cards */}
      <section className="mx-auto max-w-6xl px-6 pb-24">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((f) => (
            <Link
              key={f.href}
              href={f.soon ? "#" : f.href}
              className={[
                "group relative rounded-xl border border-[--border] bg-[--card] p-6",
                "transition-all duration-200",
                f.soon
                  ? "opacity-60 cursor-not-allowed"
                  : "hover:border-[--brand]/40 hover:bg-[--card-hover] hover:-translate-y-0.5",
              ].join(" ")}
            >
              {f.soon && (
                <span className="absolute right-3 top-3 rounded-md border border-[--border] bg-[--muted] px-1.5 py-0.5 text-[10px] text-[--muted-fg]">
                  Soon
                </span>
              )}
              <div className="mb-4 text-2xl" style={{ color: f.accent }}>{f.icon}</div>
              <h3 className="text-base font-semibold text-[--foreground]">{f.title}</h3>
              <p className="text-xs text-[--muted-fg] mt-0.5 mb-3">{f.subtitle}</p>
              <p className="text-sm text-[--muted-fg] leading-relaxed">{f.desc}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-[--border] py-8 text-center text-xs text-[--muted-fg]">
        The Prism v0.3.0-phase3 · BuildMaxxing · Local-first AI tooling
      </footer>
    </div>
  );
}
