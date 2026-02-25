import { Nav } from "@/components/nav";

export default function GeneratePage() {
  return (
    <div className="min-h-screen">
      <Nav />
      <main className="mx-auto max-w-6xl px-6 py-10">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-[--foreground]">
            <span className="text-emerald-400">⬟</span> LoRA Generator
          </h1>
          <p className="mt-1 text-sm text-[--muted-fg]">
            Generate PEFT-compatible LoRA YAML configs tuned to detected architecture.
          </p>
        </div>
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-[--border] py-24 text-center">
          <div className="text-4xl mb-4 text-[--muted-fg]">⬟</div>
          <p className="text-lg font-semibold text-[--foreground]">Coming in Phase 4</p>
          <p className="mt-2 max-w-md text-sm text-[--muted-fg]">
            The LoRA config generator synthesizes PEFT-compatible training configs from
            behavioral scan results and architecture descriptors.
          </p>
          <a
            href="/scan"
            className="mt-6 inline-flex items-center gap-2 rounded-lg border border-[--border] bg-[--card] px-4 py-2
                       text-sm text-[--muted-fg] hover:bg-[--card-hover] hover:text-[--foreground] transition-colors"
          >
            ◈ Run a behavioral scan first
          </a>
        </div>
      </main>
    </div>
  );
}
