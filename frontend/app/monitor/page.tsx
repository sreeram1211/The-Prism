import { Nav } from "@/components/nav";

export default function MonitorPage() {
  return (
    <div className="min-h-screen">
      <Nav />
      <main className="mx-auto max-w-6xl px-6 py-10">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-[--foreground]">
            <span className="text-amber-400">◉</span> Live Monitor
          </h1>
          <p className="mt-1 text-sm text-[--muted-fg]">
            Real-time training telemetry: loss curves, gradient norms, adapter convergence.
          </p>
        </div>
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-[--border] py-24 text-center">
          <div className="text-4xl mb-4 text-[--muted-fg]">◉</div>
          <p className="text-lg font-semibold text-[--foreground]">Coming in Phase 5</p>
          <p className="mt-2 max-w-md text-sm text-[--muted-fg]">
            WebSocket-powered real-time telemetry dashboard. Monitor loss curves, gradient norms,
            and Proprioceptive Nervous System layer signals during LoRA fine-tuning.
          </p>
        </div>
      </main>
    </div>
  );
}
