"use client";

import { useState } from "react";
import { Nav } from "@/components/nav";
import { generateLoRA, resolveModel, type GenerateLoRAResult } from "@/lib/api";

const DIMENSIONS = [
  { key: "sycophancy",  label: "Sycophancy",   icon: "↓", description: "Agreement bias — lower = more assertive", higherIsBetter: false },
  { key: "hedging",     label: "Hedging",       icon: "↓", description: "Qualification overload — lower = more direct", higherIsBetter: false },
  { key: "calibration", label: "Calibration",   icon: "↑", description: "Confidence alignment — higher = better epistemic precision", higherIsBetter: true },
  { key: "depth",       label: "Depth",         icon: "↑", description: "Reasoning elaboration — higher = richer analysis", higherIsBetter: true },
  { key: "coherence",   label: "Coherence",     icon: "↑", description: "Logical consistency — higher = tighter arguments", higherIsBetter: true },
  { key: "focus",       label: "Focus",         icon: "↑", description: "Topical discipline — higher = less drift", higherIsBetter: true },
  { key: "specificity", label: "Specificity",   icon: "↑", description: "Concrete grounding — higher = more examples", higherIsBetter: true },
  { key: "verbosity",   label: "Verbosity",     icon: "◎", description: "Response length — optimal at 0.5", higherIsBetter: null },
  { key: "repetition",  label: "Repetition",    icon: "↓", description: "Self-repetition — lower = higher lexical diversity", higherIsBetter: false },
] as const;

type DimKey = typeof DIMENSIONS[number]["key"];

type Targets = Record<DimKey, number>;

const DEFAULT_TARGETS: Targets = {
  sycophancy:  0.15,
  hedging:     0.20,
  calibration: 0.80,
  depth:       0.75,
  coherence:   0.85,
  focus:       0.80,
  specificity: 0.75,
  verbosity:   0.50,
  repetition:  0.10,
};

function sliderColor(key: DimKey, value: number, higherIsBetter: boolean | null): string {
  if (higherIsBetter === null) {
    const dist = Math.abs(value - 0.5);
    if (dist < 0.15) return "#22c55e";
    if (dist < 0.30) return "#f59e0b";
    return "#ef4444";
  }
  const effective = higherIsBetter ? value : 1 - value;
  if (effective >= 0.65) return "#22c55e";
  if (effective >= 0.40) return "#f59e0b";
  return "#ef4444";
}

export default function GeneratePage() {
  const [modelId, setModelId] = useState("");
  const [targets, setTargets] = useState<Targets>(DEFAULT_TARGETS);
  const [loraRank, setLoraRank] = useState<number | "auto">("auto");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateLoRAResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"config" | "yaml">("config");
  const [copied, setCopied] = useState(false);

  async function handleGenerate() {
    if (!modelId.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await generateLoRA(modelId.trim(), targets, loraRank === "auto" ? undefined : loraRank);
      setResult(data);
      setActiveTab("config");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  }

  function handleCopy(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  const configJson = result ? JSON.stringify(result.adapter_config, null, 2) : "";
  const activeContent = result ? (activeTab === "config" ? configJson : result.training_yaml) : "";

  return (
    <div className="min-h-screen">
      <Nav />
      <main className="mx-auto max-w-6xl px-6 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-[--foreground]">
            <span className="text-emerald-400">⬟</span> LoRA Generator
          </h1>
          <p className="mt-1 text-sm text-[--muted-fg]">
            Set 9 behavioral targets → derive optimal LoRA hyperparameters → get a PEFT-ready config.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
          {/* ---- Left panel: inputs ---------------------------------------- */}
          <div className="lg:col-span-2 space-y-5">
            {/* Model ID */}
            <div className="rounded-xl border border-[--border] bg-[--card] p-5">
              <label className="block text-xs font-semibold uppercase tracking-wider text-[--muted-fg] mb-2">
                Model ID
              </label>
              <input
                type="text"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
                placeholder="mistralai/Mistral-7B-v0.1"
                className="w-full rounded-lg border border-[--border] bg-[--background] px-3 py-2 text-sm
                           text-[--foreground] placeholder:text-[--muted-fg] focus:outline-none
                           focus:ring-1 focus:ring-[--brand]/50 font-mono"
              />
            </div>

            {/* LoRA Rank override */}
            <div className="rounded-xl border border-[--border] bg-[--card] p-5">
              <label className="block text-xs font-semibold uppercase tracking-wider text-[--muted-fg] mb-3">
                LoRA Rank
              </label>
              <div className="flex flex-wrap gap-2">
                {(["auto", 4, 8, 16, 32, 64, 128] as const).map((r) => (
                  <button
                    key={r}
                    onClick={() => setLoraRank(r)}
                    className={`rounded-md px-3 py-1 text-xs font-mono transition-colors ${
                      loraRank === r
                        ? "bg-[--brand] text-white"
                        : "border border-[--border] text-[--muted-fg] hover:text-[--foreground]"
                    }`}
                  >
                    {r === "auto" ? "auto" : `r=${r}`}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-xs text-[--muted-fg]">
                Auto derives rank from target complexity.
              </p>
            </div>

            {/* Behavioral Sliders */}
            <div className="rounded-xl border border-[--border] bg-[--card] p-5">
              <p className="text-xs font-semibold uppercase tracking-wider text-[--muted-fg] mb-4">
                Behavioral Targets
              </p>
              <div className="space-y-4">
                {DIMENSIONS.map((dim) => {
                  const val = targets[dim.key];
                  const color = sliderColor(dim.key, val, dim.higherIsBetter);
                  return (
                    <div key={dim.key}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-[--foreground] font-medium">
                          <span className="text-[--muted-fg] mr-1">{dim.icon}</span>
                          {dim.label}
                        </span>
                        <span className="text-xs font-mono" style={{ color }}>
                          {val.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={val}
                        onChange={(e) =>
                          setTargets((prev) => ({ ...prev, [dim.key]: parseFloat(e.target.value) }))
                        }
                        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                        style={{ accentColor: color }}
                        title={dim.description}
                      />
                    </div>
                  );
                })}
              </div>

              <button
                onClick={() => setTargets(DEFAULT_TARGETS)}
                className="mt-4 text-xs text-[--muted-fg] hover:text-[--foreground] transition-colors"
              >
                ↺ Reset to recommended defaults
              </button>
            </div>

            {/* Generate button */}
            <button
              onClick={handleGenerate}
              disabled={loading || !modelId.trim()}
              className="w-full rounded-xl bg-[--brand] px-4 py-3 text-sm font-semibold text-white
                         hover:bg-[--brand]/80 disabled:opacity-50 disabled:cursor-not-allowed
                         transition-colors"
            >
              {loading ? "Compiling…" : "⬟ Generate LoRA Config"}
            </button>

            {error && (
              <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
                {error}
              </div>
            )}
          </div>

          {/* ---- Right panel: output --------------------------------------- */}
          <div className="lg:col-span-3">
            {!result ? (
              <div className="flex h-full min-h-[400px] flex-col items-center justify-center
                              rounded-xl border border-dashed border-[--border] text-center p-10">
                <div className="text-4xl mb-4 text-[--muted-fg]">⬟</div>
                <p className="text-sm text-[--muted-fg] max-w-xs">
                  Set your behavioral targets on the left, enter a model ID, then click
                  &ldquo;Generate LoRA Config&rdquo; to compile your adapter.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Stats row */}
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {[
                    { label: "LoRA Rank", value: `r=${result.lora_rank}` },
                    { label: "Alpha", value: result.lora_alpha.toString() },
                    { label: "Dropout", value: result.lora_dropout.toFixed(3) },
                    { label: "Est. Size", value: `${result.estimated_size_mb} MB` },
                  ].map((s) => (
                    <div key={s.label} className="rounded-lg border border-[--border] bg-[--card] p-3 text-center">
                      <div className="text-lg font-bold font-mono text-[--brand]">{s.value}</div>
                      <div className="text-xs text-[--muted-fg] mt-0.5">{s.label}</div>
                    </div>
                  ))}
                </div>

                {/* Target modules */}
                <div className="rounded-xl border border-[--border] bg-[--card] p-4">
                  <p className="text-xs font-semibold uppercase tracking-wider text-[--muted-fg] mb-2">
                    Target Modules
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {result.target_modules.map((m) => (
                      <span key={m} className="rounded-md bg-[--brand]/10 border border-[--brand]/20
                                               px-2 py-0.5 text-xs font-mono text-[--brand]">
                        {m}
                      </span>
                    ))}
                  </div>
                  <p className="mt-2 text-xs text-[--muted-fg]">
                    ~{result.trainable_params.toLocaleString()} trainable parameters
                  </p>
                </div>

                {/* Code viewer */}
                <div className="rounded-xl border border-[--border] bg-[--card] overflow-hidden">
                  {/* Tab bar */}
                  <div className="flex items-center justify-between border-b border-[--border] px-4 py-2">
                    <div className="flex gap-2">
                      {(["config", "yaml"] as const).map((tab) => (
                        <button
                          key={tab}
                          onClick={() => setActiveTab(tab)}
                          className={`rounded px-2 py-1 text-xs font-mono transition-colors ${
                            activeTab === tab
                              ? "bg-[--brand]/10 text-[--brand]"
                              : "text-[--muted-fg] hover:text-[--foreground]"
                          }`}
                        >
                          {tab === "config" ? "adapter_config.json" : "training.yaml"}
                        </button>
                      ))}
                    </div>
                    <button
                      onClick={() => handleCopy(activeContent)}
                      className="text-xs text-[--muted-fg] hover:text-[--foreground] transition-colors px-2 py-1"
                    >
                      {copied ? "✓ Copied" : "⎘ Copy"}
                    </button>
                  </div>
                  <pre className="overflow-auto p-4 text-xs font-mono text-[--foreground] leading-relaxed
                                  max-h-[420px] bg-[--background]/60">
                    {activeContent}
                  </pre>
                </div>

                <p className="text-xs text-[--muted-fg] text-center">
                  Job <span className="font-mono text-[--foreground]">{result.job_id}</span>
                  {" · "}status: <span className="text-emerald-400">{result.status}</span>
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
