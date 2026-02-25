"use client";

import { useState } from "react";
import { Nav } from "@/components/nav";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { BehavioralRadarChart } from "@/components/radar-chart";
import { ScoreHeatmap } from "@/components/score-heatmap";
import { scanModel, type ScanReport } from "@/lib/api";
import { LOWER_IS_BETTER, scoreColor } from "@/lib/utils";

const ALL_DIMENSIONS = [
  "sycophancy", "hedging", "calibration", "depth",
  "coherence", "focus", "specificity", "verbosity", "repetition",
];

const EXAMPLE_MODELS = [
  "mistralai/Mistral-7B-v0.1",
  "meta-llama/Meta-Llama-3-8B",
  "tiiuae/falcon-7b",
];

function overallScore(report: ScanReport): number {
  const sum = report.scores.reduce((acc, s) => {
    const isLower = LOWER_IS_BETTER.has(s.dimension);
    const isVerbosity = s.dimension === "verbosity";
    let effective: number;
    if (isVerbosity) {
      effective = 1 - Math.abs(s.score - 0.5) * 2;
    } else {
      effective = isLower ? 1 - s.score : s.score;
    }
    return acc + effective;
  }, 0);
  return sum / report.scores.length;
}

function OverallScore({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const label = pct >= 65 ? "Strong" : pct >= 40 ? "Moderate" : "Weak";
  const color = pct >= 65 ? "#10b981" : pct >= 40 ? "#f59e0b" : "#f43f5e";
  return (
    <div className="flex flex-col items-center justify-center gap-1 py-4">
      <div className="relative inline-flex items-center justify-center">
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="none" stroke="#2a2a3a" strokeWidth="8" />
          <circle
            cx="50" cy="50" r="40" fill="none"
            stroke={color} strokeWidth="8"
            strokeDasharray={`${2 * Math.PI * 40}`}
            strokeDashoffset={`${2 * Math.PI * 40 * (1 - score)}`}
            strokeLinecap="round"
            transform="rotate(-90 50 50)"
            style={{ transition: "stroke-dashoffset 1s ease" }}
          />
        </svg>
        <span className="absolute text-xl font-bold" style={{ color }}>{pct}</span>
      </div>
      <span className="text-xs font-medium" style={{ color }}>{label}</span>
      <span className="text-[10px] text-[--muted-fg]">Behavioral score</span>
    </div>
  );
}

function ScanResultPanel({ report }: { report: ScanReport }) {
  const overall = overallScore(report);

  return (
    <div className="mt-8 space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold text-[--foreground]">{report.model_id}</h2>
        <Badge variant="brand">9-dim scan</Badge>
        <Badge variant="muted">
          GSR {report.geometric_separation_ratio.toFixed(0)}×
        </Badge>
        <Badge variant="muted">
          {report.scan_duration_ms.toFixed(0)} ms
        </Badge>
        <span className="ml-auto text-xs text-[--muted-fg] italic">
          Phase 2 mock engine · ROC-AUC probes arrive in Phase 3
        </span>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {/* Radar chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Behavioral Radar</CardTitle>
          </CardHeader>
          <BehavioralRadarChart scores={report.scores} />
        </Card>

        {/* Overall + GSR */}
        <Card>
          <CardHeader><CardTitle>Summary</CardTitle></CardHeader>
          <OverallScore score={overall} />
          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[--muted-fg]">Geometric Sep. Ratio</span>
              <span className="font-bold text-[--accent-cyan]">{report.geometric_separation_ratio.toFixed(0)}×</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[--muted-fg]">Scan duration</span>
              <span className="text-[--foreground]">{report.scan_duration_ms.toFixed(0)} ms</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[--muted-fg]">Dimensions scanned</span>
              <span className="text-[--foreground]">{report.scores.length}</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Heatmap */}
      <Card>
        <CardHeader><CardTitle>Score Heatmap</CardTitle></CardHeader>
        <ScoreHeatmap scores={report.scores} />
      </Card>

      {/* Score table */}
      <Card>
        <CardHeader><CardTitle>Dimension Detail</CardTitle></CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[--border]">
                <th className="pb-2 text-left text-[--muted-fg] font-medium">Dimension</th>
                <th className="pb-2 text-right text-[--muted-fg] font-medium">Score</th>
                <th className="pb-2 text-left pl-4 text-[--muted-fg] font-medium">Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {report.scores.map((s) => {
                const higherIsBetter = !LOWER_IS_BETTER.has(s.dimension);
                const colorCls = scoreColor(s.score, higherIsBetter);
                return (
                  <tr key={s.dimension} className="border-b border-[--border]/50 last:border-0">
                    <td className="py-2 font-medium text-[--foreground] capitalize">{s.dimension}</td>
                    <td className={`py-2 text-right font-bold ${colorCls}`}>
                      {(s.score * 100).toFixed(0)}%
                    </td>
                    <td className="py-2 pl-4 text-[--muted-fg]">{s.interpretation}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

export default function ScanPage() {
  const [modelId, setModelId] = useState("");
  const [selectedDims, setSelectedDims] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<ScanReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  function toggleDim(dim: string) {
    setSelectedDims((prev) =>
      prev.includes(dim) ? prev.filter((d) => d !== dim) : [...prev, dim],
    );
  }

  async function handleScan(e: React.FormEvent) {
    e.preventDefault();
    if (!modelId.trim()) return;
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const data = await scanModel(
        modelId.trim(),
        selectedDims.length ? selectedDims : undefined,
      );
      setReport(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen">
      <Nav />
      <main className="mx-auto max-w-6xl px-6 py-10">
        {/* Page header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-[--foreground]">
            <span className="text-[--accent-cyan]">◈</span> Behavioral Scan
          </h1>
          <p className="mt-1 text-sm text-[--muted-fg]">
            9-dimensional behavioral diagnostic: sycophancy, hedging, calibration, depth, coherence,
            focus, specificity, verbosity, repetition.
          </p>
        </div>

        {/* Form */}
        <Card>
          <form onSubmit={handleScan} className="flex flex-col gap-4">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
              <div className="flex-1">
                <label className="mb-1.5 block text-xs font-medium text-[--muted-fg]">
                  HuggingFace Model ID
                </label>
                <Input
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  placeholder="e.g. mistralai/Mistral-7B-v0.1"
                  disabled={loading}
                />
              </div>
                <Button type="submit" loading={loading} disabled={!modelId.trim()}>
                Run Scan
              </Button>
            </div>

            {/* Dimension toggles */}
            <div>
              <p className="mb-2 text-xs text-[--muted-fg]">
                Dimensions to scan{" "}
                <span className="text-[--brand]">
                  ({selectedDims.length === 0 ? "all" : selectedDims.length} selected)
                </span>
              </p>
              <div className="flex flex-wrap gap-1.5">
                {ALL_DIMENSIONS.map((dim) => {
                  const active = selectedDims.includes(dim);
                  return (
                    <button
                      key={dim}
                      type="button"
                      onClick={() => toggleDim(dim)}
                      className={[
                        "rounded-md border px-2 py-0.5 text-xs capitalize transition-all duration-150",
                        active
                          ? "border-[--brand]/40 bg-[--brand]/10 text-[--brand]"
                          : "border-[--border] bg-[--muted] text-[--muted-fg] hover:text-[--foreground]",
                      ].join(" ")}
                    >
                      {dim}
                    </button>
                  );
                })}
                {selectedDims.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setSelectedDims([])}
                    className="rounded-md border border-[--border] px-2 py-0.5 text-xs text-rose-400 hover:bg-rose-500/10 transition-colors"
                  >
                    clear
                  </button>
                )}
              </div>
            </div>
          </form>

          {/* Quick-pick examples */}
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="text-xs text-[--muted-fg]">Examples:</span>
            {EXAMPLE_MODELS.map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setModelId(m)}
                className="rounded-md border border-[--border] bg-[--muted] px-2 py-0.5 text-xs text-[--muted-fg]
                           hover:border-[--brand]/30 hover:text-[--foreground] transition-colors"
              >
                {m}
              </button>
            ))}
          </div>
        </Card>

        {/* Error */}
        {error && (
          <div className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-400">
            {error}
          </div>
        )}

        {/* Results */}
        {report && <ScanResultPanel report={report} />}
      </main>
    </div>
  );
}
