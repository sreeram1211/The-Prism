"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { Nav } from "@/components/nav";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { compareScans, getScanHistory, type CompareResult, type ScanHistoryItem } from "@/lib/api";
import { LOWER_IS_BETTER } from "@/lib/utils";

function displayScore(dim: string, score: number): number {
  if (dim === "verbosity") return 1 - Math.abs(score - 0.5) * 2;
  return LOWER_IS_BETTER.has(dim) ? 1 - score : score;
}

function DeltaCell({ delta, direction }: { delta: number; direction: string }) {
  const color =
    direction === "improved"
      ? "text-emerald-400"
      : direction === "regressed"
      ? "text-red-400"
      : "text-[--muted-fg]";
  const arrow =
    direction === "improved" ? "↑" : direction === "regressed" ? "↓" : "≈";
  return (
    <span className={color}>
      {delta > 0 ? "+" : ""}
      {delta.toFixed(3)} {arrow}
    </span>
  );
}

function ComparePageInner() {
  const params = useSearchParams();
  const [scanA, setScanA] = useState(params.get("a") ?? "");
  const [scanB, setScanB] = useState(params.get("b") ?? "");
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<ScanHistoryItem[]>([]);

  useEffect(() => {
    getScanHistory(undefined, 50, 0)
      .then((r) => setHistory(r.items))
      .catch(() => {});
  }, []);

  // Auto-compare if params were passed in URL
  useEffect(() => {
    const a = params.get("a");
    const b = params.get("b");
    if (a && b) {
      setScanA(a);
      setScanB(b);
      runCompare(a, b);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runCompare(a?: string, b?: string) {
    const idA = a ?? scanA;
    const idB = b ?? scanB;
    if (!idA || !idB) return;
    setLoading(true);
    setError(null);
    try {
      const res = await compareScans(idA, idB);
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  const radarData = result
    ? result.deltas.map((d) => ({
        label: d.dimension,
        a: displayScore(d.dimension, d.score_a),
        b: displayScore(d.dimension, d.score_b),
      }))
    : [];

  const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  return (
    <div className="min-h-screen">
      <Nav />

      <main className="mx-auto max-w-6xl px-6 py-10">
        <h1 className="text-2xl font-bold text-[--foreground] mb-2">
          ⟺ Compare Models
        </h1>
        <p className="text-sm text-[--muted-fg] mb-8">
          Select two scan IDs to compare behavioral fingerprints side-by-side.
        </p>

        {/* Pickers */}
        <div className="grid gap-6 md:grid-cols-2 mb-8">
          {/* Model A */}
          <div className="rounded-xl border border-[--border] bg-[--card] p-5">
            <p className="text-xs font-semibold text-violet-400 mb-3 uppercase tracking-wider">
              Model A
            </p>
            <select
              className="w-full mb-3 rounded-lg border border-[--border] bg-[--background] px-3 py-2 text-sm text-[--foreground] focus:outline-none focus:ring-1 focus:ring-[--brand]"
              value={scanA}
              onChange={(e) => setScanA(e.target.value)}
            >
              <option value="">— pick from history —</option>
              {history.map((h) => (
                <option key={h.scan_id} value={h.scan_id}>
                  {h.model_id} ({h.scan_id.slice(0, 8)})
                </option>
              ))}
            </select>
            <Input
              placeholder="or paste scan ID…"
              value={scanA}
              onChange={(e) => setScanA(e.target.value)}
            />
          </div>

          {/* Model B */}
          <div className="rounded-xl border border-[--border] bg-[--card] p-5">
            <p className="text-xs font-semibold text-cyan-400 mb-3 uppercase tracking-wider">
              Model B
            </p>
            <select
              className="w-full mb-3 rounded-lg border border-[--border] bg-[--background] px-3 py-2 text-sm text-[--foreground] focus:outline-none focus:ring-1 focus:ring-[--brand]"
              value={scanB}
              onChange={(e) => setScanB(e.target.value)}
            >
              <option value="">— pick from history —</option>
              {history.map((h) => (
                <option key={h.scan_id} value={h.scan_id}>
                  {h.model_id} ({h.scan_id.slice(0, 8)})
                </option>
              ))}
            </select>
            <Input
              placeholder="or paste scan ID…"
              value={scanB}
              onChange={(e) => setScanB(e.target.value)}
            />
          </div>
        </div>

        <div className="mb-8">
          <Button
            onClick={() => runCompare()}
            disabled={!scanA || !scanB || loading}
          >
            {loading ? "Comparing…" : "⟺ Run comparison"}
          </Button>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Overlaid radar */}
            <div className="rounded-xl border border-[--border] bg-[--card] p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-[--foreground]">
                  Behavioral Overlay
                </h2>
                <div className="flex gap-3 text-xs">
                  <span className="flex items-center gap-1">
                    <span className="h-2.5 w-2.5 rounded-sm bg-violet-500 opacity-80" />
                    {result.model_a.split("/").pop()}
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="h-2.5 w-2.5 rounded-sm bg-cyan-400 opacity-80" />
                    {result.model_b.split("/").pop()}
                  </span>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
                  <PolarGrid stroke="#2a2a3a" strokeDasharray="3 3" />
                  <PolarAngleAxis
                    dataKey="label"
                    tick={{ fill: "#6b6b8a", fontSize: 10, fontFamily: "monospace" }}
                  />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 1]}
                    tick={{ fill: "#6b6b8a", fontSize: 9 }}
                    tickCount={4}
                    stroke="#2a2a3a"
                  />
                  <Radar
                    name={result.model_a.split("/").pop()}
                    dataKey="a"
                    stroke="#7c6af5"
                    fill="#7c6af5"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                  <Radar
                    name={result.model_b.split("/").pop()}
                    dataKey="b"
                    stroke="#22d3ee"
                    fill="#22d3ee"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                  <Tooltip />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Delta heatmap + summary */}
            <div className="rounded-xl border border-[--border] bg-[--card] p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-[--foreground]">
                  Dimension Deltas
                </h2>
                <div className="flex gap-2">
                  <Badge
                    variant="muted"
                    className="text-xs"
                    style={{ borderColor: "#7c6af5", color: "#7c6af5" }}
                  >
                    dist: {result.composite_distance.toFixed(4)}
                  </Badge>
                  <Badge
                    variant="cyan"
                    className="text-xs"
                  >
                    winner: {result.winner === "tie" ? "tie" : result.winner === "a" ? result.model_a.split("/").pop() : result.model_b.split("/").pop()}
                  </Badge>
                </div>
              </div>

              {/* Heatmap grid */}
              <div className="grid grid-cols-3 gap-2 mb-4">
                {result.deltas.map((d) => {
                  const bg =
                    d.direction === "improved"
                      ? "bg-emerald-500/20 border-emerald-500/40"
                      : d.direction === "regressed"
                      ? "bg-red-500/20 border-red-500/40"
                      : "bg-[--muted] border-[--border]";
                  return (
                    <div
                      key={d.dimension}
                      className={`rounded-lg border p-2 text-center ${bg}`}
                    >
                      <p className="text-[10px] text-[--muted-fg] uppercase font-medium mb-1">
                        {d.dimension}
                      </p>
                      <DeltaCell delta={d.delta} direction={d.direction} />
                    </div>
                  );
                })}
              </div>

              {/* Delta table */}
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-[--muted-fg] border-b border-[--border]">
                      <th className="text-left py-1 pr-3">Dimension</th>
                      <th className="text-right py-1 pr-3">Model A</th>
                      <th className="text-right py-1 pr-3">Model B</th>
                      <th className="text-right py-1">Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.deltas.map((d) => (
                      <tr
                        key={d.dimension}
                        className="border-b border-[--border]/50"
                      >
                        <td className="py-1.5 pr-3 text-[--foreground]">
                          {d.dimension}
                        </td>
                        <td className="py-1.5 pr-3 text-right text-violet-300">
                          {d.score_a.toFixed(3)}
                        </td>
                        <td className="py-1.5 pr-3 text-right text-cyan-300">
                          {d.score_b.toFixed(3)}
                        </td>
                        <td className="py-1.5 text-right">
                          <DeltaCell delta={d.delta} direction={d.direction} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Export */}
              <div className="mt-4">
                <a
                  href={`${BASE}/api/v1/compare`}
                  onClick={(e) => {
                    e.preventDefault();
                    const blob = new Blob(
                      [JSON.stringify(result, null, 2)],
                      { type: "application/json" }
                    );
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `diff_${result.scan_id_a.slice(0, 8)}_vs_${result.scan_id_b.slice(0, 8)}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="text-xs text-[--muted-fg] hover:text-[--foreground] transition-colors"
                >
                  ↓ Export diff JSON
                </a>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="min-h-screen"><Nav /></div>}>
      <ComparePageInner />
    </Suspense>
  );
}
