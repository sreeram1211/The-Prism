"use client";

import { useEffect, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Nav } from "@/components/nav";
import { createMonitorSession } from "@/lib/api";

const LAYERS = [0, 8, 16, 24];
const LAYER_COLORS = ["#7c6af5", "#22c55e", "#f59e0b", "#38bdf8"];
const DIM_KEYS = [
  "sycophancy","hedging","calibration","depth",
  "coherence","focus","specificity","verbosity","repetition",
];

interface ActivationFrame {
  token_index: number;
  token_text: string;
  timestamp_us: number;
  layer_activations: Record<number, number[]>;
  reflex_override: boolean;
  manifold_coords: number[];
  dimension_drift: Record<string, number>;
}

interface ChartPoint {
  idx: number;
  token: string;
  l0: number;
  l8: number;
  l16: number;
  l24: number;
  reflex: number;
}

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const WS_BASE = BASE_URL.replace(/^http/, "ws");

function layerMean(acts: number[]): number {
  if (!acts?.length) return 0;
  return acts.reduce((a, b) => a + b, 0) / acts.length;
}

export default function MonitorPage() {
  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("Tell me about large language models.");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [frames, setFrames] = useState<ActivationFrame[]>([]);
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [totalReflex, setTotalReflex] = useState(0);
  const [tokenStream, setTokenStream] = useState<string[]>([]);
  const [latestDrift, setLatestDrift] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  function reset() {
    setFrames([]);
    setChartData([]);
    setTotalReflex(0);
    setTokenStream([]);
    setLatestDrift({});
    setDone(false);
    setError(null);
  }

  async function handleStart() {
    if (!modelId.trim()) return;
    reset();
    setRunning(true);
    setError(null);

    try {
      const session = await createMonitorSession(modelId.trim(), prompt.trim());
      setSessionId(session.session_id);

      const wsUrl = `${WS_BASE}${session.websocket_url}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data as string);
        if (data.type === "done") {
          setDone(true);
          setRunning(false);
          return;
        }

        const frame = data as ActivationFrame;
        setFrames((prev) => [...prev.slice(-200), frame]);
        setTokenStream((prev) => [...prev.slice(-80), frame.token_text]);
        setLatestDrift(frame.dimension_drift ?? {});

        if (frame.reflex_override) {
          setTotalReflex((n) => n + 1);
        }

        const point: ChartPoint = {
          idx: frame.token_index,
          token: frame.token_text,
          l0:  layerMean(frame.layer_activations?.[0]  ?? []),
          l8:  layerMean(frame.layer_activations?.[8]  ?? []),
          l16: layerMean(frame.layer_activations?.[16] ?? []),
          l24: layerMean(frame.layer_activations?.[24] ?? []),
          reflex: frame.reflex_override ? 0.5 : 0,
        };
        setChartData((prev) => [...prev.slice(-120), point]);
      };

      ws.onerror = () => {
        setError("WebSocket connection failed — is the backend running?");
        setRunning(false);
      };

      ws.onclose = () => {
        setRunning(false);
      };
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to start session");
      setRunning(false);
    }
  }

  function handleStop() {
    wsRef.current?.close();
    setRunning(false);
  }

  return (
    <div className="min-h-screen">
      <Nav />
      <main className="mx-auto max-w-6xl px-6 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-[--foreground]">
            <span className="text-amber-400">◉</span> Live Monitor
          </h1>
          <p className="mt-1 text-sm text-[--muted-fg]">
            Real-time PNS activation telemetry — 4 layers × token-level resolution via WebSocket.
          </p>
        </div>

        {/* Controls */}
        <div className="mb-6 flex flex-wrap gap-3 items-end">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs text-[--muted-fg] mb-1">Model ID</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="mistralai/Mistral-7B-v0.1"
              disabled={running}
              className="w-full rounded-lg border border-[--border] bg-[--card] px-3 py-2 text-sm
                         font-mono text-[--foreground] placeholder:text-[--muted-fg]
                         focus:outline-none focus:ring-1 focus:ring-[--brand]/50 disabled:opacity-50"
            />
          </div>
          <div className="flex-[2] min-w-[260px]">
            <label className="block text-xs text-[--muted-fg] mb-1">Prompt</label>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={running}
              className="w-full rounded-lg border border-[--border] bg-[--card] px-3 py-2 text-sm
                         text-[--foreground] placeholder:text-[--muted-fg]
                         focus:outline-none focus:ring-1 focus:ring-[--brand]/50 disabled:opacity-50"
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleStart}
              disabled={running || !modelId.trim()}
              className="rounded-lg bg-amber-500 px-4 py-2 text-sm font-semibold text-white
                         hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {running ? "⏳ Streaming…" : "▶ Start"}
            </button>
            {running && (
              <button
                onClick={handleStop}
                className="rounded-lg border border-[--border] px-4 py-2 text-sm text-[--muted-fg]
                           hover:text-[--foreground] transition-colors"
              >
                ■ Stop
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {!sessionId && !running && !done ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed
                          border-[--border] py-24 text-center">
            <div className="text-4xl mb-4 text-[--muted-fg]">◉</div>
            <p className="text-sm text-[--muted-fg] max-w-xs">
              Enter a model ID and prompt, then click <strong>Start</strong> to stream
              PNS activation telemetry in real time.
            </p>
          </div>
        ) : (
          <div className="space-y-5">
            {/* Stats strip */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { label: "Tokens", value: chartData.length },
                { label: "Reflex Overrides", value: totalReflex },
                { label: "Session", value: sessionId?.slice(0, 8) ?? "—" },
                { label: "Status", value: done ? "complete" : running ? "streaming" : "stopped" },
              ].map((s) => (
                <div key={s.label} className="rounded-lg border border-[--border] bg-[--card] p-3 text-center">
                  <div className={`text-lg font-bold font-mono ${
                    s.label === "Status" && running ? "text-amber-400 animate-pulse" :
                    s.label === "Status" && done    ? "text-emerald-400" : "text-[--brand]"
                  }`}>{s.value}</div>
                  <div className="text-xs text-[--muted-fg] mt-0.5">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Activation chart */}
            <div className="rounded-xl border border-[--border] bg-[--card] p-5">
              <div className="flex items-center justify-between mb-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-[--muted-fg]">
                  Layer Activation Magnitudes
                </p>
                <div className="flex gap-3">
                  {LAYERS.map((l, i) => (
                    <span key={l} className="flex items-center gap-1 text-xs text-[--muted-fg]">
                      <span className="inline-block h-2 w-2 rounded-full" style={{ background: LAYER_COLORS[i] }} />
                      L{l}
                    </span>
                  ))}
                </div>
              </div>
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                    <XAxis dataKey="idx" tick={{ fontSize: 10, fill: "var(--muted-fg)" }} tickLine={false} />
                    <YAxis tick={{ fontSize: 10, fill: "var(--muted-fg)" }} tickLine={false} domain={[-0.6, 0.6]} />
                    <Tooltip
                      contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }}
                      formatter={(v: number | undefined, name: string | undefined) => [v != null ? v.toFixed(4) : "—", name != null ? name.toUpperCase() : ""]}
                      labelFormatter={(l) => `token ${l}`}
                    />
                    {LAYERS.map((layer, i) => (
                      <Line
                        key={layer}
                        type="monotone"
                        dataKey={`l${layer}`}
                        stroke={LAYER_COLORS[i]}
                        strokeWidth={1.5}
                        dot={false}
                        isAnimationActive={false}
                      />
                    ))}
                    {chartData.filter((p) => p.reflex > 0).map((p) => (
                      <ReferenceLine key={p.idx} x={p.idx} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {totalReflex > 0 && (
                <p className="mt-2 text-xs text-red-400">
                  ↑ Red dashed lines = reflex arc override events ({totalReflex} total)
                </p>
              )}
            </div>

            {/* Token stream + dimension drift */}
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              {/* Token stream */}
              <div className="rounded-xl border border-[--border] bg-[--card] p-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-[--muted-fg] mb-3">
                  Token Stream
                </p>
                <div className="font-mono text-sm leading-relaxed text-[--foreground] min-h-[80px] max-h-[120px] overflow-hidden">
                  {tokenStream.map((t, i) => (
                    <span
                      key={i}
                      className={i === tokenStream.length - 1 ? "text-[--brand]" : ""}
                    >
                      {t}{" "}
                    </span>
                  ))}
                  {running && <span className="animate-pulse text-[--brand]">▋</span>}
                </div>
              </div>

              {/* Dimension drift */}
              <div className="rounded-xl border border-[--border] bg-[--card] p-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-[--muted-fg] mb-3">
                  Dimension Drift (latest token)
                </p>
                <div className="grid grid-cols-3 gap-2">
                  {DIM_KEYS.map((d) => {
                    const drift = latestDrift[d] ?? 0;
                    const color = Math.abs(drift) < 0.005 ? "text-[--muted-fg]"
                      : drift > 0 ? "text-emerald-400" : "text-red-400";
                    return (
                      <div key={d} className="text-center">
                        <div className={`text-xs font-mono font-bold ${color}`}>
                          {drift >= 0 ? "+" : ""}{drift.toFixed(3)}
                        </div>
                        <div className="text-[10px] text-[--muted-fg] truncate">{d}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
