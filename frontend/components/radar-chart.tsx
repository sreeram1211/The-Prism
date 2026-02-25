"use client";

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { DimensionScore } from "@/lib/api";
import { LOWER_IS_BETTER } from "@/lib/utils";

interface Props {
  scores: DimensionScore[];
}

const DIM_LABELS: Record<string, string> = {
  sycophancy:  "Sycophancy",
  hedging:     "Hedging",
  calibration: "Calibration",
  depth:       "Depth",
  coherence:   "Coherence",
  focus:       "Focus",
  specificity: "Specificity",
  verbosity:   "Verbosity",
  repetition:  "Repetition",
};

// For lower-is-better dims, invert the score so the radar shows "good = outer"
function displayScore(dim: string, score: number): number {
  if (dim === "verbosity") {
    // optimal at 0.5 — map distance from 0.5 to displayed 0-1 where 1=optimal
    return 1 - Math.abs(score - 0.5) * 2;
  }
  return LOWER_IS_BETTER.has(dim) ? 1 - score : score;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { label: string; raw: number; display: number; interpretation: string } }> }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-[--border] bg-[--card] p-3 text-xs shadow-lg">
      <p className="font-semibold text-[--foreground] mb-1">{d.label}</p>
      <p className="text-[--muted-fg]">Score: <span className="text-[--foreground]">{(d.raw * 100).toFixed(0)}%</span></p>
      <p className="mt-1 text-[--muted-fg] max-w-[180px] leading-relaxed">{d.interpretation}</p>
    </div>
  );
}

export function BehavioralRadarChart({ scores }: Props) {
  const data = scores.map((s) => ({
    label: DIM_LABELS[s.dimension] ?? s.dimension,
    dimension: s.dimension,
    raw: s.score,
    display: displayScore(s.dimension, s.score),
    interpretation: s.interpretation,
  }));

  return (
    <ResponsiveContainer width="100%" height={340}>
      <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
        <PolarGrid stroke="#2a2a3a" strokeDasharray="3 3" />
        <PolarAngleAxis
          dataKey="label"
          tick={{ fill: "#6b6b8a", fontSize: 11, fontFamily: "monospace" }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 1]}
          tick={{ fill: "#6b6b8a", fontSize: 9 }}
          tickCount={4}
          stroke="#2a2a3a"
        />
        <Radar
          name="Score"
          dataKey="display"
          stroke="#7c6af5"
          fill="#7c6af5"
          fillOpacity={0.25}
          strokeWidth={2}
          dot={{ r: 3, fill: "#7c6af5", strokeWidth: 0 }}
        />
        <Tooltip content={<CustomTooltip />} />
      </RadarChart>
    </ResponsiveContainer>
  );
}
