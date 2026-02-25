"use client";

import type { DimensionScore } from "@/lib/api";
import { LOWER_IS_BETTER, VERBOSITY_NEUTRAL } from "@/lib/utils";

interface Props {
  scores: DimensionScore[];
}

function getHeatColor(dim: string, score: number): string {
  let effective: number;
  if (dim === VERBOSITY_NEUTRAL) {
    effective = 1 - Math.abs(score - 0.5) * 2;
  } else {
    effective = LOWER_IS_BETTER.has(dim) ? 1 - score : score;
  }

  if (effective >= 0.65) {
    // green zone: interpolate #064e3b → #10b981
    const t = (effective - 0.65) / 0.35;
    const r = Math.round(6 + t * (16 - 6));
    const g = Math.round(78 + t * (185 - 78));
    const b = Math.round(59 + t * (129 - 59));
    return `rgb(${r},${g},${b})`;
  } else if (effective >= 0.40) {
    // amber zone
    const t = (effective - 0.40) / 0.25;
    const r = Math.round(120 + t * (245 - 120));
    const g = Math.round(60 + t * (158 - 60));
    const b = Math.round(10 + t * (11 - 10));
    return `rgb(${r},${g},${b})`;
  } else {
    // red zone
    const t = effective / 0.40;
    const r = Math.round(100 + t * (244 - 100));
    const g = Math.round(0 + t * (63 - 0));
    const b = Math.round(0 + t * (94 - 0));
    return `rgb(${r},${g},${b})`;
  }
}

export function ScoreHeatmap({ scores }: Props) {
  return (
    <div className="grid grid-cols-3 gap-2 sm:grid-cols-3">
      {scores.map((s) => {
        const bg = getHeatColor(s.dimension, s.score);
        const pct = Math.round(s.score * 100);
        return (
          <div
            key={s.dimension}
            className="group relative flex flex-col items-center justify-center rounded-lg p-3 cursor-default transition-all duration-200 hover:scale-105"
            style={{ backgroundColor: `${bg}22`, border: `1px solid ${bg}44` }}
            title={s.interpretation}
          >
            {/* Score pill */}
            <span
              className="text-xl font-bold"
              style={{ color: bg }}
            >
              {pct}
            </span>
            <span className="mt-0.5 text-[10px] font-medium uppercase tracking-widest text-[--muted-fg]">
              {s.dimension}
            </span>

            {/* Tooltip on hover */}
            <div className="pointer-events-none absolute bottom-full left-1/2 mb-2 -translate-x-1/2 w-max max-w-[200px]
                            rounded-lg border border-[--border] bg-[--card] px-3 py-2 text-xs text-[--foreground]
                            opacity-0 group-hover:opacity-100 transition-opacity z-10 shadow-lg">
              {s.interpretation}
            </div>
          </div>
        );
      })}
    </div>
  );
}
