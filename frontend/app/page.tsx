"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/nav";
import { getDashboardStats, type DashboardStats } from "@/lib/api";

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: number | string;
  accent: string;
}) {
  return (
    <div
      className="rounded-xl border bg-[--card] px-6 py-5"
      style={{ borderColor: accent + "33" }}
    >
      <p className="text-xs font-medium uppercase tracking-wider text-[--muted-fg] mb-1">
        {label}
      </p>
      <p className="text-3xl font-bold" style={{ color: accent }}>
        {value}
      </p>
    </div>
  );
}

function relativeTime(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export default function HomePage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);

  useEffect(() => {
    getDashboardStats()
      .then(setStats)
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-screen">
      <Nav />

      <main className="mx-auto max-w-6xl px-6 py-12">
        {/* Hero */}
        <div className="mb-10 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[--brand]/30 bg-[--brand]/10 px-4 py-1.5 text-xs text-[--brand]">
            <span className="h-1.5 w-1.5 rounded-full bg-[--brand] animate-pulse" />
            Phase 6 · Persistence + Comparison Engine
          </div>
          <h1 className="mt-4 text-5xl font-bold tracking-tight">
            <span className="text-[--brand]">The Prism</span>
            <br />
            <span className="text-[--foreground]/80 text-3xl font-normal">
              AI Behavioral Manifold Tooling
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-sm text-[--muted-fg] leading-relaxed">
            Local-first diagnostics for large language models. Resolve architectures, probe
            behavioral signatures, compare models, generate fine-tuning configs — no weights
            loaded until you ask.
          </p>
        </div>

        {/* Live counters */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-10">
          <StatCard label="Total Scans" value={stats?.total_scans ?? "—"} accent="#7c6af5" />
          <StatCard label="Generate Jobs" value={stats?.total_jobs ?? "—"} accent="#10b981" />
          <StatCard label="Agent Sessions" value={stats?.total_sessions ?? "—"} accent="#f59e0b" />
          <StatCard label="Models Seen" value={stats?.unique_models ?? "—"} accent="#22d3ee" />
        </div>

        {/* Recent activity */}
        {stats && stats.recent_scans.length > 0 && (
          <div className="mb-10 rounded-xl border border-[--border] bg-[--card] overflow-hidden">
            <div className="px-6 py-4 border-b border-[--border]">
              <h2 className="text-sm font-semibold text-[--foreground]">Recent Scans</h2>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-[--muted-fg] border-b border-[--border] bg-[--muted]">
                  <th className="text-left px-6 py-3">Model</th>
                  <th className="text-left px-4 py-3">Date</th>
                  <th className="text-right px-4 py-3">Geo-ratio</th>
                  <th className="text-left px-4 py-3">Top dim</th>
                  <th className="text-right px-6 py-3">Score</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_scans.map((s) => (
                  <tr
                    key={s.scan_id}
                    className="border-b border-[--border]/50 hover:bg-[--card-hover] transition-colors"
                  >
                    <td className="px-6 py-3 font-medium text-[--foreground] max-w-xs truncate">
                      {s.model_id}
                    </td>
                    <td className="px-4 py-3 text-[--muted-fg] text-xs">
                      {relativeTime(s.created_at)}
                    </td>
                    <td className="px-4 py-3 text-right text-[--muted-fg] text-xs">
                      {s.geometric_separation_ratio.toFixed(0)}×
                    </td>
                    <td className="px-4 py-3 text-xs text-[--muted-fg]">
                      {s.top_score.dimension}
                    </td>
                    <td className="px-6 py-3 text-right text-xs text-[--foreground]">
                      {(s.top_score.score * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="px-6 py-3 border-t border-[--border]">
              <Link href="/history" className="text-xs text-[--brand] hover:underline">
                View full history →
              </Link>
            </div>
          </div>
        )}

        {/* Quick actions */}
        <div className="flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/resolve"
            className="inline-flex items-center gap-2 rounded-lg border border-[--border] bg-[--card] px-5 py-3 text-sm font-medium text-[--foreground] hover:bg-[--card-hover] transition-all duration-150"
          >
            ⬡ Resolve a model
          </Link>
          <Link
            href="/scan"
            className="inline-flex items-center gap-2 rounded-lg bg-[--brand] px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-[--brand]/20 hover:bg-[--brand]/90 transition-all duration-150"
          >
            ◈ Run a scan
          </Link>
          <Link
            href="/compare"
            className="inline-flex items-center gap-2 rounded-lg border border-[--border] bg-[--card] px-5 py-3 text-sm font-medium text-[--foreground] hover:bg-[--card-hover] transition-all duration-150"
          >
            ⟺ Compare models
          </Link>
          <Link
            href="/history"
            className="inline-flex items-center gap-2 rounded-lg border border-[--border] bg-[--card] px-5 py-3 text-sm font-medium text-[--foreground] hover:bg-[--card-hover] transition-all duration-150"
          >
            ◷ View history
          </Link>
        </div>
      </main>

      <footer className="border-t border-[--border] py-8 text-center text-xs text-[--muted-fg]">
        The Prism v0.6.0 · BuildMaxxing · Local-first AI tooling
      </footer>
    </div>
  );
}
