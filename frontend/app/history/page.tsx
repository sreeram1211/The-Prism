"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/nav";
import { BehavioralRadarChart } from "@/components/radar-chart";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  getScanHistory,
  getScanResult,
  type ScanHistoryItem,
  type ScanReport,
} from "@/lib/api";

function relativeTime(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

interface ExpandedRow {
  scanId: string;
  report: ScanReport | null;
  loading: boolean;
}

export default function HistoryPage() {
  const [search, setSearch] = useState("");
  const [items, setItems] = useState<ScanHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<ExpandedRow | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const LIMIT = 20;

  async function load(q: string, off: number) {
    setLoading(true);
    setError(null);
    try {
      const res = await getScanHistory(q || undefined, LIMIT, off);
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load(search, offset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offset]);

  function handleSearch() {
    setOffset(0);
    load(search, 0);
  }

  async function toggleExpand(item: ScanHistoryItem) {
    if (expanded?.scanId === item.scan_id) {
      setExpanded(null);
      return;
    }
    setExpanded({ scanId: item.scan_id, report: null, loading: true });
    try {
      const report = await getScanResult(item.scan_id);
      setExpanded({ scanId: item.scan_id, report, loading: false });
    } catch {
      setExpanded({ scanId: item.scan_id, report: null, loading: false });
    }
  }

  function toggleSelect(scanId: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(scanId)) {
        next.delete(scanId);
      } else if (next.size < 2) {
        next.add(scanId);
      }
      return next;
    });
  }

  const compareHref =
    selected.size === 2
      ? `/compare?a=${[...selected][0]}&b=${[...selected][1]}`
      : null;

  return (
    <div className="min-h-screen">
      <Nav />

      <main className="mx-auto max-w-6xl px-6 py-10">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-[--foreground]">
              ◷ Scan History
            </h1>
            <p className="text-sm text-[--muted-fg] mt-1">
              {total} scan{total !== 1 ? "s" : ""} stored locally
            </p>
          </div>
          {compareHref && (
            <Link href={compareHref}>
              <Button>⟺ Compare selected</Button>
            </Link>
          )}
        </div>

        {/* Search */}
        <div className="mb-6 flex gap-3 max-w-md">
          <Input
            placeholder="Filter by model ID…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
          <Button variant="secondary" onClick={handleSearch}>
            Search
          </Button>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Table */}
        <div className="rounded-xl border border-[--border] overflow-hidden">
          {/* Header row */}
          <div className="grid grid-cols-[2rem_1fr_8rem_8rem_8rem_6rem_6rem] items-center gap-4 px-4 py-3 bg-[--muted] text-xs font-medium text-[--muted-fg] uppercase tracking-wider">
            <span />
            <span>Model</span>
            <span>Date</span>
            <span className="text-right">Geo-ratio</span>
            <span>Top dimension</span>
            <span className="text-right">Score</span>
            <span className="text-center">Actions</span>
          </div>

          {loading && (
            <div className="py-16 text-center text-sm text-[--muted-fg]">
              Loading…
            </div>
          )}

          {!loading && items.length === 0 && (
            <div className="py-16 text-center text-sm text-[--muted-fg]">
              No scans found.{" "}
              <Link href="/scan" className="text-[--brand] hover:underline">
                Run a scan
              </Link>{" "}
              first.
            </div>
          )}

          {!loading &&
            items.map((item) => {
              const isExpanded = expanded?.scanId === item.scan_id;
              const isSelected = selected.has(item.scan_id);
              return (
                <div key={item.scan_id}>
                  {/* Main row */}
                  <div
                    className={[
                      "grid grid-cols-[2rem_1fr_8rem_8rem_8rem_6rem_6rem] items-center gap-4 px-4 py-3",
                      "border-t border-[--border] cursor-pointer hover:bg-[--card-hover] transition-colors",
                      isExpanded ? "bg-[--card]" : "",
                    ].join(" ")}
                    onClick={() => toggleExpand(item)}
                  >
                    {/* Select checkbox */}
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={(e) => {
                        e.stopPropagation();
                        toggleSelect(item.scan_id);
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="accent-[--brand] cursor-pointer"
                    />
                    <span
                      className="text-sm font-medium text-[--foreground] truncate"
                      title={item.model_id}
                    >
                      {item.model_id}
                    </span>
                    <span className="text-xs text-[--muted-fg]">
                      {relativeTime(item.created_at)}
                    </span>
                    <span className="text-xs text-right text-[--muted-fg]">
                      {item.geometric_separation_ratio.toFixed(0)}×
                    </span>
                    <Badge variant="muted" className="text-xs w-fit">
                      {item.top_score.dimension}
                    </Badge>
                    <span className="text-xs text-right text-[--foreground]">
                      {(item.top_score.score * 100).toFixed(0)}%
                    </span>
                    {/* Actions */}
                    <div
                      className="flex justify-center gap-1"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <a
                        href={`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/v1/scan/results/${item.scan_id}/export?fmt=json`}
                        download
                        title="Export JSON"
                        className="text-[--muted-fg] hover:text-[--foreground] transition-colors text-sm"
                      >
                        ↓
                      </a>
                    </div>
                  </div>

                  {/* Expanded radar */}
                  {isExpanded && (
                    <div className="border-t border-[--border] bg-[--card] px-8 py-6">
                      {expanded?.loading && (
                        <p className="text-sm text-[--muted-fg]">Loading…</p>
                      )}
                      {expanded?.report && (
                        <div className="max-w-sm mx-auto">
                          <BehavioralRadarChart scores={expanded.report.scores} />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
        </div>

        {/* Pagination */}
        {total > LIMIT && (
          <div className="mt-6 flex items-center justify-between text-sm text-[--muted-fg]">
            <span>
              Showing {offset + 1}–{Math.min(offset + LIMIT, total)} of {total}
            </span>
            <div className="flex gap-2">
              <Button
                variant="secondary"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - LIMIT))}
              >
                Previous
              </Button>
              <Button
                variant="secondary"
                disabled={offset + LIMIT >= total}
                onClick={() => setOffset(offset + LIMIT)}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
