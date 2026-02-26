"use client";

import { useState } from "react";
import { Nav } from "@/components/nav";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { resolveModel, type ResolverResult } from "@/lib/api";
import { fmtParams } from "@/lib/utils";

const EXAMPLE_MODELS = [
  "mistralai/Mistral-7B-Instruct-v0.2",
  "meta-llama/Meta-Llama-3-8B",
  "google/gemma-2b",
  "microsoft/phi-2",
];

function FamilyBadge({ family }: { family: string }) {
  const colorMap: Record<string, "brand" | "cyan" | "emerald" | "amber"> = {
    llama: "brand",
    mistral: "cyan",
    mixtral: "cyan",
    gemma: "emerald",
    phi: "emerald",
    mamba: "amber",
    falcon_mamba: "amber",
    jamba: "amber",
    t5: "emerald",
    gpt2: "brand",
    unknown: "muted" as unknown as "amber",
  };
  const variant = colorMap[family.toLowerCase()] ?? "muted";
  return <Badge variant={variant as "brand" | "cyan" | "emerald" | "amber"}>{family}</Badge>;
}

function StatRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 py-2 border-b border-[--border] last:border-0">
      <span className="text-xs text-[--muted-fg] shrink-0">{label}</span>
      <span className="text-xs text-right font-medium text-[--foreground] break-all">{value}</span>
    </div>
  );
}

function ResultPanel({ result }: { result: ResolverResult }) {
  return (
    <div className="mt-8 space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold text-[--foreground]">{result.model_id}</h2>
        <FamilyBadge family={result.family} />
        {result.is_moe && <Badge variant="amber">MoE</Badge>}
        {result.uses_gqa && <Badge variant="emerald">GQA</Badge>}
        {result.state_size != null && <Badge variant="cyan">SSM</Badge>}
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {/* Architecture stats */}
        <Card>
          <CardHeader><CardTitle>Architecture</CardTitle></CardHeader>
          <StatRow label="Model type" value={result.model_type} />
          <StatRow label="Architectures" value={result.architectures.join(", ")} />
          <StatRow label="Family" value={result.family} />
          <StatRow label="Layers" value={result.num_hidden_layers} />
          <StatRow label="Hidden size" value={result.hidden_size.toLocaleString()} />
          <StatRow label="FFN size" value={result.intermediate_size.toLocaleString()} />
          <StatRow label="Vocab size" value={result.vocab_size.toLocaleString()} />
        </Card>

        {/* Attention */}
        <Card>
          <CardHeader><CardTitle>Attention</CardTitle></CardHeader>
          <StatRow label="Attn heads" value={result.num_attention_heads} />
          <StatRow label="KV heads" value={result.num_key_value_heads} />
          <StatRow label="Head dim" value={result.head_dim} />
          <StatRow label="GQA" value={result.uses_gqa ? "Yes" : "No"} />
          {result.is_moe && <>
            <StatRow label="Experts total" value={result.num_experts ?? "—"} />
            <StatRow label="Experts/token" value={result.num_experts_per_token ?? "—"} />
          </>}
          {result.state_size != null && <>
            <StatRow label="State size" value={result.state_size} />
            <StatRow label="SSM expansion" value={result.ssm_expansion_factor ?? "—"} />
          </>}
        </Card>

        {/* Size & LoRA */}
        <Card>
          <CardHeader><CardTitle>Scale &amp; LoRA</CardTitle></CardHeader>
          <StatRow label="Param count (est.)" value={fmtParams(result.param_count_estimate)} />
          <StatRow label="Model size BF16" value={`${result.model_size_gb_bf16.toFixed(1)} GB`} />
          <StatRow label="LoRA rank rec." value={
            <span className="text-[--brand] font-bold">{result.lora_rank_recommendation}</span>
          } />
          <StatRow label="Revision" value={result.revision} />
        </Card>
      </div>

      {/* LoRA targets */}
      <Card>
        <CardHeader><CardTitle>LoRA Target Modules</CardTitle></CardHeader>
        <div className="mb-3">
          <p className="text-xs text-[--muted-fg] mb-2">Full targets ({result.lora_targets.length})</p>
          <div className="flex flex-wrap gap-1.5">
            {result.lora_targets.map((t) => (
              <code key={t} className="rounded-md border border-[--border] bg-[--muted] px-2 py-0.5 text-xs text-[--accent-cyan]">
                {t}
              </code>
            ))}
          </div>
        </div>
        <div>
          <p className="text-xs text-[--muted-fg] mb-2">Minimal Q+V targets ({result.lora_targets_minimal.length})</p>
          <div className="flex flex-wrap gap-1.5">
            {result.lora_targets_minimal.map((t) => (
              <code key={t} className="rounded-md border border-[--brand]/30 bg-[--brand]/10 px-2 py-0.5 text-xs text-[--brand]">
                {t}
              </code>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}

export default function ResolvePage() {
  const [modelId, setModelId] = useState("");
  const [revision, setRevision] = useState("main");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResolverResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleResolve(e: React.FormEvent) {
    e.preventDefault();
    if (!modelId.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await resolveModel(modelId.trim(), revision);
      setResult(data);
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
            <span className="text-[--brand]">⬡</span> Auto-Resolver
          </h1>
          <p className="mt-1 text-sm text-[--muted-fg]">
            Detect architecture, estimate parameters, and get LoRA target recommendations — config.json only, no weights.
          </p>
        </div>

        {/* Form */}
        <Card>
          <form onSubmit={handleResolve} className="flex flex-col gap-4 sm:flex-row sm:items-end">
            <div className="flex-1">
              <label className="mb-1.5 block text-xs font-medium text-[--muted-fg]">
                HuggingFace Model ID
              </label>
              <Input
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="e.g. mistralai/Mistral-7B-Instruct-v0.2"
                disabled={loading}
              />
            </div>
            <div className="sm:w-36">
              <label className="mb-1.5 block text-xs font-medium text-[--muted-fg]">
                Revision
              </label>
              <Input
                value={revision}
                onChange={(e) => setRevision(e.target.value)}
                placeholder="main"
                disabled={loading}
              />
            </div>
            <Button type="submit" loading={loading} disabled={!modelId.trim()}>
              Resolve
            </Button>
          </form>

          {/* Example quick-picks */}
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
        {result && <ResultPanel result={result} />}
      </main>
    </div>
  );
}
