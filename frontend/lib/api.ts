/**
 * Prism API client — typed wrappers around the FastAPI backend.
 * Base URL is controlled by NEXT_PUBLIC_API_URL (default: http://localhost:8000).
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types — mirror the FastAPI response schemas
// ---------------------------------------------------------------------------

export interface ResolverResult {
  model_id: string;
  revision: string;
  model_type: string;
  architectures: string[];
  family: string;
  num_hidden_layers: number;
  hidden_size: number;
  intermediate_size: number;
  num_attention_heads: number;
  num_key_value_heads: number;
  head_dim: number;
  uses_gqa: boolean;
  is_moe: boolean;
  num_experts: number | null;
  num_experts_per_token: number | null;
  state_size: number | null;
  ssm_expansion_factor: number | null;
  vocab_size: number;
  param_count_estimate: number;
  model_size_gb_bf16: number;
  lora_rank_recommendation: number;
  lora_targets: string[];
  lora_targets_minimal: string[];
}

export interface DimensionScore {
  dimension: string;
  score: number;
  interpretation: string;
}

export interface ScanReport {
  model_id: string;
  geometric_separation_ratio: number;
  scan_duration_ms: number;
  scores: DimensionScore[];
}

export interface GenerateLoRAResult {
  job_id: string;
  status: string;
  model_id: string;
  lora_rank: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  adapter_config: Record<string, unknown>;
  training_yaml: string;
  trainable_params: number;
  estimated_size_mb: number;
}

export interface MonitorSession {
  session_id: string;
  model_id: string;
  websocket_url: string;
  telemetry_layers: number[];
}

export interface AgentChatResponse {
  session_id: string;
  reply: string;
  memory_hits: number;
  alpha_prime: number | null;
}

// Phase 6 — History + Compare + Dashboard

export interface ScanHistoryItem {
  scan_id: string;
  model_id: string;
  created_at: string;
  duration_ms: number;
  geometric_separation_ratio: number;
  top_score: DimensionScore;
}

export interface ScanHistoryResponse {
  items: ScanHistoryItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface DimensionDelta {
  dimension: string;
  score_a: number;
  score_b: number;
  delta: number;
  direction: string;
}

export interface CompareResult {
  scan_id_a: string;
  scan_id_b: string;
  model_a: string;
  model_b: string;
  deltas: DimensionDelta[];
  composite_distance: number;
  winner: string;
}

export interface DashboardStats {
  total_scans: number;
  total_jobs: number;
  total_sessions: number;
  unique_models: number;
  recent_scans: ScanHistoryItem[];
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function resolveModel(
  modelId: string,
  revision = "main",
): Promise<ResolverResult> {
  return apiFetch<ResolverResult>("/api/v1/resolver/resolve", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId, revision }),
  });
}

export async function scanModel(
  modelId: string,
  dimensions?: string[],
): Promise<ScanReport> {
  const body: Record<string, unknown> = { model_id: modelId };
  if (dimensions?.length) body.dimensions = dimensions;
  return apiFetch<ScanReport>("/api/v1/scan/run", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function generateLoRA(
  modelId: string,
  targets: Record<string, number>,
  loraRank?: number,
): Promise<GenerateLoRAResult> {
  return apiFetch<GenerateLoRAResult>("/api/v1/generate/lora", {
    method: "POST",
    body: JSON.stringify({
      model_id: modelId,
      behavioral_targets: targets,
      ...(loraRank != null && { lora_rank: loraRank }),
    }),
  });
}

export async function createMonitorSession(
  modelId: string,
  prompt?: string,
): Promise<MonitorSession> {
  return apiFetch<MonitorSession>("/api/v1/monitor/sessions", {
    method: "POST",
    body: JSON.stringify({
      model_id: modelId,
      ...(prompt != null && { prompt }),
    }),
  });
}

export async function agentChat(
  message: string,
  sessionId?: string,
  useMemory = true,
): Promise<AgentChatResponse> {
  return apiFetch<AgentChatResponse>("/api/v1/agent/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
      ...(sessionId != null && { session_id: sessionId }),
      use_memory: useMemory,
    }),
  });
}

// ---------------------------------------------------------------------------
// Phase 6 — History, Compare, Dashboard
// ---------------------------------------------------------------------------

export async function getScanHistory(
  modelId?: string,
  limit = 20,
  offset = 0,
): Promise<ScanHistoryResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (modelId) params.set("model_id", modelId);
  return apiFetch<ScanHistoryResponse>(`/api/v1/scan/history?${params}`);
}

export async function getScanResult(scanId: string): Promise<ScanReport> {
  return apiFetch<ScanReport>(`/api/v1/scan/results/${scanId}`);
}

export async function compareScans(
  scanA: string,
  scanB: string,
): Promise<CompareResult> {
  return apiFetch<CompareResult>("/api/v1/compare", {
    method: "POST",
    body: JSON.stringify({ scan_a: scanA, scan_b: scanB }),
  });
}

export async function getDashboardStats(): Promise<DashboardStats> {
  return apiFetch<DashboardStats>("/api/v1/dashboard/stats");
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function healthCheck(): Promise<{ status: string }> {
  return apiFetch<{ status: string }>("/health");
}
