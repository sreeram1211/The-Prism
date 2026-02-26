"use client";

import { useEffect, useRef, useState } from "react";
import { Nav } from "@/components/nav";
import { agentChat, type AgentChatResponse } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  memoryHits?: number;
  alphaPrime?: number | null;
}

function AlphaBadge({ alpha }: { alpha: number | null | undefined }) {
  if (alpha == null) return null;
  const color = alpha > 0.02 ? "text-emerald-400" : alpha < -0.02 ? "text-red-400" : "text-amber-400";
  const label = alpha > 0.02 ? "↑ accelerating" : alpha < -0.02 ? "↓ plateauing" : "→ steady";
  return (
    <span className={`text-[10px] font-mono ${color}`}>
      α′={alpha.toFixed(4)} {label}
    </span>
  );
}

export default function AgentPage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useMemory, setUseMemory] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const msg = input.trim();
    if (!msg || loading) return;
    setInput("");
    setError(null);

    const userMsg: Message = { role: "user", content: msg };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const resp = await agentChat(msg, sessionId ?? undefined, useMemory);
      if (!sessionId) setSessionId(resp.session_id);

      const assistantMsg: Message = {
        role: "assistant",
        content: resp.reply,
        memoryHits: resp.memory_hits,
        alphaPrime: resp.alpha_prime,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Chat failed");
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setMessages([]);
    setSessionId(null);
    setError(null);
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Nav />
      <main className="flex-1 mx-auto w-full max-w-3xl px-6 py-10 flex flex-col">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-[--foreground]">
            <span className="text-violet-400">◈</span> Prism Agent
          </h1>
          <p className="mt-1 text-sm text-[--muted-fg]">
            Persistent-memory chat with RSI α′ improvement tracking.
          </p>
        </div>

        {/* Controls */}
        <div className="mb-4 flex items-center justify-between">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <div
              onClick={() => setUseMemory((v) => !v)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                useMemory ? "bg-[--brand]" : "bg-[--border]"
              }`}
            >
              <span
                className={`inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform ${
                  useMemory ? "translate-x-4" : "translate-x-1"
                }`}
              />
            </div>
            <span className="text-xs text-[--muted-fg]">Memory retrieval</span>
          </label>

          <div className="flex items-center gap-3 text-xs text-[--muted-fg]">
            {sessionId && (
              <span className="font-mono">
                session: <span className="text-[--foreground]">{sessionId}</span>
              </span>
            )}
            {messages.length > 0 && (
              <button onClick={handleClear} className="hover:text-[--foreground] transition-colors">
                ↺ New session
              </button>
            )}
          </div>
        </div>

        {/* Message list */}
        <div className="flex-1 overflow-y-auto rounded-xl border border-[--border] bg-[--card]
                        p-4 space-y-4 min-h-[320px] max-h-[500px]">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center text-center text-sm text-[--muted-fg] p-8">
              <div>
                <div className="text-3xl mb-3">◈</div>
                <p className="max-w-xs">
                  Ask anything about Prism — architecture resolution, behavioral scanning,
                  LoRA generation, the manifold geometry, or training configuration.
                </p>
              </div>
            </div>
          ) : (
            messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    m.role === "user"
                      ? "bg-[--brand] text-white rounded-br-sm"
                      : "bg-[--background] border border-[--border] text-[--foreground] rounded-bl-sm"
                  }`}
                >
                  {m.content}
                  {m.role === "assistant" && (m.memoryHits != null || m.alphaPrime != null) && (
                    <div className="mt-2 flex flex-wrap items-center gap-3 border-t border-[--border]/50 pt-1.5">
                      {(m.memoryHits ?? 0) > 0 && (
                        <span className="text-[10px] text-[--muted-fg]">
                          {m.memoryHits} memory hit{m.memoryHits === 1 ? "" : "s"}
                        </span>
                      )}
                      <AlphaBadge alpha={m.alphaPrime} />
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-sm bg-[--background] border border-[--border] px-4 py-3">
                <span className="inline-flex gap-1 text-[--muted-fg]">
                  {[0, 1, 2].map((i) => (
                    <span
                      key={i}
                      className="h-1.5 w-1.5 rounded-full bg-current animate-bounce"
                      style={{ animationDelay: `${i * 0.15}s` }}
                    />
                  ))}
                </span>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {error && (
          <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Input bar */}
        <div className="mt-3 flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
            placeholder="Ask about LoRA, behavioral scan dimensions, the manifold…"
            disabled={loading}
            className="flex-1 rounded-xl border border-[--border] bg-[--card] px-4 py-3 text-sm
                       text-[--foreground] placeholder:text-[--muted-fg] focus:outline-none
                       focus:ring-1 focus:ring-[--brand]/50 disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="rounded-xl bg-[--brand] px-5 py-3 text-sm font-semibold text-white
                       hover:bg-[--brand]/80 disabled:opacity-50 disabled:cursor-not-allowed
                       transition-colors"
          >
            ↑
          </button>
        </div>
        <p className="mt-2 text-center text-xs text-[--muted-fg]">
          Enter ↵ to send · memory retrieval injects relevant past context into each reply
        </p>
      </main>
    </div>
  );
}
