"use client";

import { useEffect, useRef } from "react";
import type { LogEntry } from "@/lib/usePipeline";
import { STAGE_LABELS, type PipelineStage } from "@/lib/api";

interface DreamLogProps {
  logs: LogEntry[];
  vlmAnalysis: Record<string, unknown> | null;
  dreamPrompt: string | null;
}

const STAGE_COLORS: Record<string, string> = {
  vlm: "text-blue-400",
  llm: "text-purple-400",
  multiview: "text-cyan-400",
  reconstruction: "text-amber-400",
  difix: "text-pink-400",
  compile: "text-success",
  system: "text-yellow-400",
  pipeline: "text-accent",
};

const EVENT_ICONS: Record<string, string> = {
  stage_start: "▶",
  stage_progress: "·",
  stage_done: "✓",
  pipeline_done: "★",
  error: "✗",
  warning: "⚠",
};

function formatTime(timestamp: number): string {
  const d = new Date(timestamp * 1000);
  return d.toLocaleTimeString("en-US", { hour12: false });
}

function renderData(stage: string, data: Record<string, unknown>): string {
  if (stage === "vlm" && data.analysis) {
    const a = data.analysis as Record<string, unknown>;
    const parts: string[] = [];
    if (a.era) parts.push(`era: ${a.era}`);
    if (a.style) parts.push(`style: ${a.style}`);
    if (a.mood) parts.push(`mood: ${a.mood}`);
    if (a.lighting) parts.push(`lighting: ${a.lighting}`);
    if (a.atmosphere) parts.push(`atmosphere: ${a.atmosphere}`);
    return parts.join(" | ");
  }
  if (stage === "llm" && data.prompt) {
    return `"${data.prompt}"`;
  }
  if (stage === "multiview" && data.view_count) {
    return `${data.view_count} views generated`;
  }
  if (stage === "reconstruction" && data.gaussian_count) {
    return `${data.gaussian_count} Gaussians`;
  }
  if (stage === "compile" && data.splat_url) {
    return `→ ${data.splat_url}`;
  }
  return JSON.stringify(data, null, 2).slice(0, 200);
}

export default function DreamLog({ logs, vlmAnalysis, dreamPrompt }: DreamLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="flex flex-col h-full bg-panel/50 rounded-lg border border-secondary/30">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-secondary/30">
        <span className="text-xs font-mono text-success">●</span>
        <span className="text-xs font-mono text-gray-400 tracking-wider">DREAM_LOG</span>
        <span className="text-xs font-mono text-gray-600 ml-auto">{logs.length} events</span>
      </div>

      <div
        ref={scrollRef}
        className="dream-log flex-1 overflow-y-auto p-4 font-mono text-xs space-y-1"
      >
        {logs.length === 0 && (
          <div className="text-gray-600 italic">
            Waiting for pipeline to start...
          </div>
        )}

        {logs.map((log) => {
          const color = STAGE_COLORS[log.stage] || "text-gray-400";
          const icon = EVENT_ICONS[log.event] || "·";
          const label = STAGE_LABELS[log.stage as PipelineStage] || log.stage;

          return (
            <div key={log.id} className="leading-relaxed animate-fade-in">
              <span className="text-gray-600">{formatTime(log.timestamp)}</span>{" "}
              <span className={color}>{icon}</span>{" "}
              <span className="text-gray-500">[{label}]</span>{" "}
              <span className="text-gray-300">{log.message}</span>
              {log.data && Object.keys(log.data).length > 0 && (
                <div className={`pl-6 ${color} opacity-70`}>
                  {renderData(log.stage, log.data)}
                </div>
              )}
            </div>
          );
        })}

        {dreamPrompt && (
          <div className="mt-4 p-3 border border-purple-500/30 rounded bg-purple-950/20">
            <div className="text-purple-400 text-xs mb-1">MASTER SCENE PROMPT</div>
            <div className="text-purple-200 italic text-sm leading-relaxed">
              &ldquo;{dreamPrompt}&rdquo;
            </div>
          </div>
        )}

        {vlmAnalysis && (
          <div className="mt-2 p-3 border border-blue-500/30 rounded bg-blue-950/20">
            <div className="text-blue-400 text-xs mb-1">VLM VISUAL ANALYSIS</div>
            <div className="text-blue-200 text-xs space-y-0.5">
              {Object.entries(vlmAnalysis).map(([key, val]) => (
                <div key={key}>
                  <span className="text-blue-500">{key}:</span>{" "}
                  <span>{typeof val === "string" ? val : JSON.stringify(val)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
