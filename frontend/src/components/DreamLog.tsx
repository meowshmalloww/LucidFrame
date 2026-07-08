"use client";

import { useEffect, useRef } from "react";
import type { LogEntry } from "@/lib/usePipeline";
import { STAGE_LABELS, type PipelineStage } from "@/lib/api";
import {
  PlayIcon, CheckIcon, XIcon, AlertIcon, StarIcon, DotIcon, TerminalIcon,
} from "./Icons";

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

function EventIcon({ event, className }: { event: string; className?: string }) {
  const size = 12;
  switch (event) {
    case "stage_start":
      return <PlayIcon size={size} className={className} />;
    case "stage_done":
      return <CheckIcon size={size} className={className} />;
    case "pipeline_done":
      return <StarIcon size={size} className={className} />;
    case "error":
      return <XIcon size={size} className={className} />;
    case "warning":
      return <AlertIcon size={size} className={className} />;
    default:
      return <DotIcon size={size} className={className} />;
  }
}

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
    return `-> ${data.splat_url}`;
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
    <div className="flex flex-col h-full glass rounded-xl overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/5">
        <TerminalIcon size={14} className="text-success/60" />
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
          const label = STAGE_LABELS[log.stage as PipelineStage] || log.stage;

          return (
            <div key={log.id} className="leading-relaxed animate-fade-in flex items-start gap-1.5">
              <span className="text-gray-600 shrink-0">{formatTime(log.timestamp)}</span>
              <span className={`${color} shrink-0 mt-0.5`}>
                <EventIcon event={log.event} />
              </span>
              <span className="text-gray-500 shrink-0">[{label}]</span>
              <span className="text-gray-300">{log.message}</span>
              {log.data && Object.keys(log.data).length > 0 && (
                <div className={`pl-4 ${color} opacity-60 w-full`}>
                  {renderData(log.stage, log.data)}
                </div>
              )}
            </div>
          );
        })}

        {dreamPrompt && (
          <div className="mt-4 p-3 border border-purple-500/20 rounded-lg bg-purple-950/10">
            <div className="text-purple-400 text-xs mb-1.5 tracking-wider">MASTER SCENE PROMPT</div>
            <div className="text-purple-200/80 italic text-sm leading-relaxed">
              &ldquo;{dreamPrompt}&rdquo;
            </div>
          </div>
        )}

        {vlmAnalysis && (
          <div className="mt-2 p-3 border border-blue-500/20 rounded-lg bg-blue-950/10">
            <div className="text-blue-400 text-xs mb-1.5 tracking-wider">VLM VISUAL ANALYSIS</div>
            <div className="text-blue-200/70 text-xs space-y-0.5">
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
