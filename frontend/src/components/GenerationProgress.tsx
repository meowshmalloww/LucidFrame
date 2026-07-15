"use client";

import { useEffect, useMemo, useState } from "react";
import { STAGE_ORDER, type PipelineStage } from "@/lib/api";
import type { PipelineState } from "@/lib/usePipeline";

const STEP_COPY: Record<PipelineStage, { title: string; detail: string }> = {
  restoration: {
    title: "Preparing image",
    detail: "Checking the source and restoring small images when needed.",
  },
  vlm: {
    title: "Reading image",
    detail: "Finding the layout, lighting, and main objects in the source.",
  },
  llm: {
    title: "Planning scene",
    detail: "Preparing a consistent direction for the reconstruction.",
  },
  multiview: {
    title: "Building views",
    detail: "Preparing the views needed to cover the scene.",
  },
  reconstruction: {
    title: "Creating geometry",
    detail: "Predicting depth and placing Gaussian points. This is usually the longest step.",
  },
  stitch: {
    title: "Joining scene",
    detail: "Aligning overlapping views and checking their coverage.",
  },
  difix: {
    title: "Cleaning details",
    detail: "Reducing visible artifacts before the scene is saved.",
  },
  compile: {
    title: "Finishing",
    detail: "Writing the scene file and preparing it for the browser.",
  },
  system: { title: "Preparing", detail: "Starting the local generation process." },
  pipeline: { title: "Preparing", detail: "Starting the local generation process." },
};

export function GenerationProgress({ state }: { state: PipelineState }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const progress = useMemo(() => {
    const completed = STAGE_ORDER.filter((stage) => state.stages[stage] === "done").length;
    const active = STAGE_ORDER.find((stage) => state.stages[stage] === "active")
      || STAGE_ORDER.find((stage) => state.stages[stage] === "pending")
      || "compile";
    const activeIndex = Math.max(0, STAGE_ORDER.indexOf(active));
    const percent = Math.min(100, Math.round(((completed + (state.stages[active] === "active" ? 0.45 : 0)) / STAGE_ORDER.length) * 100));
    return { active, activeIndex, completed, percent };
  }, [state.stages]);

  const copy = STEP_COPY[progress.active];
  const recentMessages = state.logs
    .filter((log) => log.message)
    .slice(-4)
    .reverse();

  return (
    <div className="absolute inset-3 z-30 overflow-auto rounded-[10px] border border-[#cbc9bf] bg-[#f7f6f0] p-4 text-[#1b1b18] sm:inset-4 sm:p-6" aria-busy="true">
      <div className="mx-auto grid min-h-full w-full max-w-[1050px] items-center gap-7 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="generation-preview relative aspect-[16/11] min-h-[280px] overflow-hidden rounded-[8px] border border-[#cfcdc4] bg-[#dfddd5]" aria-hidden>
          <div className="absolute inset-x-[8%] bottom-[12%] h-[38%] border border-[#c5c2b8] bg-[#e9e7df]" />
          <div className="absolute left-[13%] top-[17%] h-[42%] w-[28%] border border-[#c7c4ba] bg-[#ebe9e2]" />
          <div className="absolute right-[12%] top-[12%] h-[50%] w-[34%] border border-[#c7c4ba] bg-[#e5e3db]" />
          <div className="skeleton-sweep absolute inset-0" />
        </div>

        <section aria-labelledby="generation-title">
          <p className="text-sm font-medium text-[#686760]">Creating your scene</p>
          <h2 id="generation-title" className="mt-2 text-3xl font-medium tracking-[-0.035em]">{copy.title}</h2>
          <p className="mt-3 text-sm leading-6 text-[#686760]">{copy.detail}</p>

          <div className="mt-7">
            <div
              role="progressbar"
              aria-label="Scene generation progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progress.percent}
              className="h-1.5 overflow-hidden bg-[#d9d6cd]"
            >
              <div className="h-full bg-[#315c4a] transition-[width] duration-500 ease-out" style={{ width: `${progress.percent}%` }} />
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-[#77766f]">
              <span>Step {Math.min(progress.activeIndex + 1, STAGE_ORDER.length)} of {STAGE_ORDER.length}</span>
              <span>{formatTime(elapsed)}</span>
            </div>
          </div>

          <details className="mt-7 border-t border-[#d5d3ca] pt-4 text-xs text-[#686760]">
            <summary className="cursor-pointer font-medium text-[#4f4e48]">Processing details</summary>
            <div className="mt-3 space-y-2 leading-5">
              {recentMessages.length > 0
                ? recentMessages.map((log) => <p key={log.id}>{log.message}</p>)
                : <p>The local process is starting.</p>}
            </div>
          </details>
        </section>
      </div>
      <span className="sr-only" role="status">{copy.title}. {copy.detail}</span>
    </div>
  );
}

function formatTime(seconds: number) {
  if (seconds < 60) return `${seconds}s elapsed`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}m ${remainder.toString().padStart(2, "0")}s elapsed`;
}
