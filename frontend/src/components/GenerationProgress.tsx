"use client";

import { useEffect, useMemo, useState } from "react";
import { STAGE_ORDER, type PipelineStage } from "@/lib/api";
import type { PipelineState } from "@/lib/usePipeline";

const STEP_COPY: Record<PipelineStage, { title: string; detail: string }> = {
  restoration: {
    title: "Preparing the source",
    detail: "Preserving the image texture and enlarging it only when the input needs it.",
  },
  vlm: {
    title: "Reading the composition",
    detail: "Locating the horizon, important forms, and the structure that can be inferred from the source.",
  },
  llm: {
    title: "Setting the scene",
    detail: "Keeping the reconstruction grounded in the uploaded image.",
  },
  multiview: {
    title: "Extending the view",
    detail: "Preparing the directions needed by the selected reconstruction method.",
  },
  reconstruction: {
    title: "Building the 3D scene",
    detail: "Predicting depth and placing the Gaussian representation. This is usually the longest part.",
  },
  stitch: {
    title: "Checking coverage",
    detail: "Aligning overlapping regions and checking the scene for open seams.",
  },
  difix: {
    title: "Cleaning the result",
    detail: "Running the final artifact checks before export.",
  },
  compile: {
    title: "Saving the scene",
    detail: "Writing the splat file and preparing it for the viewer.",
  },
  system: { title: "Starting", detail: "Preparing the local generation process." },
  pipeline: { title: "Starting", detail: "Preparing the local generation process." },
};

const PHASES: Array<{ label: string; stages: PipelineStage[] }> = [
  { label: "Source", stages: ["restoration", "vlm", "llm"] },
  { label: "Views", stages: ["multiview"] },
  { label: "Geometry", stages: ["reconstruction", "stitch", "difix"] },
  { label: "Scene", stages: ["compile"] },
];

type Provider = "local" | "local_world" | "local_pano" | "worldlabs";

export function GenerationProgress({
  state,
  sourcePreview,
  provider,
}: {
  state: PipelineState;
  sourcePreview: string | null;
  provider: Provider;
}) {
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
    const percent = Math.round((completed / STAGE_ORDER.length) * 100);
    return { active, completed, percent };
  }, [state.stages]);

  const copy = STEP_COPY[progress.active];
  const currentMessage = [...state.logs].reverse().find((log) => log.message)?.message;

  return (
    <section
      className="absolute inset-3 z-30 overflow-auto rounded-[10px] border border-[#cbc9bf] bg-[#f7f6f0] text-[#1b1b18] sm:inset-4"
      aria-busy="true"
      aria-labelledby="generation-title"
    >
      <div className="grid min-h-full lg:grid-cols-[minmax(320px,0.92fr)_minmax(390px,1.08fr)]">
        <div className="relative min-h-[300px] overflow-hidden border-b border-[#d3d0c7] bg-[#dedbd2] lg:min-h-full lg:border-b-0 lg:border-r">
          {sourcePreview ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element -- source is served by the local FastAPI process. */}
              <img src={sourcePreview} alt="" className="absolute inset-0 h-full w-full scale-110 object-cover opacity-20 blur-2xl" aria-hidden />
              <div className="absolute inset-0 bg-[#25241f]/10" aria-hidden />
              <div className="absolute inset-7 grid place-items-center sm:inset-10">
                {/* eslint-disable-next-line @next/next/no-img-element -- source is served by the local FastAPI process. */}
                <img src={sourcePreview} alt="Uploaded source" className="max-h-full max-w-full border border-black/10 bg-[#f3f1e8] object-contain shadow-[0_22px_60px_rgba(32,30,24,0.22)]" />
              </div>
            </>
          ) : (
            <div className="grid h-full place-items-center p-10 text-center text-[#77766f]">
              <SourceIcon />
            </div>
          )}
        </div>

        <div className="flex min-h-[420px] flex-col justify-between p-6 sm:p-9 lg:p-11">
          <div>
            <div className="flex items-center justify-between gap-4 text-xs text-[#686760]">
              <span>{methodLabel(provider)}</span>
              <span>{formatTime(elapsed)}</span>
            </div>

            <div className="py-12 sm:py-16">
              <h2 id="generation-title" className="max-w-xl text-[clamp(2rem,3.7vw,3.25rem)] font-medium leading-[1.02] tracking-[-0.045em]">
                {copy.title}
              </h2>
              <p className="mt-5 max-h-16 max-w-lg overflow-hidden text-sm leading-6 text-[#686760]">
                {currentMessage || copy.detail}
              </p>
            </div>
          </div>

          <div>
            <div className="mb-3 flex items-center justify-between text-xs text-[#686760]">
              <span>Progress</span>
              <span>{progress.percent}%</span>
            </div>
            <div
              role="progressbar"
              aria-label="Scene generation progress"
              aria-valuemin={0}
              aria-valuemax={STAGE_ORDER.length}
              aria-valuenow={progress.completed}
              aria-valuetext={`${copy.title}. ${progress.completed} of ${STAGE_ORDER.length} tasks complete.`}
              className="h-1 overflow-hidden bg-[#d9d6cd]"
            >
              <div className="h-full bg-[#315c4a] transition-[width] duration-500 ease-out" style={{ width: `${progress.percent}%` }} />
            </div>
            <ol className="mt-4 flex flex-wrap gap-x-5 gap-y-2" aria-label="Generation phases">
              {PHASES.map((phase) => {
                const status = phaseStatus(phase.stages, state);
                return (
                  <li key={phase.label} className={status === "pending" ? "text-[11px] text-[#9a9890]" : status === "active" ? "text-[11px] font-semibold text-[#315c4a]" : "text-[11px] font-medium text-[#4f4e48]"}>
                    {phase.label}
                  </li>
                );
              })}
            </ol>
          </div>
        </div>
      </div>
      <span className="sr-only" role="status" aria-live="polite">{copy.title}. {copy.detail}</span>
    </section>
  );
}

function phaseStatus(stages: PipelineStage[], state: PipelineState): "pending" | "active" | "done" {
  if (stages.every((stage) => state.stages[stage] === "done")) return "done";
  if (stages.some((stage) => state.stages[stage] === "active" || state.stages[stage] === "done")) return "active";
  return "pending";
}

function methodLabel(provider: Provider) {
  if (provider === "local_world") return "Image to 360";
  if (provider === "local_pano") return "Panorama to 360";
  if (provider === "worldlabs") return "World Labs";
  return "Image to 3D";
}

function SourceIcon() {
  return (
    <svg viewBox="0 0 48 48" className="h-12 w-12 fill-none stroke-current stroke-[1.2]" aria-hidden>
      <rect x="6" y="8" width="36" height="30" rx="2" />
      <path d="m10 33 10-11 8 8 5-5 5 8" />
      <circle cx="16" cy="16" r="2.5" />
    </svg>
  );
}

function formatTime(seconds: number) {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}
