"use client";

import { STAGE_ORDER, STAGE_LABELS, type PipelineStage } from "@/lib/api";
import type { StageStatus } from "@/lib/usePipeline";
import { CheckIcon, XIcon, DotIcon, LayersIcon } from "./Icons";

interface StageProgressProps {
  stages: Record<string, StageStatus>;
}

const STATUS_STYLES: Record<StageStatus, string> = {
  pending: "bg-white/[0.02] text-gray-600 border-white/5",
  active: "bg-accent/10 text-accent border-accent/30",
  done: "bg-success/10 text-success border-success/30",
  error: "bg-red-500/10 text-red-400 border-red-500/30",
};

function StatusIcon({ status }: { status: StageStatus }) {
  switch (status) {
    case "done":
      return <CheckIcon size={12} />;
    case "error":
      return <XIcon size={12} />;
    case "active":
      return <DotIcon size={12} className="animate-subtle-pulse" />;
    default:
      return <DotIcon size={12} className="opacity-30" />;
  }
}

export default function StageProgress({ stages }: StageProgressProps) {
  const doneCount = Object.values(stages).filter((s) => s === "done").length;
  const totalCount = STAGE_ORDER.length;
  const progressPercent = (doneCount / totalCount) * 100;

  return (
    <div className="flex flex-col gap-2.5">
      {/* Progress bar */}
      <div className="flex items-center gap-3">
        <LayersIcon size={14} className="text-gray-600" />
        <span className="text-xs font-mono text-gray-600 tracking-wider">PROGRESS</span>
        <div className="flex-1 h-[3px] bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-accent to-success transition-all duration-500 ease-out"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
        <span className="text-xs font-mono text-gray-500">
          {doneCount}/{totalCount}
        </span>
      </div>

      {/* Stage segments */}
      <div className="flex gap-1.5">
        {STAGE_ORDER.map((stage: PipelineStage) => {
          const status = stages[stage] || "pending";
          return (
            <div
              key={stage}
              className={`
                stage-segment ${status}
                flex-1 px-2.5 py-2 rounded-lg border text-xs font-mono text-center
                flex items-center justify-center gap-1.5
                ${STATUS_STYLES[status]}
              `}
            >
              <StatusIcon status={status} />
              <span className="truncate">{STAGE_LABELS[stage]}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
