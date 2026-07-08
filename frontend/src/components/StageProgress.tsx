"use client";

import { STAGE_ORDER, STAGE_LABELS, type PipelineStage } from "@/lib/api";
import type { StageStatus } from "@/lib/usePipeline";

interface StageProgressProps {
  stages: Record<string, StageStatus>;
}

const STATUS_STYLES: Record<StageStatus, string> = {
  pending: "bg-secondary/30 text-gray-600 border-secondary/20",
  active: "bg-accent/20 text-accent border-accent animate-pulse",
  done: "bg-success/20 text-success border-success/50",
  error: "bg-red-500/20 text-red-400 border-red-500/50",
};

export default function StageProgress({ stages }: StageProgressProps) {
  const doneCount = Object.values(stages).filter((s) => s === "done").length;
  const totalCount = STAGE_ORDER.length;
  const progressPercent = (doneCount / totalCount) * 100;

  return (
    <div className="flex flex-col gap-3">
      {/* Progress bar */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono text-gray-500 tracking-wider">PROGRESS</span>
        <div className="flex-1 h-1 bg-secondary/30 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-accent to-success transition-all duration-500 ease-out"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
        <span className="text-xs font-mono text-gray-400">
          {doneCount}/{totalCount}
        </span>
      </div>

      {/* Stage segments */}
      <div className="flex gap-2">
        {STAGE_ORDER.map((stage: PipelineStage) => {
          const status = stages[stage] || "pending";
          return (
            <div
              key={stage}
              className={`
                stage-segment ${status}
                flex-1 px-3 py-2 rounded border text-xs font-mono text-center
                transition-all duration-400
                ${STATUS_STYLES[status]}
              `}
            >
              {STAGE_LABELS[stage]}
            </div>
          );
        })}
      </div>
    </div>
  );
}
