"use client";

import { useCallback, useRef, useState } from "react";
import {
  createPipelineWebSocket,
  generateWorld,
  type PipelineEvent,
  type PipelineStage,
} from "./api";

export type StageStatus = "pending" | "active" | "done" | "error";

export interface LogEntry {
  id: string;
  timestamp: number;
  stage: PipelineStage;
  event: string;
  message: string;
  data?: Record<string, unknown>;
}

export interface PipelineState {
  isRunning: boolean;
  isDone: boolean;
  worldUrl: string | null;
  error: string | null;
  splatUrl: string | null;
  stages: Record<string, StageStatus>;
  logs: LogEntry[];
  vlmAnalysis: Record<string, unknown> | null;
  dreamPrompt: string | null;
  gaussianCount: number | null;
  viewCount: number | null;
  renderer: "local-splat";
  coverage: string | null;
}

const INITIAL_STATE: PipelineState = {
  isRunning: false,
  isDone: false,
  error: null,
  splatUrl: null,
  worldUrl: null,
  stages: {
    vlm: "pending",
    llm: "pending",
    multiview: "pending",
    reconstruction: "pending",
    stitch: "pending",
    difix: "pending",
    compile: "pending",
  },
  logs: [],
  vlmAnalysis: null,
  dreamPrompt: null,
  gaussianCount: null,
  viewCount: null,
  renderer: "local-splat",
  coverage: null,
};

export function usePipeline() {
  const [state, setState] = useState<PipelineState>(INITIAL_STATE);
  const wsRef = useRef<WebSocket | null>(null);
  const logIdRef = useRef(0);

  const addLog = useCallback(
    (entry: Omit<LogEntry, "id">) => {
      const id = `log-${logIdRef.current++}`;
      setState((prev) => ({
        ...prev,
        logs: [...prev.logs, { ...entry, id }],
      }));
    },
    [],
  );

  const handleEvent = useCallback(
    (event: PipelineEvent) => {
      const stage = event.stage;
      const eventType = event.event;

      // Add to log
      addLog({
        timestamp: event.timestamp || Date.now(),
        stage,
        event: eventType,
        message: event.message || event.error || "",
        data: event.data,
      });

      // Update stage status
      if (eventType === "stage_start") {
        setState((prev) => ({
          ...prev,
          stages: { ...prev.stages, [stage]: "active" },
        }));
      } else if (eventType === "stage_done") {
        setState((prev) => ({
          ...prev,
          stages: { ...prev.stages, [stage]: "done" },
        }));

        // Extract stage-specific data
        if (stage === "vlm" && event.data?.analysis) {
          setState((prev) => ({
            ...prev,
            vlmAnalysis: event.data!.analysis as Record<string, unknown>,
          }));
        }
        if (stage === "llm" && event.data?.prompt) {
          setState((prev) => ({
            ...prev,
            dreamPrompt: event.data!.prompt as string,
          }));
        }
        if (stage === "multiview" && event.data?.view_count) {
          setState((prev) => ({
            ...prev,
            viewCount: event.data!.view_count as number,
          }));
        }
        if (stage === "reconstruction" && event.data?.gaussian_count) {
          setState((prev) => ({
            ...prev,
            gaussianCount: event.data!.gaussian_count as number,
          }));
        }
      } else if (eventType === "pipeline_done") {
        setState((prev) => ({
          ...prev,
          isRunning: false,
          isDone: true,
          splatUrl: event.splat_url || null,
          renderer: "local-splat",
          worldUrl: event.world_url || (typeof event.data?.world_url === "string" ? event.data.world_url : null),
          coverage: typeof event.data?.coverage === "string" ? event.data.coverage : null,
        }));
      } else if (eventType === "error") {
        setState((prev) => ({
          ...prev,
          isRunning: false,
          error: event.error || "Unknown error",
          stages: { ...prev.stages, [stage]: "error" },
        }));
      }
    },
    [addLog],
  );

  const start = useCallback(
    async (image: File) => {
      setState({ ...INITIAL_STATE, isRunning: true });

      try {
        const { job_id } = await generateWorld(image);
        const ws = createPipelineWebSocket(job_id, handleEvent, () => {
          setState((prev) => ({ ...prev, isRunning: false }));
        });
        wsRef.current = ws;
      } catch (err) {
        setState((prev) => ({
          ...prev,
          isRunning: false,
          error: err instanceof Error ? err.message : "Failed to start pipeline",
        }));
      }
    },
    [handleEvent],
  );

  const connect = useCallback(
    (jobId: string) => {
      setState({ ...INITIAL_STATE, isRunning: true });
      const ws = createPipelineWebSocket(jobId, handleEvent, () => {
        setState((prev) => ({ ...prev, isRunning: false }));
      });
      wsRef.current = ws;
    },
    [handleEvent],
  );

  const loadSplat = useCallback(
    (splatUrl: string) => {
      setState({
        ...INITIAL_STATE,
        isDone: true,
        splatUrl,
        stages: {
          vlm: "pending",
          llm: "pending",
          multiview: "pending",
          reconstruction: "done",
          stitch: "pending",
          difix: "pending",
          compile: "done",
        },
      });
    },
    [],
  );

  const reset = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setState(INITIAL_STATE);
  }, []);

  return { state, start, connect, reset, loadSplat };
}
