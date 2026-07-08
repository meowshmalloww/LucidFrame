/**
 * API client for LucidFrame backend.
 * REST + WebSocket communication.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const WS_BASE = API_BASE.replace("http://", "ws://").replace("https://", "wss://");

export interface GenerateWorldResponse {
  job_id: string;
}

export interface HealthResponse {
  status: string;
  gpu: {
    available: boolean;
    name?: string;
    total_mb?: number;
    allocated_mb?: number;
    safe_budget_mb?: number;
    safe?: boolean;
    usage_percent?: number;
  };
  ram: {
    total_gb: number;
    used_gb: number;
    usage_percent: number;
    safe: boolean;
  };
  cpu: {
    percent: number;
    core_count: number;
    safe: boolean;
  };
  warnings: string[];
}

export async function generateWorld(image: File): Promise<GenerateWorldResponse> {
  const formData = new FormData();
  formData.append("image", image);

  const res = await fetch(`${API_BASE}/api/generate-world`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Failed to start pipeline: ${res.statusText}`);
  }

  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export function createPipelineWebSocket(
  jobId: string,
  onEvent: (event: PipelineEvent) => void,
  onClose?: () => void,
  onError?: (error: Event) => void,
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/ws/pipeline/${jobId}`);

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      onEvent(data);
    } catch (err) {
      console.error("Failed to parse WebSocket message:", err);
    }
  };

  ws.onclose = () => {
    onClose?.();
  };

  ws.onerror = (err) => {
    onError?.(err);
  };

  return ws;
}

export type PipelineEventType =
  | "stage_start"
  | "stage_progress"
  | "stage_done"
  | "pipeline_done"
  | "error"
  | "warning";

export type PipelineStage =
  | "vlm"
  | "llm"
  | "multiview"
  | "reconstruction"
  | "difix"
  | "compile"
  | "system"
  | "pipeline";

export interface PipelineEvent {
  event: PipelineEventType;
  stage: PipelineStage;
  message?: string;
  data?: Record<string, unknown>;
  timestamp?: number;
  job_id?: string;
  elapsed_sec?: number;
  splat_url?: string;
  error?: string;
}

export const STAGE_ORDER: PipelineStage[] = [
  "vlm",
  "llm",
  "multiview",
  "reconstruction",
  "difix",
  "compile",
];

export const STAGE_LABELS: Record<PipelineStage, string> = {
  vlm: "VLM Analysis",
  llm: "Dream Narrative",
  multiview: "Multi-View Synthesis",
  reconstruction: "3D Reconstruction",
  difix: "Artifact Fix",
  compile: "Compile .splat",
  system: "System",
  pipeline: "Pipeline",
};

export const SPLAT_URL_BASE = API_BASE;
