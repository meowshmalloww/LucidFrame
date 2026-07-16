/**
 * API client for LucidFrame backend.
 * REST + WebSocket communication.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const WS_BASE = API_BASE.replace("http://", "ws://").replace("https://", "wss://");

export type PipelineProvider = "local" | "local_world" | "local_pano" | "worldlabs";
export type QualityProfile = "balanced" | "detail";
export type SourceProfile = "artwork" | "photo";

export interface SceneCameraMetadata {
  horizontal_fov_deg: number;
  focal_px: number;
  move_speed_mps: number;
  near_depth_m?: number;
  recommended_lateral_m?: number;
  recommended_forward_m?: number;
}

export interface SceneMetadata {
  camera?: SceneCameraMetadata | null;
  coverage?: string;
  provider?: PipelineProvider;
}

export interface GenerateWorldResponse {
  job_id: string;
  provider: PipelineProvider;
  source_url: string;
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

export async function generateWorld(
  image: File,
  provider: PipelineProvider = "local",
  creativeDirection = "",
  qualityProfile: QualityProfile = "balanced",
  sourceProfile: SourceProfile = "artwork",
): Promise<GenerateWorldResponse> {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("provider", provider);
  formData.append("creative_direction", creativeDirection);
  formData.append("quality_profile", qualityProfile);
  formData.append("source_profile", sourceProfile);

  const res = await fetch(`${API_BASE}/api/generate-world`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let detail = res.statusText || "The generator did not accept this request.";
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Preserve the HTTP status text if the error is not JSON.
    }
    throw new Error(detail);
  }

  return res.json();
}


export interface ProviderStatus {
  available: boolean;
  label: string;
  estimated_time: string;
  description?: string;
  requires_credits?: boolean;
  input?: string;
  model?: string;
  estimated_credits?: number;
  estimate_label?: string;
  license_note?: string;
}

export async function getProviders(): Promise<Record<PipelineProvider, ProviderStatus>> {
  const res = await fetch(`${API_BASE}/api/providers`);
  if (!res.ok) throw new Error("Could not read generation modes");
  return res.json();
}

export interface WorldLabsModelOption {
  id: "marble-1.0-draft" | "marble-1.1" | "marble-1.1-plus";
  label: string;
  estimated_credits: number;
  max_estimated_credits?: number;
  estimate_label: string;
  quality: string;
}

export interface WorldLabsSettings {
  available: boolean;
  key_source: "session" | "environment" | null;
  session_only: boolean;
  model: WorldLabsModelOption["id"];
  credit_balance: number | null;
  model_options: WorldLabsModelOption[];
  input: string;
}

export async function getWorldLabsSettings(): Promise<WorldLabsSettings> {
  const res = await fetch(`${API_BASE}/api/settings/worldlabs`);
  if (!res.ok) throw new Error("Could not read World Labs settings.");
  return res.json();
}

export async function saveWorldLabsSettings(apiKey: string, model: WorldLabsModelOption["id"]): Promise<WorldLabsSettings> {
  const res = await fetch(`${API_BASE}/api/settings/worldlabs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey, model }),
  });
  if (!res.ok) {
    let detail = "Could not validate the World Labs API key.";
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export async function clearWorldLabsSettings(): Promise<WorldLabsSettings> {
  const res = await fetch(`${API_BASE}/api/settings/worldlabs`, { method: "DELETE" });
  if (!res.ok) throw new Error("Could not clear the World Labs session key.");
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
  | "restoration"
  | "vlm"
  | "llm"
  | "multiview"
  | "reconstruction"
  | "stitch"
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
  world_url?: string;
  job_id?: string;
  elapsed_sec?: number;
  splat_url?: string;
  error?: string;
}

export const STAGE_ORDER: PipelineStage[] = [
  "restoration",
  "vlm",
  "llm",
  "multiview",
  "reconstruction",
  "stitch",
  "difix",
  "compile",
];

export const STAGE_LABELS: Record<PipelineStage, string> = {
  restoration: "Source Restoration",
  vlm: "Image Composition",
  llm: "Dream Direction",
  multiview: "Scene Input",
  reconstruction: "Neural Reconstruction",
  stitch: "World Repair (optional)",
  difix: "Artifact Pass (optional)",
  compile: "Compile .splat",
  system: "System",
  pipeline: "Pipeline",
};

export const SPLAT_URL_BASE = API_BASE;

export interface GallerySplat {
  name: string;
  url: string;
  size_kb: number;
  id?: string;
  source_url?: string | null;
  provider?: PipelineProvider;
  coverage?: string;
  camera?: SceneCameraMetadata | null;
  created_at?: number;
}

export async function getSceneMetadata(jobId: string): Promise<SceneMetadata | null> {
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(jobId)) return null;
  try {
    const res = await fetch(`${API_BASE}/api/scenes/${encodeURIComponent(jobId)}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function getGallery(): Promise<GallerySplat[]> {
  try {
    const res = await fetch(`${API_BASE}/api/gallery`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.splats || [];
  } catch {
    return [];
  }
}

export interface DeleteGalleryResult {
  deleted: string[];
  missing: string[];
  freed_bytes: number;
}

export async function deleteGalleryProjects(ids: string[]): Promise<DeleteGalleryResult> {
  const safeIds = Array.from(new Set(ids.filter(Boolean)));
  if (safeIds.length === 0) return { deleted: [], missing: [], freed_bytes: 0 };
  const res = await fetch(`${API_BASE}/api/gallery:delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: safeIds }),
  });
  if (!res.ok) {
    let detail = "The selected scenes could not be deleted.";
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}
