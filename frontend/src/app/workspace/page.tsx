"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import SplatViewer, { type SplatViewerHandle } from "@/components/SplatViewer";
import { usePipeline } from "@/lib/usePipeline";
import { STAGE_LABELS, STAGE_ORDER, type PipelineStage } from "@/lib/api";

export default function WorkspacePage() {
  const router = useRouter();
  const { state, connect, loadSplat } = usePipeline();
  const splatViewerRef = useRef<SplatViewerHandle>(null);
  const [activeTab, setActiveTab] = useState<"scene" | "camera" | "export" | "assets">("scene");
  const [cameraMode, setCameraMode] = useState<"orbit" | "fly" | "ortho" | "persp">("orbit");
  const [fov, setFov] = useState(60);
  const [exposure, setExposure] = useState(1);
  const [pointSize, setPointSize] = useState(1);
  const [opacity, setOpacity] = useState(1);

  // Check for job_id or splat URL from session storage on mount
  useEffect(() => {
    if (typeof window === "undefined") return;
    const jobId = sessionStorage.getItem("lucidframe-job-id");
    const storedUrl = sessionStorage.getItem("lucidframe-splat-url");
    if (jobId && !state.splatUrl && !state.isRunning) {
      // Connect to WebSocket for this job
      sessionStorage.removeItem("lucidframe-job-id");
      connect(jobId);
    } else if (storedUrl && !state.splatUrl && !state.isRunning) {
      loadSplat(storedUrl);
    }
  }, []);

  const splatUrl = state.splatUrl || (typeof window !== "undefined" ? sessionStorage.getItem("lucidframe-splat-url") : null);
  const projectName = (typeof window !== "undefined" ? sessionStorage.getItem("lucidframe-splat-name") : null) || "Untitled";

  const isRunning = state.isRunning;
  const isDone = state.isDone || !!splatUrl;

  const handleNewProject = () => {
    router.push("/");
  };

  const handleResetCamera = () => {
    splatViewerRef.current?.resetCamera();
  };

  const handleSetFov = (value: number) => {
    setFov(value);
    splatViewerRef.current?.setFov(value);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Top bar */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-white/[0.06] bg-surface px-4">
        <div className="flex items-center gap-3">
          <span className="text-[14px] font-semibold text-white">{projectName}</span>
          {isRunning && (
            <span className="flex items-center gap-1.5 bg-white/10 px-2.5 py-0.5 text-[11px] font-medium text-white">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" />
              Generating
            </span>
          )}
          {isDone && !isRunning && (
            <span className="flex items-center gap-1.5 rounded-full bg-success/10 px-2.5 py-0.5 text-[11px] font-medium text-success">
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              Complete
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {splatUrl && (
            <a
              href={splatUrl}
              download
              className="flex items-center gap-1.5 rounded-btn border border-white/[0.08] bg-elevated px-3 py-1.5 text-[13px] font-medium text-zinc-200 transition-colors hover:bg-white/[0.06]"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Download
            </a>
          )}
          <button className="flex items-center gap-1.5 rounded-btn border border-white/[0.08] bg-elevated px-3 py-1.5 text-[13px] font-medium text-zinc-200 transition-colors hover:bg-white/[0.06]">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="18" cy="5" r="3" />
              <circle cx="6" cy="12" r="3" />
              <circle cx="18" cy="19" r="3" />
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
              <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
            </svg>
            Share
          </button>
          <button
            onClick={handleNewProject}
            className="flex items-center gap-1.5 rounded-btn bg-white px-3 py-1.5 text-[13px] font-medium text-black transition-colors hover:bg-zinc-200"
          >
            New Project
          </button>
        </div>
      </header>

      {/* Three columns */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Pipeline Activity */}
        <div className="flex w-[340px] shrink-0 flex-col border-r border-white/[0.06] bg-surface">
          <div className="flex h-9 shrink-0 items-center border-b border-white/[0.04] px-4">
            <span className="text-[12px] font-medium uppercase tracking-wider text-zinc-500">Pipeline Activity</span>
          </div>
          <div className="flex-1 overflow-y-auto thin-scroll p-3">
            <div className="flex flex-col gap-1.5">
              {STAGE_ORDER.map((stage: PipelineStage) => {
                const status = state.stages[stage] || "pending";
                const logs = state.logs.filter((l) => l.stage === stage);
                return (
                  <ActivityCard
                    key={stage}
                    stage={stage}
                    label={STAGE_LABELS[stage]}
                    status={status}
                    logs={logs}
                  />
                );
              })}
            </div>
          </div>
        </div>

        {/* Center: 3D Viewer */}
        <div className="relative flex flex-1 flex-col bg-void">
          {/* Top overlay — camera controls */}
          <div className="absolute left-1/2 top-3 z-10 flex -translate-x-1/2 items-center gap-1 rounded-btn border border-white/[0.06] bg-surface/80 p-1 backdrop-blur-sm">
            {(["orbit", "fly", "ortho", "persp"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setCameraMode(mode)}
                className={`rounded-[10px] px-3 py-1 text-[12px] font-medium capitalize transition-colors ${
                  cameraMode === mode
                    ? "bg-white/[0.08] text-white"
                    : "text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {mode === "persp" ? "Perspective" : mode === "ortho" ? "Orthographic" : mode}
              </button>
            ))}
            <div className="mx-1 h-4 w-px bg-white/[0.06]" />
            <button
              onClick={handleResetCamera}
              className="rounded-[10px] px-3 py-1 text-[12px] font-medium text-zinc-500 transition-colors hover:text-zinc-300"
            >
              Reset
            </button>
            <button className="rounded-[10px] px-3 py-1 text-[12px] font-medium text-zinc-500 transition-colors hover:text-zinc-300">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
              </svg>
            </button>
          </div>

          {/* Stats overlay — top left */}
          {splatUrl && (
            <div className="absolute left-3 top-3 z-10 flex flex-col gap-1 rounded-btn border border-white/[0.06] bg-surface/80 p-2.5 text-[11px] backdrop-blur-sm">
              <div className="flex items-center justify-between gap-6">
                <span className="text-zinc-500">FPS</span>
                <span className="font-mono text-zinc-300">60</span>
              </div>
              <div className="flex items-center justify-between gap-6">
                <span className="text-zinc-500">Vertices</span>
                <span className="font-mono text-zinc-300">
                  {state.gaussianCount?.toLocaleString() || "—"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-6">
                <span className="text-zinc-500">Gaussians</span>
                <span className="font-mono text-zinc-300">
                  {state.gaussianCount?.toLocaleString() || "—"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-6">
                <span className="text-zinc-500">Memory</span>
                <span className="font-mono text-zinc-300">— MB</span>
              </div>
            </div>
          )}

          {/* Loading state */}
          {isRunning && !splatUrl && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/[0.06] border-t-accent" />
              <p className="text-[14px] text-zinc-500">Generating 3D world...</p>
            </div>
          )}

          {/* Splat viewer */}
          <div className="flex-1">
            <SplatViewer
              ref={splatViewerRef}
              splatUrl={splatUrl}
              className="h-full w-full"
              mode={cameraMode === "fly" ? "fps" : "orbit"}
              onModeChange={() => {}}
            />
          </div>

          {/* Bottom overlay — sliders */}
          {splatUrl && (
            <div className="absolute bottom-3 left-1/2 z-10 flex -translate-x-1/2 items-center gap-4 rounded-btn border border-white/[0.06] bg-surface/80 px-4 py-2.5 backdrop-blur-sm">
              <SliderControl label="Exposure" value={exposure} min={0} max={2} step={0.05} onChange={setExposure} />
              <div className="h-4 w-px bg-white/[0.06]" />
              <SliderControl label="Point" value={pointSize} min={0.5} max={3} step={0.1} onChange={setPointSize} />
              <div className="h-4 w-px bg-white/[0.06]" />
              <SliderControl label="Opacity" value={opacity} min={0} max={1} step={0.05} onChange={setOpacity} />
            </div>
          )}
        </div>

        {/* Right: Inspector */}
        <div className="flex w-[280px] shrink-0 flex-col border-l border-white/[0.06] bg-surface">
          {/* Tabs */}
          <div className="flex h-9 shrink-0 items-center gap-1 border-b border-white/[0.04] px-2">
            {(["scene", "camera", "export", "assets"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`rounded-[8px] px-2.5 py-1 text-[12px] font-medium capitalize transition-colors ${
                  activeTab === tab
                    ? "bg-white/[0.08] text-white"
                    : "text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto thin-scroll p-4">
            {activeTab === "scene" && (
              <div className="flex flex-col gap-4">
                <InspectorField label="Bounding box" value="—" />
                <InspectorField label="Scene size" value="—" />
                <InspectorField label="Lighting" value="auto" />
                <InspectorField
                  label="Gaussian count"
                  value={state.gaussianCount?.toLocaleString() || "—"}
                />
              </div>
            )}
            {activeTab === "camera" && (
              <div className="flex flex-col gap-4">
                <div className="flex flex-col gap-2">
                  <label className="text-[12px] font-medium text-zinc-400">FOV</label>
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min="30"
                      max="100"
                      value={fov}
                      onChange={(e) => handleSetFov(Number(e.target.value))}
                      className="flex-1 accent-accent"
                    />
                    <span className="w-10 text-right text-[12px] font-mono text-zinc-300">{fov}°</span>
                  </div>
                </div>
                <InspectorField label="Near" value="0.1" />
                <InspectorField label="Far" value="100" />
                <InspectorField label="Speed" value="1.0" />
                <InspectorField label="Sensitivity" value="1.0" />
              </div>
            )}
            {activeTab === "export" && (
              <div className="flex flex-col gap-2">
                <ExportButton label="Download .splat" />
                <ExportButton label="Download PLY" />
                <ExportButton label="Export OBJ" />
                <ExportButton label="Generate Video" />
              </div>
            )}
            {activeTab === "assets" && (
              <div className="grid grid-cols-2 gap-2">
                {state.viewCount ? (
                  Array.from({ length: state.viewCount }).map((_, i) => (
                    <div
                      key={i}
                      className="aspect-square rounded-card border border-white/[0.04] bg-elevated"
                    />
                  ))
                ) : (
                  <p className="col-span-2 text-[13px] text-zinc-600">
                    Generated multi-view images will appear here.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom status bar */}
      <footer className="flex h-6 shrink-0 items-center justify-between border-t border-white/[0.06] bg-surface px-4 text-[11px] text-zinc-600">
        <div className="flex items-center gap-4">
          <span>Duration: —</span>
          <span>GPU: RTX 4080</span>
          <span>Server: Connected</span>
        </div>
        <div className="flex items-center gap-4">
          <span>Latency: —</span>
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-success" />
            Online
          </span>
        </div>
      </footer>
    </div>
  );
}

function ActivityCard({
  stage,
  label,
  status,
  logs,
}: {
  stage: PipelineStage;
  label: string;
  status: string;
  logs: { id: string; message: string; timestamp: number }[];
}) {
  const [expanded, setExpanded] = useState(false);

  const statusIcon = {
    pending: <div className="h-2 w-2 rounded-full bg-zinc-700" />,
    active: <div className="h-2 w-2 animate-pulse rounded-full bg-white" />,
    done: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" className="text-success">
        <polyline points="20 6 9 17 4 12" />
      </svg>
    ),
    error: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-danger">
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
      </svg>
    ),
  };

  const statusText = {
    pending: "Queued",
    active: "Running",
    done: "Completed",
    error: "Error",
  };

  return (
    <div className="overflow-hidden rounded-card border border-white/[0.04] bg-elevated">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between px-3 py-2.5 text-left transition-colors hover:bg-white/[0.02]"
      >
        <div className="flex items-center gap-2.5">
          {statusIcon[status as keyof typeof statusIcon]}
          <span className="text-[13px] font-medium text-zinc-200">{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-[11px] ${status === "active" ? "text-white" : status === "done" ? "text-success" : status === "error" ? "text-danger" : "text-zinc-600"}`}>
            {statusText[status as keyof typeof statusText]}
          </span>
          {logs.length > 0 && (
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className={`text-zinc-600 transition-transform ${expanded ? "rotate-90" : ""}`}
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
          )}
        </div>
      </button>
      {expanded && logs.length > 0 && (
        <div className="border-t border-white/[0.04] p-3">
          <div className="flex flex-col gap-1.5">
            {logs.map((log) => (
              <div key={log.id} className="flex flex-col gap-0.5">
                <span className="text-[11px] text-zinc-500">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className="text-[12px] text-zinc-400">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] font-medium text-zinc-500">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-20 accent-accent"
      />
      <span className="w-8 text-right text-[11px] font-mono text-zinc-400">{value.toFixed(2)}</span>
    </div>
  );
}

function InspectorField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[13px] text-zinc-500">{label}</span>
      <span className="text-[13px] font-mono text-zinc-300">{value}</span>
    </div>
  );
}

function ExportButton({ label }: { label: string }) {
  return (
    <button className="flex items-center justify-between rounded-btn border border-white/[0.06] bg-elevated px-3 py-2 text-[13px] font-medium text-zinc-200 transition-colors hover:bg-white/[0.06]">
      {label}
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-zinc-500">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
      </svg>
    </button>
  );
}
