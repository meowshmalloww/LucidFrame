"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import SplatViewer, { type SplatRenderQuality, type SplatViewerHandle } from "@/components/SplatViewer";
import { GenerationProgress } from "@/components/GenerationProgress";
import { SPLAT_URL_BASE } from "@/lib/api";
import { usePipeline } from "@/lib/usePipeline";

type Provider = "local" | "local_world" | "local_pano" | "worldlabs";

export function ImmersiveWorkspace() {
  const router = useRouter();
  const { state, connect, loadSplat } = usePipeline();
  const viewerRef = useRef<SplatViewerHandle>(null);
  const [cameraMode, setCameraMode] = useState<"orbit" | "fps">("orbit");
  const [renderQuality, setRenderQuality] = useState<SplatRenderQuality>("standard");
  const [projectName, setProjectName] = useState("Untitled scene");
  const [sessionSplat, setSessionSplat] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [provider, setProvider] = useState<Provider>("local");

  useEffect(() => {
    const jobId = sessionStorage.getItem("lucidframe-job-id");
    const queryMode = new URLSearchParams(window.location.search).get("mode");
    const querySplat = new URLSearchParams(window.location.search).get("splat");
    const safeQuerySplat = querySplat?.startsWith("/outputs/") && querySplat.endsWith(".splat") ? querySplat : null;
    const storedSplat = safeQuerySplat || sessionStorage.getItem("lucidframe-splat-url");
    const storedName = sessionStorage.getItem("lucidframe-project-name") || sessionStorage.getItem("lucidframe-splat-name");
    const storedProvider = sessionStorage.getItem("lucidframe-provider");

    if (safeQuerySplat && queryMode === "panorama") {
      setProvider("local_pano");
    } else if (safeQuerySplat && queryMode === "image") {
      setProvider("local");
    } else if (storedProvider === "local" || storedProvider === "local_world" || storedProvider === "local_pano" || storedProvider === "worldlabs") {
      setProvider(storedProvider);
    }
    if (storedName) setProjectName(storedName);
    if (safeQuerySplat) setProjectName("Local scene preview");
    if (storedSplat) setSessionSplat(storedSplat);

    if (safeQuerySplat && !state.splatUrl && !state.isRunning) {
      loadSplat(safeQuerySplat);
    } else if (jobId && !state.splatUrl && !state.isRunning) {
      sessionStorage.removeItem("lucidframe-job-id");
      connect(jobId);
    } else if (storedSplat && !state.splatUrl && !state.isRunning) {
      loadSplat(storedSplat);
    }
    // Restoring once prevents duplicate WebSocket jobs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!state.splatUrl) return;
    setSessionSplat(state.splatUrl);
    sessionStorage.setItem("lucidframe-splat-url", state.splatUrl);
  }, [state.splatUrl]);

  const splatUrl = state.splatUrl || sessionSplat;
  const worldUrl = state.worldUrl;
  const isPanorama = provider === "local_pano" || provider === "local_world";
  const empty = !splatUrl && !state.isRunning && !state.error && !worldUrl;

  if (worldUrl) {
    return <WorldLabsHandoff name={projectName} url={worldUrl} onBack={() => router.push("/")} />;
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-[#e9e7df] text-[#1b1b18]">
      <header className="flex h-[66px] shrink-0 items-center justify-between border-b border-[#cbc9bf] bg-[#f7f6f0] px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-4">
          <button
            type="button"
            onClick={() => router.push("/")}
            className="flex items-center gap-2 rounded-[7px] border border-[#bbb9af] bg-white px-3 py-2 text-xs font-semibold hover:border-[#77766f]"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4 fill-none stroke-current stroke-2" aria-hidden><path d="m15 6-6 6 6 6" /></svg>
            <span className="hidden sm:inline">New scene</span>
          </button>
          <div className="min-w-0 border-l border-[#d1cfc5] pl-4">
            <h1 className="truncate text-sm font-semibold">{projectName}</h1>
            <p className="mt-0.5 text-xs text-[#77766f]">{methodLabel(provider)}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowDetails((value) => !value)}
          className={showDetails ? "rounded-[7px] border border-[#315c4a] bg-[#edf2ee] px-3 py-2 text-xs font-semibold text-[#274c3d]" : "rounded-[7px] border border-[#bbb9af] bg-white px-3 py-2 text-xs font-semibold hover:border-[#77766f]"}
        >
          Info
        </button>
      </header>

      <main className="relative min-h-0 flex-1 p-3 sm:p-4">
        <div className="h-full overflow-hidden rounded-[10px] border border-[#3c3c37] bg-[#20201d] shadow-[0_20px_55px_rgba(30,30,26,0.16)]">
          {empty ? (
            <EmptyWorld onNew={() => router.push("/")} />
          ) : (
            <SplatViewer
              ref={viewerRef}
              splatUrl={splatUrl}
              className="h-full w-full"
              mode={cameraMode}
              sceneMode={isPanorama ? "panorama" : "image"}
              renderQuality={renderQuality}
            />
          )}
        </div>

        {state.isRunning && !splatUrl && <GenerationProgress state={state} />}

        {state.error && (
          <div role="alert" className="absolute bottom-24 left-1/2 z-30 w-[min(36rem,calc(100%-3rem))] -translate-x-1/2 rounded-[9px] border border-[#b55b54] bg-[#f8e9e7] p-4 text-sm text-[#7e302b] shadow-xl">
            {state.error}
            <button type="button" onClick={() => router.push("/")} className="ml-3 font-semibold underline underline-offset-4">Return to Create</button>
          </div>
        )}

        {splatUrl && (
          <div className="absolute bottom-7 left-1/2 z-20 flex max-w-[calc(100%-3rem)] -translate-x-1/2 items-center gap-1 rounded-[9px] border border-white/15 bg-[#1b1b18]/90 p-1.5 text-white shadow-xl backdrop-blur">
            <button
              type="button"
              onClick={() => setCameraMode("orbit")}
              className={cameraMode === "orbit" ? "rounded-[6px] bg-white px-3 py-2 text-xs font-semibold text-[#1b1b18]" : "rounded-[6px] px-3 py-2 text-xs text-white/65 hover:text-white"}
            >
              Look
            </button>
            <button
              type="button"
              onClick={() => setCameraMode("fps")}
              className={cameraMode === "fps" ? "rounded-[6px] bg-white px-3 py-2 text-xs font-semibold text-[#1b1b18]" : "rounded-[6px] px-3 py-2 text-xs text-white/65 hover:text-white"}
            >
              Explore
            </button>
            <span className="mx-1 h-5 w-px bg-white/15" />
            <button type="button" onClick={() => viewerRef.current?.resetCamera()} className="rounded-[6px] px-3 py-2 text-xs text-white/65 hover:text-white">Reset view</button>
            <button
              type="button"
              aria-pressed={renderQuality === "high"}
              title="Supersample the viewport and gently widen only subpixel Gaussian footprints. The scene data is unchanged."
              onClick={() => setRenderQuality((current) => current === "high" ? "standard" : "high")}
              className={renderQuality === "high" ? "rounded-[6px] bg-white px-3 py-2 text-xs font-semibold text-[#1b1b18]" : "rounded-[6px] px-3 py-2 text-xs text-white/65 hover:text-white"}
            >
              HD render
            </button>
            <a href={downloadUrl(splatUrl)} download className="hidden rounded-[6px] px-3 py-2 text-xs text-white/65 hover:text-white sm:block">Save .splat</a>
          </div>
        )}

        {showDetails && (
          <aside className="absolute right-7 top-7 z-30 w-[min(340px,calc(100%-3.5rem))] rounded-[10px] border border-[#cbc9bf] bg-[#faf9f5] p-5 shadow-2xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold">Scene info</h2>
              </div>
              <button type="button" onClick={() => setShowDetails(false)} className="rounded-[6px] border border-[#c6c3b9] p-1.5" aria-label="Close details">
                <svg viewBox="0 0 24 24" className="h-4 w-4 stroke-current stroke-2" aria-hidden><path d="m6 6 12 12M18 6 6 18" /></svg>
              </button>
            </div>
            <dl className="mt-5 divide-y divide-[#dedcd3] border-y border-[#dedcd3] text-xs">
              <Row label="Method" value={methodLabel(provider)} />
              <Row label="Splats" value={state.gaussianCount?.toLocaleString() || "Loaded from file"} />
              <Row label="View" value={isPanorama ? "360°" : "Source image"} />
              <Row label="Movement" value="Unrestricted" />
              <Row label="Rendering" value={renderQuality === "high" ? "HD" : "Standard"} />
            </dl>
          </aside>
        )}
      </main>
    </div>
  );
}

function methodLabel(provider: Provider) {
  if (provider === "local_world") return "Image to 360";
  if (provider === "local_pano") return "Panorama to 360";
  if (provider === "worldlabs") return "World Labs Marble";
  return "Image to 3D";
}

function downloadUrl(url: string) {
  return url.startsWith("http") ? url : SPLAT_URL_BASE + url;
}

function Row({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-4 py-2.5"><dt className="text-[#77766f]">{label}</dt><dd className="text-right font-medium">{value}</dd></div>;
}

function EmptyWorld({ onNew }: { onNew: () => void }) {
  return (
    <div className="grid h-full place-items-center p-8 text-center text-white">
      <div>
        <svg viewBox="0 0 48 48" className="mx-auto h-12 w-12 fill-none stroke-white/45 stroke-[1.2]" aria-hidden><path d="M24 5 7 14l17 9 17-9-17-9Z" /><path d="m7 23 17 9 17-9M7 32l17 9 17-9" /></svg>
        <h1 className="mt-4 text-xl font-semibold">No scene is open</h1>
        <p className="mt-2 text-sm text-white/55">Choose an image or open a generated scene from the Library.</p>
        <button type="button" onClick={onNew} className="mt-5 rounded-[8px] bg-white px-4 py-3 text-sm font-semibold text-[#1b1b18]">Choose an image</button>
      </div>
    </div>
  );
}

function WorldLabsHandoff({ name, url, onBack }: { name: string; url: string; onBack: () => void }) {
  return (
    <div className="grid h-full place-items-center bg-[#f3f2ec] p-6 text-center">
      <div className="w-full max-w-xl rounded-[12px] border border-[#d1cfc5] bg-[#faf9f5] p-8">
        <h1 className="text-3xl font-semibold tracking-[-0.04em]">{name}</h1>
        <p className="mt-4 text-sm leading-6 text-[#686760]">Your World Labs scene is ready in its hosted viewer.</p>
        <a href={url} target="_blank" rel="noreferrer" className="mt-6 inline-flex rounded-[8px] bg-[#1b1b18] px-5 py-3 text-sm font-semibold text-white hover:bg-[#315c4a]">Open hosted world</a>
        <button type="button" onClick={onBack} className="mt-4 block w-full text-xs font-semibold text-[#686760] underline underline-offset-4">Return to Create</button>
      </div>
    </div>
  );
}
