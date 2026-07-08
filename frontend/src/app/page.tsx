"use client";

import { useCallback, useRef, useState } from "react";
import UploadDropzone from "@/components/UploadDropzone";
import DreamLog from "@/components/DreamLog";
import StageProgress from "@/components/StageProgress";
import SplatViewer, { type SplatViewerHandle } from "@/components/SplatViewer";
import Controls from "@/components/Controls";
import Gallery from "@/components/Gallery";
import { usePipeline } from "@/lib/usePipeline";
import { SparklesIcon, ArrowRightIcon, ArrowLeftIcon, CheckIcon, XIcon, DotIcon } from "@/components/Icons";

export default function HomePage() {
  const { state, start, reset, loadSplat } = usePipeline();
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [viewerMode, setViewerMode] = useState<"orbit" | "fps">("orbit");
  const splatViewerRef = useRef<SplatViewerHandle>(null);

  const handleImageSelected = useCallback((image: File) => {
    setSelectedImage(image);
  }, []);

  const handleBeginDream = useCallback(() => {
    if (selectedImage) {
      start(selectedImage);
    }
  }, [selectedImage, start]);

  const handleReset = useCallback(() => {
    reset();
    setSelectedImage(null);
  }, [reset]);

  const handleGallerySelect = useCallback((splatUrl: string) => {
    const fullUrl = splatUrl.startsWith("http")
      ? splatUrl
      : `${process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}${splatUrl}`;
    loadSplat(fullUrl);
  }, [loadSplat]);

  const showPipeline = state.isRunning || state.isDone || state.error;

  return (
    <main className="h-screen w-screen overflow-hidden bg-void relative z-10">
      {!showPipeline ? (
        // ── Landing / Upload View ──────────────────────────────────────────
        <div className="h-full flex flex-col items-center justify-center px-4 py-8 overflow-y-auto thin-scroll">
          <div className="mb-8 text-center flex flex-col items-center gap-4 shrink-0">
            <div className="relative">
              <SparklesIcon size={36} className="text-accent/50" />
              <div className="absolute inset-0 blur-xl bg-accent/10 rounded-full" />
            </div>
            <h1 className="text-4xl sm:text-5xl font-extralight tracking-[0.2em] text-gray-200">
              LUCID<span className="text-accent">FRAME</span>
            </h1>
            <p className="text-xs sm:text-sm text-gray-600 tracking-widest font-light">
              ONE IMAGE IN / A WORLD OUT
            </p>
          </div>

          <div className="w-full max-w-2xl shrink-0">
            <UploadDropzone
              onImageSelected={handleImageSelected}
              disabled={state.isRunning}
            />

            {selectedImage && (
              <div className="mt-6 flex justify-center animate-slide-up">
                <button
                  onClick={handleBeginDream}
                  disabled={state.isRunning}
                  className="flex items-center gap-2 px-8 py-3 bg-accent text-white font-light tracking-wider rounded-xl
                           hover:bg-accent/80 transition-all duration-300
                           disabled:opacity-40 disabled:pointer-events-none
                           shadow-lg shadow-accent/20"
                >
                  <span>Begin Dream</span>
                  <ArrowRightIcon size={16} />
                </button>
              </div>
            )}
          </div>

          {/* Gallery of pre-baked splats */}
          <div className="w-full max-w-2xl mt-8 shrink-0">
            <Gallery onSelect={handleGallerySelect} />
          </div>
        </div>
      ) : (
        // ── Pipeline / Viewer View ─────────────────────────────────────────
        <div className="h-full flex flex-col p-3 sm:p-4 gap-2 sm:gap-3">
          {/* Header bar */}
          <div className="flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2 sm:gap-3 min-w-0">
              <h1 className="text-base sm:text-lg font-extralight tracking-[0.2em] text-gray-300 shrink-0">
                LUCID<span className="text-accent">FRAME</span>
              </h1>
              {state.isDone && (
                <span className="flex items-center gap-1.5 text-success text-xs font-mono shrink-0">
                  <CheckIcon size={12} /> DREAM COMPLETE
                </span>
              )}
              {state.isRunning && (
                <span className="flex items-center gap-1.5 text-accent text-xs font-mono animate-subtle-pulse shrink-0">
                  <DotIcon size={12} /> DREAMING...
                </span>
              )}
              {state.error && (
                <span className="flex items-center gap-1.5 text-red-400 text-xs font-mono shrink-0">
                  <XIcon size={12} /> ERROR
                </span>
              )}
            </div>
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 px-2 sm:px-3 py-1.5 text-xs font-mono text-gray-500 hover:text-accent
                       border border-white/5 rounded-lg transition-all hover:border-accent/20 shrink-0"
            >
              <ArrowLeftIcon size={12} />
              <span className="hidden sm:inline">New Dream</span>
            </button>
          </div>

          {/* Stage progress */}
          <div className="shrink-0">
            <StageProgress stages={state.stages} />
          </div>

          {/* Error display */}
          {state.error && (
            <div className="p-3 border border-red-500/20 bg-red-950/10 rounded-xl flex items-center gap-2 shrink-0">
              <XIcon size={14} className="text-red-400 shrink-0" />
              <p className="text-red-400 text-sm font-mono">{state.error}</p>
            </div>
          )}

          {/* Main split: DreamLog + SplatViewer — stacked on mobile, side-by-side on desktop */}
          <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-2 sm:gap-3 min-h-0 overflow-hidden">
            {/* Left: DreamLog */}
            <div className="min-h-0 h-48 md:h-auto">
              <DreamLog
                logs={state.logs}
                vlmAnalysis={state.vlmAnalysis}
                dreamPrompt={state.dreamPrompt}
              />
            </div>

            {/* Right: Splat Viewer + Controls */}
            <div className="min-h-0 flex flex-col gap-2 h-64 md:h-auto">
              <SplatViewer
                ref={splatViewerRef}
                splatUrl={state.splatUrl}
                className="h-full w-full flex-1"
                mode={viewerMode}
                onModeChange={setViewerMode}
              />
              <Controls
                viewerRef={splatViewerRef}
                visible={!!state.splatUrl}
              />
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
