"use client";

import { useCallback, useState } from "react";
import UploadDropzone from "@/components/UploadDropzone";
import DreamLog from "@/components/DreamLog";
import StageProgress from "@/components/StageProgress";
import SplatViewer from "@/components/SplatViewer";
import { usePipeline } from "@/lib/usePipeline";

export default function HomePage() {
  const { state, start, reset } = usePipeline();
  const [selectedImage, setSelectedImage] = useState<File | null>(null);

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

  const showPipeline = state.isRunning || state.isDone || state.error;

  return (
    <main className="h-screen w-screen overflow-hidden bg-void">
      {!showPipeline ? (
        // ── Landing / Upload View ──────────────────────────────────────────
        <div className="h-full flex flex-col items-center justify-center px-4">
          <div className="mb-8 text-center">
            <h1 className="text-5xl font-extralight tracking-widest text-gray-200 mb-2">
              LUCID<span className="text-accent">FRAME</span>
            </h1>
            <p className="text-sm text-gray-500 tracking-wide">
              One image in · A world out
            </p>
          </div>

          <div className="w-full max-w-2xl">
            <UploadDropzone
              onImageSelected={handleImageSelected}
              disabled={state.isRunning}
            />

            {selectedImage && (
              <div className="mt-6 flex justify-center">
                <button
                  onClick={handleBeginDream}
                  disabled={state.isRunning}
                  className="px-8 py-3 bg-accent text-white font-light tracking-wider rounded-lg
                           hover:bg-accent/80 transition-all duration-300 animate-glow
                           disabled:opacity-50 disabled:pointer-events-none"
                >
                  Begin Dream →
                </button>
              </div>
            )}
          </div>
        </div>
      ) : (
        // ── Pipeline / Viewer View ─────────────────────────────────────────
        <div className="h-full flex flex-col p-4 gap-4">
          {/* Header bar */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-extralight tracking-widest text-gray-300">
                LUCID<span className="text-accent">FRAME</span>
              </h1>
              {state.isDone && (
                <span className="text-success text-xs font-mono">● DREAM COMPLETE</span>
              )}
              {state.isRunning && (
                <span className="text-accent text-xs font-mono animate-pulse">● DREAMING...</span>
              )}
              {state.error && (
                <span className="text-red-400 text-xs font-mono">● ERROR</span>
              )}
            </div>
            <button
              onClick={handleReset}
              className="px-3 py-1 text-xs font-mono text-gray-500 hover:text-accent
                       border border-secondary/30 rounded transition-all"
            >
              ← New Dream
            </button>
          </div>

          {/* Stage progress */}
          <StageProgress stages={state.stages} />

          {/* Error display */}
          {state.error && (
            <div className="p-3 border border-red-500/30 bg-red-950/20 rounded-lg">
              <p className="text-red-400 text-sm font-mono">✗ {state.error}</p>
            </div>
          )}

          {/* Main split: DreamLog + SplatViewer */}
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0">
            {/* Left: DreamLog */}
            <div className="min-h-0">
              <DreamLog
                logs={state.logs}
                vlmAnalysis={state.vlmAnalysis}
                dreamPrompt={state.dreamPrompt}
              />
            </div>

            {/* Right: Splat Viewer */}
            <div className="min-h-0">
              <SplatViewer
                splatUrl={state.splatUrl}
                className="h-full w-full"
              />
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
