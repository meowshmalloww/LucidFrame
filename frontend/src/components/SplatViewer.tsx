"use client";

import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from "react";
import { SPLAT_URL_BASE } from "@/lib/api";

export interface SplatViewerHandle {
  resetCamera: () => void;
  setFov: (fov: number) => void;
  getCamera: () => unknown;
}

interface SplatViewerProps {
  splatUrl: string | null;
  className?: string;
  mode: "orbit" | "fps";
  onModeChange: (mode: "orbit" | "fps") => void;
}

const SplatViewer = forwardRef<SplatViewerHandle, SplatViewerProps>(
  ({ splatUrl, className, mode, onModeChange }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const engineRef = useRef<{
      scene: any;
      camera: any;
      renderer: any;
      controls: any;
      SPLAT: any;
      initialCameraPos: any;
      initialCameraRot: any;
    } | null>(null);
    const [showHint, setShowHint] = useState(true);
    const [isLoading, setIsLoading] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);

    // Main render loop + splat loading
    useEffect(() => {
      if (!splatUrl || !canvasRef.current) return;

      let cancelled = false;
      let animationId: number | null = null;
      let scene: any, camera: any, renderer: any, controls: any;

      (async () => {
        const SPLAT = await import("gsplat");

        if (cancelled) return;

        scene = new SPLAT.Scene();
        camera = new SPLAT.Camera();
        renderer = new SPLAT.WebGLRenderer(canvasRef.current);

        // Set up controls based on current mode
        if (mode === "orbit") {
          controls = new SPLAT.OrbitControls(camera, renderer.canvas);
        } else {
          controls = new SPLAT.FPSControls(camera, canvasRef.current!);
        }

        setIsLoading(true);
        setLoadProgress(0);

        const fullUrl = splatUrl.startsWith("http")
          ? splatUrl
          : `${SPLAT_URL_BASE}${splatUrl}`;
        await SPLAT.Loader.LoadAsync(fullUrl, scene, (progress: number) => {
          setLoadProgress(progress);
        });

        if (cancelled) {
          controls.dispose();
          renderer.dispose();
          return;
        }

        setIsLoading(false);

        // Store initial camera state for reset
        const initialPos = camera.position.clone();
        const initialRot = camera.rotation.clone();

        const frame = () => {
          controls.update();
          renderer.render(scene, camera);
          animationId = requestAnimationFrame(frame);
        };
        frame();

        engineRef.current = {
          scene,
          camera,
          renderer,
          controls,
          SPLAT,
          initialCameraPos: initialPos,
          initialCameraRot: initialRot,
        };
      })();

      return () => {
        cancelled = true;
        if (animationId) cancelAnimationFrame(animationId);
        if (controls) controls.dispose();
        if (renderer) renderer.dispose();
        engineRef.current = null;
      };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mode is handled in a separate effect to avoid reloading the splat
    }, [splatUrl]); // Only re-run when splatUrl changes, NOT mode

    // Handle mode switches without reloading the splat
    useEffect(() => {
      const engine = engineRef.current;
      if (!engine) return;

      // Dispose old controls
      engine.controls.dispose();

      // Create new controls
      if (mode === "orbit") {
        engine.controls = new engine.SPLAT.OrbitControls(engine.camera, engine.renderer.canvas);
      } else {
        engine.controls = new engine.SPLAT.FPSControls(engine.camera, canvasRef.current!);
      }

      setShowHint(true);
    }, [mode]);

    // Expose imperative API to parent Controls component
    useImperativeHandle(ref, () => ({
      resetCamera: () => {
        const engine = engineRef.current;
        if (!engine) return;
        engine.camera.position = engine.initialCameraPos.clone();
        engine.camera.rotation = engine.initialCameraRot.clone();
        // Reset orbit controls target to origin
        if (mode === "orbit" && engine.controls.setCameraTarget) {
          engine.controls.setCameraTarget(new engine.SPLAT.Vector3(0, 0, 0));
        }
      },
      setFov: (fov: number) => {
        const engine = engineRef.current;
        if (!engine) return;
        // Convert FOV (degrees) to focal length
        // fx = (width / 2) / tan(fov/2 in radians)
        const width = engine.camera.data.width || 800;
        const fx = (width / 2) / Math.tan((fov * Math.PI) / 360);
        engine.camera.data.fx = fx;
        engine.camera.data.fy = fx;
      },
      getCamera: () => engineRef.current?.camera,
    }));

    const handleCanvasClick = useCallback(() => {
      if (mode === "fps" && showHint) setShowHint(false);
    }, [mode, showHint]);

    if (!splatUrl) {
      return (
        <div className={`flex items-center justify-center rounded-card bg-void ${className}`}>
          <div className="flex flex-col items-center gap-3 text-center">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-zinc-800">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
            <p className="text-[14px] text-zinc-600">The 3D world will appear here</p>
          </div>
        </div>
      );
    }

    return (
      <div ref={containerRef} className={`relative overflow-hidden rounded-card bg-void ${className}`}>
        <canvas
          ref={canvasRef}
          className="block h-full w-full"
          onClick={handleCanvasClick}
        />

        {/* Loading overlay */}
        {isLoading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-void/90">
            <div className="flex flex-col items-center gap-4">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/[0.06] border-t-accent" />
              <div className="text-[14px] font-medium text-zinc-400">
                Loading 3D scene...
              </div>
              <div className="h-[3px] w-48 overflow-hidden rounded-full bg-white/[0.04]">
                <div
                  className="h-full bg-accent transition-all duration-300"
                  style={{ width: `${loadProgress * 100}%` }}
                />
              </div>
              <div className="text-[12px] font-mono text-zinc-600">
                {Math.round(loadProgress * 100)}%
              </div>
            </div>
          </div>
        )}

        {/* WASD hint */}
        {showHint && mode === "fps" && !isLoading && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
            <div className="flex flex-col items-center gap-2 rounded-card bg-surface/80 px-6 py-4 text-center backdrop-blur-sm animate-fade-in">
              <p className="text-[15px] font-medium text-white">Click to enter</p>
              <p className="text-[12px] text-zinc-500">WASD to move / Mouse to look / Space+Shift for up-down</p>
            </div>
          </div>
        )}

        {/* Orbit hint */}
        {showHint && mode === "orbit" && !isLoading && (
          <div className="absolute bottom-3 left-3 z-10 rounded-btn bg-surface/80 px-3 py-1.5 text-[12px] text-zinc-500 backdrop-blur-sm">
            Drag to orbit / Scroll to zoom
          </div>
        )}
      </div>
    );
  }
);

SplatViewer.displayName = "SplatViewer";
export default SplatViewer;
