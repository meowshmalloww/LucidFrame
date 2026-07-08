"use client";

import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from "react";
import { SPLAT_URL_BASE } from "@/lib/api";
import { CubeIcon, OrbitIcon, FpsIcon, EyeIcon } from "./Icons";

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

        const fullUrl = `${SPLAT_URL_BASE}${splatUrl}`;
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
        <div className={`flex items-center justify-center glass rounded-xl ${className}`}>
          <div className="text-center flex flex-col items-center gap-3">
            <CubeIcon size={40} className="text-gray-700" />
            <p className="text-gray-600 text-sm">The dream will appear here</p>
          </div>
        </div>
      );
    }

    return (
      <div ref={containerRef} className={`relative rounded-xl overflow-hidden bg-black ${className}`}>
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
          onClick={handleCanvasClick}
        />

        {/* Loading overlay */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-20">
            <div className="text-center flex flex-col items-center gap-4">
              <CubeIcon size={32} className="text-accent animate-subtle-pulse" />
              <div className="text-sm text-accent font-light tracking-wider">
                Loading dream...
              </div>
              <div className="w-48 h-[3px] bg-white/5 rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent transition-all duration-300"
                  style={{ width: `${loadProgress * 100}%` }}
                />
              </div>
              <div className="text-xs text-gray-500 font-mono">
                {Math.round(loadProgress * 100)}%
              </div>
            </div>
          </div>
        )}

        {/* Mode toggle */}
        <div className="absolute top-3 right-3 flex gap-1 z-10">
          <button
            onClick={() => onModeChange("orbit")}
            className={`px-2.5 py-1.5 text-xs font-mono rounded-lg transition-all flex items-center gap-1.5 ${
              mode === "orbit"
                ? "bg-accent/20 text-accent border border-accent/30"
                : "bg-panel/60 text-gray-500 hover:text-white border border-white/5"
            }`}
          >
            <OrbitIcon size={12} />
            <span>ORBIT</span>
          </button>
          <button
            onClick={() => onModeChange("fps")}
            className={`px-2.5 py-1.5 text-xs font-mono rounded-lg transition-all flex items-center gap-1.5 ${
              mode === "fps"
                ? "bg-accent/20 text-accent border border-accent/30"
                : "bg-panel/60 text-gray-500 hover:text-white border border-white/5"
            }`}
          >
            <FpsIcon size={12} />
            <span>WASD</span>
          </button>
        </div>

        {/* WASD hint */}
        {showHint && mode === "fps" && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
            <div className="bg-black/60 px-6 py-4 rounded-xl text-center animate-fade-in flex flex-col items-center gap-2">
              <EyeIcon size={20} className="text-accent" />
              <p className="text-accent text-base font-light">Click to step into the dream</p>
              <p className="text-gray-500 text-xs">WASD to move / Mouse to look / Space+Shift for up-down</p>
            </div>
          </div>
        )}

        {/* Orbit hint */}
        {showHint && mode === "orbit" && !isLoading && (
          <div className="absolute bottom-3 left-3 bg-black/50 px-3 py-1.5 rounded-lg text-xs text-gray-500 z-10">
            Drag to orbit / Scroll to zoom
          </div>
        )}
      </div>
    );
  }
);

SplatViewer.displayName = "SplatViewer";
export default SplatViewer;
