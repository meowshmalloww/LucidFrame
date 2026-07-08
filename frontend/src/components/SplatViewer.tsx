"use client";

import { useEffect, useRef, useState } from "react";
import { SPLAT_URL_BASE } from "@/lib/api";

interface SplatViewerProps {
  splatUrl: string | null;
  className?: string;
}

export default function SplatViewer({ splatUrl, className }: SplatViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const splatRef = useRef<unknown>(null);
  const [mode, setMode] = useState<"orbit" | "fps">("orbit");
  const [showHint, setShowHint] = useState(true);

  useEffect(() => {
    if (!splatUrl || !canvasRef.current) return;

    let animationId: number;
    let scene: any, camera: any, renderer: any, controls: any;

    (async () => {
      const SPLAT = await import("gsplat");

      scene = new SPLAT.Scene();
      camera = new SPLAT.Camera();
      renderer = new SPLAT.WebGLRenderer(canvasRef.current!);

      if (mode === "orbit") {
        controls = new SPLAT.OrbitControls(camera, renderer.canvas);
      }

      const fullUrl = `${SPLAT_URL_BASE}${splatUrl}`;
      await SPLAT.Loader.LoadAsync(fullUrl, scene, (progress: number) => {
        console.log("Loading splat:", progress);
      });

      const frame = () => {
        if (controls) controls.update();
        renderer.render(scene, camera);
        animationId = requestAnimationFrame(frame);
      };
      frame();

      splatRef.current = { scene, camera, renderer, controls, SPLAT };
    })();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
    };
  }, [splatUrl, mode]);

  // WASD + pointer lock for FPS mode
  useEffect(() => {
    if (mode !== "fps" || !splatRef.current) return;

    const { camera } = splatRef.current as any;
    if (!camera) return;

    const canvas = canvasRef.current;
    const keys: Record<string, boolean> = {};
    let isLocked = false;

    const onKeyDown = (e: KeyboardEvent) => {
      keys[e.code] = true;
      if (e.code === "KeyW" && showHint) setShowHint(false);
    };
    const onKeyUp = (e: KeyboardEvent) => {
      keys[e.code] = false;
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isLocked || !camera) return;
      camera.rotation.y -= e.movementX * 0.002;
      camera.rotation.x -= e.movementY * 0.002;
    };

    const onPointerLockChange = () => {
      isLocked = document.pointerLockElement === canvasRef.current;
    };

    const onClick = () => {
      canvasRef.current?.requestPointerLock();
    };

    const update = () => {
      if (!camera || !isLocked) return;
      const speed = 0.05;
      const forward = new (window as any).Float32Array(3);
      const right = new (window as any).Float32Array(3);

      // Simple WASD movement
      if (keys["KeyW"]) camera.position[2] -= speed;
      if (keys["KeyS"]) camera.position[2] += speed;
      if (keys["KeyA"]) camera.position[0] -= speed;
      if (keys["KeyD"]) camera.position[0] += speed;
      if (keys["Space"]) camera.position[1] += speed;
      if (keys["ShiftLeft"]) camera.position[1] -= speed;
    };

    const interval = setInterval(update, 16);

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("mousemove", onMouseMove);
    document.addEventListener("pointerlockchange", onPointerLockChange);
    canvas?.addEventListener("click", onClick);

    return () => {
      clearInterval(interval);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("pointerlockchange", onPointerLockChange);
      canvas?.removeEventListener("click", onClick);
    };
  }, [mode, showHint]);

  if (!splatUrl) {
    return (
      <div className={`flex items-center justify-center bg-void/80 rounded-lg ${className}`}>
        <div className="text-center">
          <div className="text-4xl mb-3 opacity-30">🔮</div>
          <p className="text-gray-600 text-sm">The dream will appear here</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={`relative rounded-lg overflow-hidden bg-black ${className}`}>
      <canvas ref={canvasRef} className="w-full h-full block" />

      {/* Mode toggle */}
      <div className="absolute top-3 right-3 flex gap-1 z-10">
        <button
          onClick={() => setMode("orbit")}
          className={`px-3 py-1 text-xs font-mono rounded transition-all ${
            mode === "orbit"
              ? "bg-accent text-white"
              : "bg-panel/80 text-gray-400 hover:text-white"
          }`}
        >
          ORBIT
        </button>
        <button
          onClick={() => setMode("fps")}
          className={`px-3 py-1 text-xs font-mono rounded transition-all ${
            mode === "fps"
              ? "bg-accent text-white"
              : "bg-panel/80 text-gray-400 hover:text-white"
          }`}
        >
          WASD
        </button>
      </div>

      {/* WASD hint */}
      {showHint && mode === "fps" && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
          <div className="bg-black/60 px-6 py-4 rounded-lg text-center animate-fade-in">
            <p className="text-accent text-lg font-light mb-1">Press W to step through the frame</p>
            <p className="text-gray-500 text-xs">Click to enable mouse look · WASD to move · Space/Shift for up/down</p>
          </div>
        </div>
      )}

      {/* Orbit hint */}
      {showHint && mode === "orbit" && (
        <div className="absolute bottom-3 left-3 bg-black/60 px-3 py-1.5 rounded text-xs text-gray-400 z-10">
          Drag to orbit · Scroll to zoom
        </div>
      )}
    </div>
  );
}
