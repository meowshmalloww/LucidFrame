"use client";

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
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
  sceneMode?: "image" | "panorama";
  onModeChange?: (mode: "orbit" | "fps") => void;
}

type Control = {
  update: () => void;
  dispose: () => void;
  setCameraTarget?: (target: unknown) => void;
  moveSpeed?: number;
  lookSpeed?: number;
};

const SplatViewer = forwardRef<SplatViewerHandle, SplatViewerProps>(
  ({ splatUrl, className = "", mode, sceneMode = "image" }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const engineRef = useRef<{
      scene: any;
      camera: any;
      renderer: any;
      controls: Control;
      SPLAT: any;
      initialCameraPos: any;
      initialCameraRot: any;
      sceneMode: "image" | "panorama";
    } | null>(null);
    const modeRef = useRef(mode);
    modeRef.current = mode;
    const [showHint, setShowHint] = useState(true);
    const [isLoading, setIsLoading] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);
    const [loadError, setLoadError] = useState<string | null>(null);

    const makeControls = useCallback((engine: NonNullable<typeof engineRef.current>, nextMode: "orbit" | "fps") => {
      if (nextMode === "fps") {
        return createBoundedMoveControls(
          engine.camera,
          engine.renderer.canvas,
          engine.SPLAT,
          engine.sceneMode,
        );
      }
      return createLookControls(
        engine.camera,
        engine.renderer.canvas,
        engine.SPLAT,
        engine.sceneMode,
      );
    }, []);

    useEffect(() => {
      if (!splatUrl || !canvasRef.current) return;

      let cancelled = false;
      let animationId: number | null = null;
      let renderer: any;
      let controls: Control | null = null;

      setLoadError(null);
      setIsLoading(true);
      setLoadProgress(0);

      (async () => {
        try {
          const SPLAT = await import("gsplat");
          if (cancelled || !canvasRef.current) return;

          const scene = new SPLAT.Scene();
          const camera = new SPLAT.Camera();
          renderer = new SPLAT.WebGLRenderer(canvasRef.current);
          const fullUrl = splatUrl.startsWith("http") ? splatUrl : SPLAT_URL_BASE + splatUrl;

          await SPLAT.Loader.LoadAsync(fullUrl, scene, (progress: number) => {
            if (!cancelled) setLoadProgress(progress);
          });
          if (cancelled) {
            renderer.dispose();
            return;
          }

          camera.position = new SPLAT.Vector3(0, 0, 0);
          camera.rotation = SPLAT.Quaternion.FromEuler(new SPLAT.Vector3(0, 0, 0));

          const engine = {
            scene,
            camera,
            renderer,
            controls: { update: () => {}, dispose: () => {} } as Control,
            SPLAT,
            initialCameraPos: camera.position.clone(),
            initialCameraRot: camera.rotation.clone(),
            sceneMode,
          };
          controls = makeControls(engine, modeRef.current);
          engine.controls = controls;
          engineRef.current = engine;
          setIsLoading(false);
          setShowHint(true);

          const frame = () => {
            const current = engineRef.current;
            if (!current) return;
            current.controls.update();
            clampTranslation(
              current.camera,
              current.SPLAT,
              current.sceneMode === "panorama" ? 0.48 : 0.35,
            );
            current.renderer.resize();
            // SHARP estimates a perspective camera close to a 60–65° horizontal
            // field of view for ordinary photos. Matching that view fills the
            // frame without exposing unsupported border Gaussians.
            setCameraFov(current.camera, current.sceneMode === "panorama" ? 78 : 62);
            current.renderer.render(current.scene, current.camera);
            animationId = requestAnimationFrame(frame);
          };
          frame();
        } catch (caught) {
          if (!cancelled) {
            setIsLoading(false);
            setLoadError(caught instanceof Error ? caught.message : "The .splat scene could not be opened.");
          }
        }
      })();

      return () => {
        cancelled = true;
        if (animationId !== null) cancelAnimationFrame(animationId);
        controls?.dispose();
        renderer?.dispose();
        engineRef.current = null;
      };
    }, [makeControls, sceneMode, splatUrl]);

    useEffect(() => {
      const engine = engineRef.current;
      if (!engine || isLoading) return;
      engine.controls.dispose();
      engine.controls = makeControls(engine, mode);
      setShowHint(true);
    }, [isLoading, makeControls, mode]);

    useImperativeHandle(ref, () => ({
      resetCamera: () => {
        const engine = engineRef.current;
        if (!engine) return;
        engine.controls.dispose();
        engine.camera.position = engine.initialCameraPos.clone();
        engine.camera.rotation = engine.initialCameraRot.clone();
        engine.controls = makeControls(engine, mode);
        setShowHint(true);
      },
      setFov: (fov: number) => {
        const engine = engineRef.current;
        if (!engine) return;
        const width = engine.camera.data.width || 800;
        const focal = (width / 2) / Math.tan((fov * Math.PI) / 360);
        engine.camera.data.fx = focal;
        engine.camera.data.fy = focal;
      },
      getCamera: () => engineRef.current?.camera,
    }), [makeControls, mode]);

    if (!splatUrl) {
      return (
        <div className={"grid place-items-center bg-[#20201d] " + className}>
          <div className="text-center text-[#aaa89f]">
            <svg viewBox="0 0 48 48" className="mx-auto h-11 w-11 fill-none stroke-current stroke-[1.2]" aria-hidden><path d="M24 5 7 14l17 9 17-9-17-9Z" /><path d="m7 23 17 9 17-9M7 32l17 9 17-9" /></svg>
            <p className="mt-3 text-sm">The spatial scene will appear here.</p>
          </div>
        </div>
      );
    }

    const hint = sceneMode === "panorama"
      ? mode === "fps"
        ? "Drag to look. Use WASD to move within the captured volume."
        : "Drag to look in every direction."
      : mode === "fps"
        ? "Drag within the reliable view cone. Use WASD for a small metric move."
        : "Drag within the nearby-view cone predicted from the photograph.";

    return (
      <div className={"relative overflow-hidden bg-[#1f1f1c] " + className}>
        <canvas
          ref={canvasRef}
          className="block h-full w-full"
          onPointerDown={() => {
            if (mode === "fps") setShowHint(false);
          }}
        />

        {isLoading && (
          <div className="absolute inset-0 z-20 grid place-items-center bg-[#1f1f1c] text-white">
            <div className="w-64">
              <div className="flex items-center justify-between text-xs">
                <span>Loading Gaussian scene</span>
                <span>{Math.round(loadProgress * 100)}%</span>
              </div>
              <div className="mt-3 h-1 overflow-hidden bg-white/15">
                <div className="h-full bg-white transition-[width]" style={{ width: String(loadProgress * 100) + "%" }} />
              </div>
            </div>
          </div>
        )}

        {loadError && (
          <div className="absolute inset-0 z-20 grid place-items-center bg-[#1f1f1c] p-8 text-center text-white">
            <div>
              <p className="text-sm font-semibold">Scene loading failed</p>
              <p className="mt-2 max-w-md text-xs leading-5 text-white/60">{loadError}</p>
            </div>
          </div>
        )}

        {showHint && !isLoading && !loadError && (
          <button
            type="button"
            onClick={() => setShowHint(false)}
            className="absolute bottom-4 left-4 z-10 rounded-[7px] border border-white/15 bg-[#1b1b18]/85 px-3 py-2 text-left text-xs text-white/75 backdrop-blur"
          >
            {hint}
          </button>
        )}
      </div>
    );
  },
);

function createLookControls(
  camera: any,
  canvas: HTMLCanvasElement,
  SPLAT: any,
  sceneMode: "image" | "panorama",
): Control {
  const initial = camera.rotation.toEuler();
  let pitch = initial.x;
  let yaw = initial.y;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const down = (event: PointerEvent) => {
    if (event.button !== 0) return;
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
    canvas.style.cursor = "grabbing";
  };
  const move = (event: PointerEvent) => {
    if (!dragging) return;
    yaw -= (event.clientX - lastX) * 0.004;
    pitch -= (event.clientY - lastY) * 0.004;
    if (sceneMode === "image") {
      yaw = Math.max(-0.16, Math.min(0.16, yaw));
      pitch = Math.max(-0.12, Math.min(0.12, pitch));
    } else {
      pitch = Math.max(-1.48, Math.min(1.48, pitch));
    }
    lastX = event.clientX;
    lastY = event.clientY;
    camera.rotation = SPLAT.Quaternion.FromEuler(new SPLAT.Vector3(pitch, yaw, 0));
  };
  const up = (event: PointerEvent) => {
    dragging = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    canvas.style.cursor = "grab";
  };

  canvas.style.cursor = "grab";
  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);

  return {
    update: () => {},
    dispose: () => {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      canvas.style.cursor = "";
    },
  };
}

function createBoundedMoveControls(
  camera: any,
  canvas: HTMLCanvasElement,
  SPLAT: any,
  sceneMode: "image" | "panorama",
): Control {
  const initial = camera.rotation.toEuler();
  let pitch = initial.x;
  let yaw = initial.y;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let lastFrame = performance.now();
  const pressed = new Set<string>();

  const updateRotation = () => {
    camera.rotation = SPLAT.Quaternion.FromEuler(new SPLAT.Vector3(pitch, yaw, 0));
  };
  const down = (event: PointerEvent) => {
    if (event.button !== 0) return;
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
    canvas.style.cursor = "grabbing";
  };
  const move = (event: PointerEvent) => {
    if (!dragging) return;
    yaw -= (event.clientX - lastX) * 0.004;
    pitch -= (event.clientY - lastY) * 0.004;
    if (sceneMode === "image") {
      yaw = Math.max(-0.16, Math.min(0.16, yaw));
      pitch = Math.max(-0.12, Math.min(0.12, pitch));
    } else {
      pitch = Math.max(-1.48, Math.min(1.48, pitch));
    }
    lastX = event.clientX;
    lastY = event.clientY;
    updateRotation();
  };
  const up = (event: PointerEvent) => {
    dragging = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    canvas.style.cursor = "grab";
  };
  const keyDown = (event: KeyboardEvent) => {
    const target = event.target as HTMLElement | null;
    if (target?.matches("input, textarea, select, [contenteditable='true']")) return;
    if (["KeyW", "KeyA", "KeyS", "KeyD"].includes(event.code)) {
      event.preventDefault();
      pressed.add(event.code);
    }
  };
  const keyUp = (event: KeyboardEvent) => pressed.delete(event.code);
  const blur = () => pressed.clear();

  canvas.style.cursor = "grab";
  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  window.addEventListener("keydown", keyDown);
  window.addEventListener("keyup", keyUp);
  window.addEventListener("blur", blur);

  return {
    update: () => {
      const now = performance.now();
      const delta = Math.min((now - lastFrame) / 1000, 0.05);
      lastFrame = now;
      const forward = Number(pressed.has("KeyW")) - Number(pressed.has("KeyS"));
      const right = Number(pressed.has("KeyD")) - Number(pressed.has("KeyA"));
      if (forward === 0 && right === 0) return;
      const length = Math.hypot(forward, right) || 1;
      const distance = delta * 0.55;
      const forwardX = Math.sin(yaw);
      const forwardZ = Math.cos(yaw);
      const rightX = Math.cos(yaw);
      const rightZ = -Math.sin(yaw);
      const position = camera.position;
      camera.position = new SPLAT.Vector3(
        position.x + ((forward * forwardX + right * rightX) / length) * distance,
        position.y,
        position.z + ((forward * forwardZ + right * rightZ) / length) * distance,
      );
    },
    dispose: () => {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", blur);
      canvas.style.cursor = "";
      pressed.clear();

    },
  };
}
function clampTranslation(camera: any, SPLAT: any, radius: number) {
  const position = camera.position;
  const distance = position.magnitude();
  if (distance <= radius) return;
  camera.position = new SPLAT.Vector3(
    (position.x / distance) * radius,
    (position.y / distance) * radius,
    (position.z / distance) * radius,
  );
}

function setCameraFov(camera: any, fov: number) {
  const width = camera.data.width || 800;
  const focal = (width / 2) / Math.tan((fov * Math.PI) / 360);
  camera.data.fx = focal;
  camera.data.fy = focal;
}

SplatViewer.displayName = "SplatViewer";
export default SplatViewer;
