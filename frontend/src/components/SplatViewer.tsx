"use client";

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type {
  PerspectiveCamera,
  Quaternion,
  Scene,
  Vector3,
  WebGLRenderer,
} from "three";
import type { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { SPLAT_URL_BASE, type SceneCameraMetadata } from "@/lib/api";

export interface SplatViewerHandle {
  resetCamera: () => void;
  setFov: (fov: number) => void;
  getCamera: () => unknown;
}

export type SplatRenderQuality = "standard" | "high";

interface SplatViewerProps {
  splatUrl: string | null;
  className?: string;
  mode: "orbit" | "fps";
  sceneMode?: "image" | "panorama";
  renderQuality?: SplatRenderQuality;
  cameraCalibration?: SceneCameraMetadata | null;
}

type ThreeModule = typeof import("three");

type Control = {
  update: () => void;
  dispose: () => void;
};

type SparkEngine = {
  scene: Scene;
  camera: PerspectiveCamera;
  renderer: WebGLRenderer;
  spark: SparkRenderer;
  splat: SplatMesh;
  controls: Control;
  THREE: ThreeModule;
  initialCameraPos: Vector3;
  initialCameraRot: Quaternion;
  sceneMode: "image" | "panorama";
  cameraCalibration: SceneCameraMetadata | null;
  disposing: boolean;
};

const SplatViewer = forwardRef<SplatViewerHandle, SplatViewerProps>(
  ({ splatUrl, className = "", mode, sceneMode = "image", renderQuality = "standard", cameraCalibration = null }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const engineRef = useRef<SparkEngine | null>(null);
    const modeRef = useRef(mode);
    modeRef.current = mode;
    const renderQualityRef = useRef(renderQuality);
    renderQualityRef.current = renderQuality;
    const cameraCalibrationRef = useRef(cameraCalibration);
    cameraCalibrationRef.current = cameraCalibration;
    const [showHint, setShowHint] = useState(true);
    const [isLoading, setIsLoading] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);
    const [loadError, setLoadError] = useState<string | null>(null);

    const makeControls = useCallback((engine: SparkEngine, nextMode: "orbit" | "fps") => {
      if (nextMode === "fps") {
        return createFreeFlyControls(
          engine.camera,
          engine.renderer.domElement,
          engine.THREE,
          engine.cameraCalibration?.move_speed_mps,
        );
      }
      return createLookControls(
        engine.camera,
        engine.renderer.domElement,
        engine.THREE,
        engine.sceneMode,
      );
    }, []);

    useEffect(() => {
      if (!splatUrl || !canvasRef.current) return;

      let cancelled = false;
      let animationId: number | null = null;
      let engine: SparkEngine | null = null;
      let pendingRenderer: WebGLRenderer | null = null;
      let pendingSpark: SparkRenderer | null = null;
      let pendingSplat: SplatMesh | null = null;

      setLoadError(null);
      setIsLoading(true);
      setLoadProgress(0);

      (async () => {
        try {
          const [THREE, sparkModule] = await Promise.all([
            import("three"),
            import("@sparkjsdev/spark"),
          ]);
          if (cancelled || !canvasRef.current) return;

          const scene = new THREE.Scene();
          const camera = new THREE.PerspectiveCamera(60, 1, 0.01, sceneMode === "image" ? 1000 : 250);
          // SHARP exports OpenCV camera coordinates: +x right, +y down, +z
          // forward. Looking along +z with a -y up vector preserves the source
          // photograph without mirroring or rotating the learned Gaussians.
          camera.up.set(0, -1, 0);
          camera.position.set(0, 0, 0);
          pointCamera(camera, THREE, 0, 0);

          const renderer = new THREE.WebGLRenderer({
            canvas: canvasRef.current,
            antialias: false,
            alpha: false,
            premultipliedAlpha: true,
            powerPreference: "high-performance",
          });
          pendingRenderer = renderer;
          renderer.setClearColor(0x1f1f1c, 1);
          renderer.outputColorSpace = THREE.SRGBColorSpace;

          const spark = new sparkModule.SparkRenderer({
            renderer,
            // Radial ordering is more stable while the viewer rotates. A small
            // pre-filter follows Spark's recommendation for splats learned
            // without an anti-aliasing covariance term.
            sortRadial: true,
            preBlurAmount: 0.3,
            maxStdDev: gaussianExtent(renderQualityRef.current),
            minAlpha: 0.5 / 255,
            // A panoramic scene can contain several million splats. Keep every
            // splat in the draw, but avoid immediately starting another full
            // worker sort after the previous one finishes. Radial ordering
            // remains stable between sorts while the render loop stays live.
            minSortIntervalMs: sceneMode === "panorama" ? 80 : 20,
            enableLod: false,
          });
          pendingSpark = spark;
          scene.add(spark);

          const fullUrl = splatUrl.startsWith("http") ? splatUrl : SPLAT_URL_BASE + splatUrl;
          const splat = new sparkModule.SplatMesh({
            url: fullUrl,
            nonLod: true,
            onProgress: (event: ProgressEvent) => {
              if (cancelled) return;
              if (event.lengthComputable && event.total > 0) {
                setLoadProgress(Math.min(event.loaded / event.total, 0.98));
              }
            },
          });
          pendingSplat = splat;
          scene.add(splat);
          await splat.initialized;

          if (cancelled) {
            splat.dispose();
            spark.dispose();
            renderer.dispose();
            pendingSplat = null;
            pendingSpark = null;
            pendingRenderer = null;
            return;
          }

          const nextEngine: SparkEngine = {
            scene,
            camera,
            renderer,
            spark,
            splat,
            controls: { update: () => {}, dispose: () => {} },
            THREE,
            initialCameraPos: camera.position.clone(),
            initialCameraRot: camera.quaternion.clone(),
            sceneMode,
            cameraCalibration: cameraCalibrationRef.current,
            disposing: false,
          };
          nextEngine.controls = makeControls(nextEngine, modeRef.current);
          engine = nextEngine;
          pendingSplat = null;
          pendingSpark = null;
          pendingRenderer = null;
          engineRef.current = nextEngine;
          updateCanvasDiagnostics(canvasRef.current, renderQualityRef.current, sceneMode, cameraCalibrationRef.current);
          setLoadProgress(1);
          setIsLoading(false);
          setShowHint(true);

          const frame = () => {
            const current = engineRef.current;
            if (!current) return;
            current.controls.update();
            resizeRendererForDisplay(current.renderer, current.camera, renderQualityRef.current);
            current.spark.maxStdDev = gaussianExtent(renderQualityRef.current);
            setCameraHorizontalFov(
              current.camera,
              current.cameraCalibration?.horizontal_fov_deg || (current.sceneMode === "panorama" ? 78 : 62),
            );
            current.renderer.render(current.scene, current.camera);
            animationId = requestAnimationFrame(frame);
          };
          frame();
        } catch (caught) {
          pendingSplat?.dispose();
          pendingSpark?.dispose();
          pendingRenderer?.dispose();
          pendingSplat = null;
          pendingSpark = null;
          pendingRenderer = null;
          if (!cancelled) {
            setIsLoading(false);
            setLoadError(caught instanceof Error ? caught.message : "The .splat scene could not be opened.");
          }
        }
      })();

      return () => {
        cancelled = true;
        if (animationId !== null) cancelAnimationFrame(animationId);
        const current = engine || engineRef.current;
        engineRef.current = null;
        if (current) void disposeSparkEngineWhenIdle(current);
      };
    }, [makeControls, sceneMode, splatUrl]);

    useEffect(() => {
      const engine = engineRef.current;
      if (!engine || isLoading) return;
      engine.controls.dispose();
      engine.controls = makeControls(engine, mode);
      setShowHint(true);
    }, [isLoading, makeControls, mode]);

    useEffect(() => {
      if (canvasRef.current) {
        updateCanvasDiagnostics(canvasRef.current, renderQuality, sceneMode, cameraCalibration);
      }
      const engine = engineRef.current;
      if (engine) engine.spark.maxStdDev = gaussianExtent(renderQuality);
    }, [cameraCalibration, renderQuality, sceneMode]);

    useEffect(() => {
      const engine = engineRef.current;
      if (!engine) return;
      engine.cameraCalibration = cameraCalibration;
      if (modeRef.current === "fps") {
        engine.controls.dispose();
        engine.controls = makeControls(engine, "fps");
      }
    }, [cameraCalibration, makeControls]);

    useImperativeHandle(ref, () => ({
      resetCamera: () => {
        const engine = engineRef.current;
        if (!engine) return;
        engine.controls.dispose();
        engine.camera.position.copy(engine.initialCameraPos);
        engine.camera.quaternion.copy(engine.initialCameraRot);
        engine.controls = makeControls(engine, mode);
        setShowHint(true);
      },
      setFov: (fov: number) => {
        const engine = engineRef.current;
        if (engine) setCameraHorizontalFov(engine.camera, fov);
      },
      getCamera: () => engineRef.current?.camera,
    }), [makeControls, mode]);

    if (!splatUrl) {
      return (
        <div className={"grid place-items-center bg-[#20201d] " + className}>
          <div className="text-center text-[#aaa89f]">
            <svg viewBox="0 0 48 48" className="mx-auto h-11 w-11 fill-none stroke-current stroke-[1.2]" aria-hidden><path d="M24 5 7 14l17 9 17-9-17-9Z" /><path d="m7 23 17 9 17-9M7 32l17 9 17-9" /></svg>
            <p className="mt-3 text-sm">Your scene will appear here.</p>
          </div>
        </div>
      );
    }

    const hint = mode === "fps"
      ? "Drag to look. Use WASD to move, Q/E for height, and Shift to move faster. Short movements preserve the clearest view."
      : "Drag to look around.";

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
                <span>Preparing Gaussian scene</span>
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
  camera: PerspectiveCamera,
  canvas: HTMLCanvasElement,
  THREE: ThreeModule,
  sceneMode: "image" | "panorama",
): Control {
  const angles = cameraAngles(camera, THREE);
  let pitch = angles.pitch;
  let yaw = angles.yaw;
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
    pointCamera(camera, THREE, pitch, yaw);
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

function createFreeFlyControls(
  camera: PerspectiveCamera,
  canvas: HTMLCanvasElement,
  THREE: ThreeModule,
  calibratedMoveSpeed = 0.12,
): Control {
  const angles = cameraAngles(camera, THREE);
  let pitch = angles.pitch;
  let yaw = angles.yaw;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let lastFrame = performance.now();
  const pressed = new Set<string>();

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
    pitch = Math.max(-1.48, Math.min(1.48, pitch));
    lastX = event.clientX;
    lastY = event.clientY;
    pointCamera(camera, THREE, pitch, yaw);
  };
  const up = (event: PointerEvent) => {
    dragging = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    canvas.style.cursor = "grab";
  };
  const keyDown = (event: KeyboardEvent) => {
    const target = event.target as HTMLElement | null;
    if (target?.matches("input, textarea, select, [contenteditable='true']")) return;
    if (["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", "ShiftLeft", "ShiftRight"].includes(event.code)) {
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
      const vertical = Number(pressed.has("KeyE")) - Number(pressed.has("KeyQ"));
      if (forward === 0 && right === 0 && vertical === 0) return;
      const length = Math.hypot(forward, right, vertical) || 1;
      const sprinting = pressed.has("ShiftLeft") || pressed.has("ShiftRight");
      const distance = delta * calibratedMoveSpeed * 1.45 * (sprinting ? 2 : 1);
      const forwardX = Math.sin(yaw);
      const forwardZ = Math.cos(yaw);
      const rightX = Math.cos(yaw);
      const rightZ = -Math.sin(yaw);
      camera.position.x += ((forward * forwardX + right * rightX) / length) * distance;
      camera.position.y -= (vertical / length) * distance;
      camera.position.z += ((forward * forwardZ + right * rightZ) / length) * distance;
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

function cameraAngles(camera: PerspectiveCamera, THREE: ThreeModule) {
  const direction = camera.getWorldDirection(new THREE.Vector3());
  return {
    pitch: Math.asin(Math.max(-1, Math.min(1, direction.y))),
    yaw: Math.atan2(direction.x, direction.z),
  };
}

function pointCamera(camera: PerspectiveCamera, THREE: ThreeModule, pitch: number, yaw: number) {
  const cosPitch = Math.cos(pitch);
  const target = new THREE.Vector3(
    camera.position.x + Math.sin(yaw) * cosPitch,
    camera.position.y + Math.sin(pitch),
    camera.position.z + Math.cos(yaw) * cosPitch,
  );
  camera.lookAt(target);
}

function setCameraHorizontalFov(camera: PerspectiveCamera, horizontalFov: number) {
  const aspect = Math.max(camera.aspect || 1, 1e-6);
  const horizontalRadians = (horizontalFov * Math.PI) / 180;
  camera.fov = (2 * Math.atan(Math.tan(horizontalRadians / 2) / aspect) * 180) / Math.PI;
  camera.updateProjectionMatrix();
}

function resizeRendererForDisplay(renderer: WebGLRenderer, camera: PerspectiveCamera, quality: SplatRenderQuality) {
  const canvas = renderer.domElement as HTMLCanvasElement;
  const pixelRatio = quality === "high"
    ? Math.min((window.devicePixelRatio || 1) * 1.18, 2)
    : 1;
  const width = Math.max(1, Math.round(canvas.clientWidth));
  const height = Math.max(1, Math.round(canvas.clientHeight));
  const drawingWidth = Math.max(1, Math.round(width * pixelRatio));
  const drawingHeight = Math.max(1, Math.round(height * pixelRatio));
  if (
    renderer.getPixelRatio() !== pixelRatio ||
    canvas.width !== drawingWidth ||
    canvas.height !== drawingHeight
  ) {
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height, false);
  }
  const aspect = width / height;
  if (camera.aspect !== aspect) camera.aspect = aspect;
}

function gaussianExtent(quality: SplatRenderQuality) {
  return quality === "high" ? 3 : Math.sqrt(8);
}

function updateCanvasDiagnostics(
  canvas: HTMLCanvasElement,
  quality: SplatRenderQuality,
  sceneMode: "image" | "panorama",
  calibration: SceneCameraMetadata | null,
) {
  canvas.dataset.renderQuality = quality;
  canvas.dataset.adaptiveRenderer = "spark-2.1";
  canvas.dataset.cameraFov = String(calibration?.horizontal_fov_deg || (sceneMode === "panorama" ? 78 : 62));
  canvas.dataset.moveSpeed = String((calibration?.move_speed_mps || 0.12) * 1.45);
}

async function disposeSparkEngineWhenIdle(engine: SparkEngine) {
  if (engine.disposing) return;
  engine.disposing = true;

  const { spark } = engine;
  engine.controls.dispose();
  spark.autoUpdate = false;
  spark.sortDirty = false;
  spark.lodDirty = false;

  if (spark.updateTimeoutId !== -1) {
    window.clearTimeout(spark.updateTimeoutId);
    spark.updateTimeoutId = -1;
  }
  if (spark.sortTimeoutId !== -1) {
    window.clearTimeout(spark.sortTimeoutId);
    spark.sortTimeoutId = -1;
  }

  // Spark 2.1 rejects an in-flight worker request if dispose() terminates the
  // sorter. It can also dispose the GPU readback target while driveSort() is
  // still using it. Detach the scene now, then release its resources only once
  // that final sort has naturally completed.
  engine.scene.remove(engine.splat);
  engine.scene.remove(spark);
  while (spark.sorting) {
    await new Promise<void>((resolve) => window.setTimeout(resolve, 16));
  }

  spark.dispose();
  engine.splat.dispose();
  engine.renderer.dispose();
}

SplatViewer.displayName = "SplatViewer";
export default SplatViewer;
