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

export type SplatRenderQuality = "standard" | "high";

interface SplatViewerProps {
  splatUrl: string | null;
  className?: string;
  mode: "orbit" | "fps";
  sceneMode?: "image" | "panorama";
  renderQuality?: SplatRenderQuality;
  onModeChange?: (mode: "orbit" | "fps") => void;
}

type Control = {
  update: () => void;
  dispose: () => void;
  setCameraTarget?: (target: unknown) => void;
  moveSpeed?: number;
  lookSpeed?: number;
};

type AdaptiveRenderProgram = {
  setCovariancePadding: (value: number) => void;
};

const SplatViewer = forwardRef<SplatViewerHandle, SplatViewerProps>(
  ({ splatUrl, className = "", mode, sceneMode = "image", renderQuality = "standard" }, ref) => {
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
      adaptiveRenderProgram: AdaptiveRenderProgram | null;
    } | null>(null);
    const modeRef = useRef(mode);
    modeRef.current = mode;
    const renderQualityRef = useRef(renderQuality);
    renderQualityRef.current = renderQuality;
    const [showHint, setShowHint] = useState(true);
    const [isLoading, setIsLoading] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);
    const [loadError, setLoadError] = useState<string | null>(null);

    const makeControls = useCallback((engine: NonNullable<typeof engineRef.current>, nextMode: "orbit" | "fps") => {
      if (nextMode === "fps") {
        return createFreeFlyControls(
          engine.camera,
          engine.renderer.canvas,
          engine.SPLAT,
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
          const adaptiveRenderProgram = installAdaptiveRenderProgram(SPLAT, renderer);
          adaptiveRenderProgram?.setCovariancePadding(covariancePadding(renderQualityRef.current));
          canvasRef.current.dataset.renderQuality = renderQualityRef.current;
          canvasRef.current.dataset.adaptiveRenderer = adaptiveRenderProgram ? "available" : "fallback";
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
            adaptiveRenderProgram,
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
            resizeRendererForDisplay(current.renderer, renderQualityRef.current);
            const cameraPosition = current.camera.position;
            const translation = Math.hypot(
              Number(cameraPosition.x) || 0,
              Number(cameraPosition.y) || 0,
              Number(cameraPosition.z) || 0,
            );
            current.adaptiveRenderProgram?.setCovariancePadding(
              covariancePadding(renderQualityRef.current, translation),
            );
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

    useEffect(() => {
      if (canvasRef.current) canvasRef.current.dataset.renderQuality = renderQuality;
      const engine = engineRef.current;
      if (!engine) return;
      engine.adaptiveRenderProgram?.setCovariancePadding(covariancePadding(renderQuality));
    }, [renderQuality]);

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
            <p className="mt-3 text-sm">Your scene will appear here.</p>
          </div>
        </div>
      );
    }

    const hint = mode === "fps"
      ? "Drag to look. Use WASD to move, Q/E for height, and Shift to move faster."
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

function createFreeFlyControls(
  camera: any,
  canvas: HTMLCanvasElement,
  SPLAT: any,
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
    // Explore mode is deliberately free-flight. A local Gaussian scene has no
    // collision mesh, so restricting the camera behind an invisible radius is
    // more confusing than exposing the model's evidence boundary honestly.
    pitch = Math.max(-1.48, Math.min(1.48, pitch));
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
      const distance = delta * (sprinting ? 2.1 : 0.7);
      const forwardX = Math.sin(yaw);
      const forwardZ = Math.cos(yaw);
      const rightX = Math.cos(yaw);
      const rightZ = -Math.sin(yaw);
      const position = camera.position;
      camera.position = new SPLAT.Vector3(
        position.x + ((forward * forwardX + right * rightX) / length) * distance,
        position.y - (vertical / length) * distance,
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

function setCameraFov(camera: any, fov: number) {
  const width = camera.data.width || 800;
  const focal = (width / 2) / Math.tan((fov * Math.PI) / 360);
  camera.data.fx = focal;
  camera.data.fy = focal;
}

function resizeRendererForDisplay(renderer: any, quality: SplatRenderQuality) {
  const canvas = renderer.canvas as HTMLCanvasElement;
  // Standard preserves gsplat.js's original CSS-pixel drawing buffer exactly.
  // HD allocates a bounded high-DPI buffer while preserving the CSS layout.
  // It is presentation-only supersampling: the learned scene stays untouched
  // and the user can disable it instantly if a device cannot sustain it.
  const pixelRatio = quality === "high"
    ? Math.min((window.devicePixelRatio || 1) * 1.18, 2)
    : 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * pixelRatio));
  const height = Math.max(1, Math.round(canvas.clientHeight * pixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    renderer.setSize(width, height);
  }
}

function covariancePadding(quality: SplatRenderQuality, cameraTranslation = 0) {
  // gsplat.js 1.2.9 and the reference 3DGS rasterizer use 0.3 px². The HD
  // option raises this only slightly; larger values noticeably dilate edges.
  const base = quality === "high" ? 0.45 : 0.3;
  const ceiling = quality === "high" ? 0.72 : 0.60;
  return Math.min(ceiling, base + Math.max(0, cameraTranslation) * 0.12);
}

function installAdaptiveRenderProgram(SPLAT: typeof import("gsplat"), renderer: any): AdaptiveRenderProgram | null {
  try {
    class LucidFrameRenderProgram extends SPLAT.RenderProgram {
      protected override _getVertexSource(): string {
        let source = super._getVertexSource();
        source = source.replace(
          "uniform vec2 viewport;",
          "uniform vec2 viewport;\nuniform float lucidframeCovariancePadding;",
        );
        source = source.replace(
          /cov2d\[0\]\[0\]\s*\+=\s*0\.3;/,
          [
            "float lucidframeBaselineDeterminant = max((cov2d[0][0] + 0.3) * (cov2d[1][1] + 0.3) - cov2d[0][1] * cov2d[0][1], 1e-8);",
            "cov2d[0][0] += lucidframeCovariancePadding;",
          ].join("\n"),
        );
        source = source.replace(
          /cov2d\[1\]\[1\]\s*\+=\s*0\.3;/,
          [
            "cov2d[1][1] += lucidframeCovariancePadding;",
            "float lucidframeFilteredDeterminant = max(cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1], 1e-8);",
            "float lucidframeOpacityCompensation = sqrt(clamp(lucidframeBaselineDeterminant / lucidframeFilteredDeterminant, 0.0, 1.0));",
          ].join("\n"),
        );
        source = source.replace(
          "vColor = colorTransform * color;",
          "vColor = colorTransform * color;\nvColor.a *= lucidframeOpacityCompensation;",
        );
        const covarianceWasReplaced =
          source.includes("cov2d[0][0] += lucidframeCovariancePadding;") &&
          source.includes("cov2d[1][1] += lucidframeCovariancePadding;");
        const opacityWasCompensated = source.includes("vColor.a *= lucidframeOpacityCompensation;");
        if (!source.includes("uniform float lucidframeCovariancePadding;") || !covarianceWasReplaced || !opacityWasCompensated) {
          throw new Error("The installed gsplat.js shader no longer matches the HD renderer adapter.");
        }
        return source;
      }

      setCovariancePadding(value: number) {
        const gl = this.renderer.gl;
        gl.useProgram(this.program);
        const location = gl.getUniformLocation(this.program, "lucidframeCovariancePadding");
        if (location !== null) gl.uniform1f(location, value);
      }
    }

    const baseline = renderer.renderProgram;
    const program = new LucidFrameRenderProgram(renderer, baseline.passes);
    const gl = renderer.gl as WebGL2RenderingContext;
    if (!gl.getProgramParameter(program.program, gl.LINK_STATUS)) {
      gl.deleteProgram(program.program);
      throw new Error("The HD Gaussian shader did not link on this WebGL device.");
    }
    renderer.removeProgram(baseline);
    renderer.addProgram(program);
    return program;
  } catch (caught) {
    console.warn("LucidFrame HD rendering is unavailable; using the original gsplat.js program.", caught);
    return null;
  }
}

SplatViewer.displayName = "SplatViewer";
export default SplatViewer;
