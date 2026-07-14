"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  generateWorld,
  getProviders,
  type PipelineProvider,
  type ProviderStatus,
} from "@/lib/api";

type ImageInfo = { width: number; height: number; ratio: number };

const modes: Array<{
  id: PipelineProvider;
  title: string;
  eyebrow: string;
  description: string;
  icon: "frame" | "pano" | "cloud";
}> = [
  {
    id: "local",
    title: "Image to 3D",
    eyebrow: "Local / no credits",
    description: "Predict a high-detail metric Gaussian scene from one photograph for nearby exploration.",
    icon: "frame",
  },
  {
    id: "local_pano",
    title: "Panorama to 360",
    eyebrow: "Local / no credits",
    description: "Merge overlapping learned Gaussian views with depth alignment and measured pole coverage.",
    icon: "pano",
  },
  {
    id: "worldlabs",
    title: "Image to World",
    eyebrow: "World Labs / paid",
    description: "Send one normal image to Marble and open the result in its hosted world viewer.",
    icon: "cloud",
  },
];

export function ArtHome() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [imageInfo, setImageInfo] = useState<ImageInfo | null>(null);
  const [dragging, setDragging] = useState(false);
  const [mode, setMode] = useState<PipelineProvider>("local");
  const [providers, setProviders] = useState<Partial<Record<PipelineProvider, ProviderStatus>>>({});
  const [backendReady, setBackendReady] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    getProviders()
      .then((value) => {
        setProviders(value);
        setBackendReady(true);
      })
      .catch(() => setBackendReady(false));
  }, []);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const chooseFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Choose a JPG, PNG, or WebP image.");
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      setImageInfo({
        width: image.naturalWidth,
        height: image.naturalHeight,
        ratio: image.naturalWidth / Math.max(image.naturalHeight, 1),
      });
    };
    image.onerror = () => setError("This image could not be decoded.");
    image.src = objectUrl;

    setError(null);
    setPreview((current) => {
      if (current) URL.revokeObjectURL(current);
      return objectUrl;
    });
    setSelected(file);
  }, []);

  const clearFile = useCallback(() => {
    setSelected(null);
    setImageInfo(null);
    setPreview((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const begin = useCallback(async () => {
    if (!selected || starting) return;
    if (backendReady === false) {
      setError("The local backend is not reachable. Start FastAPI on port 8000, then refresh this page.");
      return;
    }
    if (mode === "worldlabs" && providers.worldlabs?.available !== true) {
      setError("Add a World Labs API key on the Settings page before using the paid mode.");
      return;
    }
    if (
      mode === "worldlabs" &&
      !window.confirm(
        "This starts a paid World Labs generation (" +
          (providers.worldlabs?.estimate_label || "credits depend on the selected model") +
          "). Continue?",
      )
    ) {
      return;
    }

    setStarting(true);
    setError(null);
    try {
      const result = await generateWorld(selected, mode, "");
      sessionStorage.setItem("lucidframe-job-id", result.job_id);
      sessionStorage.setItem("lucidframe-project-name", selected.name.replace(/\.[^.]+$/, ""));
      sessionStorage.setItem("lucidframe-provider", mode);
      if (preview) sessionStorage.setItem("lucidframe-source-preview", preview);
      sessionStorage.removeItem("lucidframe-splat-url");
      router.push("/workspace");
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Generation could not start.";
      setError(message);
    } finally {
      setStarting(false);
    }
  }, [backendReady, mode, preview, providers.worldlabs?.available, providers.worldlabs?.estimate_label, router, selected, starting]);

  const selectedMode = modes.find((item) => item.id === mode) || modes[0];
  const panoProfile = describePanorama(imageInfo);
  const canBegin = Boolean(selected) && !starting && backendReady !== false;

  return (
    <div className="h-full overflow-y-auto bg-[#f3f2ec] text-[#1b1b18]">
      <div className="mx-auto w-full max-w-[1320px] px-6 py-8 lg:px-10 lg:py-11">
        <header className="grid gap-4 border-b border-[#d5d3ca] pb-8 lg:grid-cols-[1fr_28rem] lg:items-end">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#707068]">Spatial art studio</p>
            <h1 className="mt-3 max-w-3xl text-[clamp(2.3rem,5vw,4.8rem)] font-medium leading-[0.96] tracking-[-0.055em]">
              Continue an image beyond its frame.
            </h1>
          </div>
          <div className="flex items-start justify-between gap-6 lg:pb-1">
            <p className="max-w-sm text-sm leading-6 text-[#6d6c65]">
              Build a real Gaussian-splat scene locally, or connect World Labs when a generated world is worth the credits.
            </p>
            <ConnectionStatus ready={backendReady} />
          </div>
        </header>

        <section className="mt-8 grid gap-7 xl:grid-cols-[minmax(0,1fr)_370px]">
          <div className="min-w-0">
            <div
              role="button"
              tabIndex={0}
              aria-label="Choose source image"
              onClick={() => inputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                const file = event.dataTransfer.files[0];
                if (file) chooseFile(file);
              }}
              className={[
                "relative flex min-h-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-[12px] border bg-[#e7e5dc] transition",
                dragging ? "border-[#315c4a] bg-[#dfe8e2]" : "border-[#c9c7bd] hover:border-[#96948b]",
              ].join(" ")}
            >
              {preview ? (
                // eslint-disable-next-line @next/next/no-img-element -- blob previews are local and have runtime dimensions.
                <img src={preview} alt="Selected source" className="h-full max-h-[660px] w-full object-contain" />
              ) : (
                <EmptyUpload />
              )}
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) chooseFile(file);
                }}
              />
            </div>

            <div className="mt-3 flex min-h-11 flex-wrap items-center justify-between gap-3 border-b border-[#d5d3ca] pb-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">{selected?.name || "No source selected"}</p>
                <p className="mt-0.5 text-xs text-[#77766f]">
                  {imageInfo ? imageInfo.width + " x " + imageInfo.height + " / " + imageInfo.ratio.toFixed(2) + ":1" : "JPG, PNG or WebP"}
                </p>
              </div>
              {selected && (
                <button
                  type="button"
                  onClick={clearFile}
                  className="rounded-[8px] border border-[#bbb9af] bg-[#faf9f5] px-3 py-2 text-xs font-medium hover:border-[#77766f]"
                >
                  Replace image
                </button>
              )}
            </div>
          </div>

          <aside className="self-start rounded-[12px] border border-[#d1cfc5] bg-[#faf9f5] p-5 shadow-[0_18px_45px_rgba(39,39,33,0.06)]">
            <div className="border-b border-[#dedcd3] pb-4">
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#77766f]">Reconstruction method</p>
              <p className="mt-2 text-sm leading-6 text-[#5f5e58]">Choose based on what the source actually contains.</p>
            </div>

            <div className="mt-4 grid gap-2">
              {modes.map((item) => (
                <ModeChoice
                  key={item.id}
                  item={item}
                  active={mode === item.id}
                  available={item.id !== "worldlabs" || providers.worldlabs?.available === true}
                  onClick={() => {
                    setMode(item.id);
                    setError(null);
                  }}
                />
              ))}
            </div>

            <div className="mt-5 rounded-[9px] border border-[#dddacf] bg-[#f3f2ec] p-4">
              <p className="text-xs font-semibold">{selectedMode.title}</p>
              <p className="mt-2 text-xs leading-5 text-[#686760]">
                {mode === "local_pano"
                  ? panoProfile
                  : mode === "local"
                    ? "SHARP predicts 1.18M metric Gaussians at a fixed high inference resolution. Nearby views are strongest; a single photo still cannot reveal its back side. Research/non-commercial model license."
                    : providers.worldlabs?.available
                      ? (providers.worldlabs.estimate_label || "Paid hosted generation is connected.")
                      : "API key required. Configure it on the Settings page."}
              </p>
            </div>

            {error && (
              <p role="alert" className="mt-4 border-l-2 border-[#a23d37] pl-3 text-xs leading-5 text-[#8c332f]">
                {error}
              </p>
            )}

            <button
              type="button"
              onClick={begin}
              disabled={!canBegin}
              className="mt-5 flex w-full items-center justify-between rounded-[9px] bg-[#1c1c19] px-4 py-3.5 text-sm font-semibold text-white transition hover:bg-[#315c4a] disabled:cursor-not-allowed disabled:bg-[#b5b3aa]"
            >
              <span>{starting ? "Starting reconstruction..." : actionLabel(mode)}</span>
              <svg viewBox="0 0 24 24" className="h-4 w-4 fill-none stroke-current stroke-2" aria-hidden>
                <path d="M5 12h14M14 7l5 5-5 5" />
              </svg>
            </button>
          </aside>
        </section>

        <section className="mt-12 grid gap-6 border-t border-[#d5d3ca] pt-7 md:grid-cols-3">
          <Fact index="01" title="Visible evidence" text="Source pixels remain the visual anchor instead of being replaced by a text-only generation." />
          <Fact index="02" title="Learned geometry" text="SHARP predicts anisotropic Gaussian position, scale, orientation, color, and opacity rather than stretching one depth sheet." />
          <Fact index="03" title="Reliable motion" text="Perspective mode supports nearby motion; the eight-view panorama mode adds full look-around and bounded translation." />
        </section>
      </div>
    </div>
  );
}

function describePanorama(info: ImageInfo | null) {
  if (!info) return "Use a landscape panorama. A true 2:1 equirectangular capture gives SHARP-360 real pixels for the horizon, zenith, and nadir.";
  if (info.ratio >= 1.9 && info.ratio <= 2.1) return "Detected full-sphere equirectangular panorama. Learned Gaussian views will cover the horizon and both measured poles.";
  if (info.ratio > 2.1) return "Detected a vertically cropped or wide panorama. The horizon receives learned geometry, but absent zenith and nadir pixels must still be extended.";
  if (info.ratio >= 1.2) return "Detected a partial panorama. It can run, but unobserved side directions are synthesized by reflection and cannot match a true 360 capture.";
  return "This is not a landscape panorama. It can still be uploaded, but Image to 3D is the more reliable mode.";
}

function actionLabel(mode: PipelineProvider) {
  if (mode === "local_pano") return "Build local 360 scene";
  if (mode === "worldlabs") return "Create paid World Labs world";
  return "Build local 3D scene";
}

function ConnectionStatus({ ready }: { ready: boolean | null }) {
  return (
    <div className="flex shrink-0 items-center gap-2 rounded-[8px] border border-[#cbc9bf] bg-[#faf9f5] px-3 py-2 text-xs text-[#5f5e58]">
      <span className={ready === false ? "h-2 w-2 rounded-[2px] bg-[#a23d37]" : ready === true ? "h-2 w-2 rounded-[2px] bg-[#3f765e]" : "h-2 w-2 rounded-[2px] bg-[#a29f94]"} />
      {ready === false ? "Backend offline" : ready === true ? "Local backend ready" : "Checking backend"}
    </div>
  );
}

function EmptyUpload() {
  return (
    <div className="max-w-sm px-8 text-center">
      <svg viewBox="0 0 48 48" className="mx-auto h-12 w-12 fill-none stroke-[#5f5e58] stroke-[1.2]" aria-hidden>
        <rect x="6" y="8" width="36" height="30" rx="2" />
        <path d="m10 33 10-11 8 8 5-5 5 8" />
        <circle cx="16" cy="16" r="2.5" />
      </svg>
      <h2 className="mt-5 text-xl font-medium tracking-[-0.025em]">Choose a source image</h2>
      <p className="mt-2 text-sm leading-6 text-[#6d6c65]">Drop an image here, or click to browse. Nothing is uploaded to a third party in either local mode.</p>
      <span className="mt-5 inline-block rounded-[8px] border border-[#aaa89e] bg-[#f7f6f0] px-4 py-2 text-xs font-semibold">Browse files</span>
    </div>
  );
}

function ModeChoice({
  item,
  active,
  available,
  onClick,
}: {
  item: (typeof modes)[number];
  active: boolean;
  available: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "grid w-full grid-cols-[34px_1fr] gap-3 rounded-[9px] border p-3.5 text-left transition",
        active ? "border-[#315c4a] bg-[#edf2ee]" : "border-[#dedcd3] bg-white hover:border-[#aaa89e]",
      ].join(" ")}
    >
      <ModeIcon type={item.icon} />
      <span>
        <span className="flex items-center justify-between gap-2">
          <span className="text-sm font-semibold text-[#24241f]">{item.title}</span>
          {!available && item.id === "worldlabs" && <span className="text-[10px] uppercase tracking-wider text-[#8d5f26]">Key needed</span>}
        </span>
        <span className="mt-0.5 block text-[10px] font-semibold uppercase tracking-[0.12em] text-[#77766f]">{item.eyebrow}</span>
        <span className="mt-2 block text-xs leading-5 text-[#686760]">{item.description}</span>
      </span>
    </button>
  );
}

function ModeIcon({ type }: { type: "frame" | "pano" | "cloud" }) {
  if (type === "pano") {
    return <svg viewBox="0 0 32 32" className="h-8 w-8 fill-none stroke-[#3f5148] stroke-[1.35]" aria-hidden><ellipse cx="16" cy="16" rx="13" ry="7" /><path d="M3 16h26M16 9c3 2 4 4 4 7s-1 5-4 7M16 9c-3 2-4 4-4 7s1 5 4 7" /></svg>;
  }
  if (type === "cloud") {
    return <svg viewBox="0 0 32 32" className="h-8 w-8 fill-none stroke-[#3f5148] stroke-[1.35]" aria-hidden><path d="M9 24h14a6 6 0 0 0 1-11.9A8 8 0 0 0 9.2 9.5 5.5 5.5 0 0 0 9 24Z" /><path d="M16 13v8M12.5 16.5 16 13l3.5 3.5" /></svg>;
  }
  return <svg viewBox="0 0 32 32" className="h-8 w-8 fill-none stroke-[#3f5148] stroke-[1.35]" aria-hidden><rect x="5" y="6" width="22" height="20" rx="1.5" /><path d="m8 22 6-7 5 5 3-3 3 5" /></svg>;
}

function Fact({ index, title, text }: { index: string; title: string; text: string }) {
  return (
    <div className="grid grid-cols-[2rem_1fr] gap-3">
      <span className="text-[10px] font-semibold text-[#8a8981]">{index}</span>
      <div>
        <h2 className="text-sm font-semibold">{title}</h2>
        <p className="mt-2 text-xs leading-5 text-[#6d6c65]">{text}</p>
      </div>
    </div>
  );
}
