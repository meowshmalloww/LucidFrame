"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  generateWorld,
  getProviders,
  type PipelineProvider,
  type ProviderStatus,
  type QualityProfile,
} from "@/lib/api";

type ImageInfo = { width: number; height: number; ratio: number };

const modes: Array<{
  id: PipelineProvider;
  title: string;
  meta: string;
  description: string;
  icon: "frame" | "pano" | "cloud";
}> = [
  {
    id: "local",
    title: "Image to 3D",
    meta: "On this computer",
    description: "Create an explorable scene from one photo.",
    icon: "frame",
  },
  {
    id: "local_world",
    title: "Image to 360",
    meta: "On this computer · Slowest",
    description: "Generate and connect every direction around one photo.",
    icon: "pano",
  },
  {
    id: "local_pano",
    title: "Panorama to 360",
    meta: "On this computer",
    description: "Turn a wide or equirectangular image into a scene.",
    icon: "pano",
  },
  {
    id: "worldlabs",
    title: "World Labs",
    meta: "Hosted · Uses credits",
    description: "Create a Marble world from one image.",
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
  const [qualityProfile, setQualityProfile] = useState<QualityProfile>("detail");
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
      const result = await generateWorld(selected, mode, "", qualityProfile);
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
  }, [backendReady, mode, preview, providers.worldlabs?.available, providers.worldlabs?.estimate_label, qualityProfile, router, selected, starting]);

  const panoProfile = describePanorama(imageInfo);
  const canBegin = Boolean(selected) && !starting && backendReady === true;

  return (
    <div className="h-full overflow-y-auto bg-[#f3f2ec] text-[#1b1b18]">
      <div className="mx-auto w-full max-w-[1320px] px-6 py-8 lg:px-10 lg:py-11">
        <header className="border-b border-[#d5d3ca] pb-7">
          <h1 className="text-[clamp(2rem,4vw,3.25rem)] font-medium leading-none tracking-[-0.045em]">Create a scene</h1>
          <p className="mt-3 text-sm leading-6 text-[#686760]">Choose an image, then select how you want to build it.</p>
        </header>

        <section className="mt-8 grid gap-7 lg:grid-cols-[minmax(0,1fr)_360px]">
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
                "relative flex min-h-[450px] cursor-pointer items-center justify-center overflow-hidden rounded-[10px] border bg-[#e7e5dc] transition",
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

          <aside className="self-start rounded-[10px] border border-[#d1cfc5] bg-[#faf9f5] p-5">
            <div className="border-b border-[#dedcd3] pb-4">
              <h2 className="text-sm font-semibold">Method</h2>
              <p className="mt-1.5 text-xs leading-5 text-[#686760]">Choose the option that matches your image.</p>
            </div>

            <div className="mt-4 grid gap-2">
              {modes.map((item) => (
                <ModeChoice
                  key={item.id}
                  item={item}
                  active={mode === item.id}
                  available={providers[item.id]?.available !== false}
                  onClick={() => {
                    setMode(item.id);
                    setQualityProfile(item.id === "local" ? "detail" : "balanced");
                    setError(null);
                  }}
                />
              ))}
            </div>

            {mode === "local_pano" && <p className="mt-4 text-xs leading-5 text-[#686760]">{panoProfile}</p>}
            {mode === "worldlabs" && (
              <p className="mt-4 text-xs leading-5 text-[#686760]">
                {providers.worldlabs?.available ? (providers.worldlabs.estimate_label || "World Labs is connected.") : "Add an API key in Settings before using this option."}
              </p>
            )}

            {mode !== "worldlabs" && (
              <fieldset className="mt-5 border-t border-[#dedcd3] pt-4">
                <legend className="text-xs font-semibold text-[#4f4e48]">Quality</legend>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  <QualityChoice
                    active={qualityProfile === "balanced"}
                    title="Balanced"
                    note={mode === "local_pano" ? "4 overlapping views" : mode === "local_world" ? "Fewer generation steps" : "Standard input"}
                    onClick={() => setQualityProfile("balanced")}
                  />
                  <QualityChoice
                    active={qualityProfile === "detail"}
                    title="Detail"
                    note={mode === "local_pano" ? "6 views, larger file" : mode === "local_world" ? "More generation steps" : "Restore small images"}
                    onClick={() => setQualityProfile("detail")}
                  />
                </div>
                <p className="mt-3 text-[11px] leading-5 text-[#77766f]">
                  {mode === "local_world"
                    ? "Detail adds more generation steps and takes longer."
                    : mode === "local_pano"
                    ? "Detail adds two more overlapping views and creates a larger file."
                    : "Detail restores small source images before reconstruction."}
                </p>
              </fieldset>
            )}

            {backendReady === false && (
              <p role="status" className="mt-4 border-l-2 border-[#a23d37] pl-3 text-xs leading-5 text-[#8c332f]">
                The local backend is unavailable. Start it on port 8000 and refresh.
              </p>
            )}

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
              <span>{starting ? "Starting…" : actionLabel(mode)}</span>
              <svg viewBox="0 0 24 24" className="h-4 w-4 fill-none stroke-current stroke-2" aria-hidden>
                <path d="M5 12h14M14 7l5 5-5 5" />
              </svg>
            </button>
          </aside>
        </section>

      </div>
    </div>
  );
}

function describePanorama(info: ImageInfo | null) {
  if (!info) return "Landscape panoramas work best. A 2:1 image includes the full sphere.";
  if (info.ratio >= 1.9 && info.ratio <= 2.1) return "This looks like a full 2:1 panorama.";
  if (info.ratio > 2.1) return "This wide panorama can run, but the top and bottom may need to be filled.";
  if (info.ratio >= 1.2) return "This appears to be a partial panorama. Missing directions will be estimated.";
  return "This does not look like a panorama. Image to 3D may work better.";
}

function actionLabel(mode: PipelineProvider) {
  if (mode === "worldlabs") return "Create with World Labs";
  return "Create scene";
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
      <p className="mt-2 text-sm leading-6 text-[#6d6c65]">Drop an image here or choose a JPG, PNG, or WebP file.</p>
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
      disabled={!available}
      onClick={onClick}
      className={[
        "grid w-full grid-cols-[34px_1fr] gap-3 rounded-[9px] border p-3.5 text-left transition disabled:cursor-not-allowed disabled:opacity-55",
        active ? "border-[#315c4a] bg-[#edf2ee]" : "border-[#dedcd3] bg-white hover:border-[#aaa89e]",
      ].join(" ")}
    >
      <ModeIcon type={item.icon} />
      <span>
        <span className="flex items-center justify-between gap-2">
          <span className="text-sm font-semibold text-[#24241f]">{item.title}</span>
          {!available && <span className="text-[11px] text-[#8d5f26]">{item.id === "worldlabs" ? "Needs API key" : "Not installed"}</span>}
        </span>
        <span className="mt-0.5 block text-[11px] text-[#77766f]">{item.meta}</span>
        <span className="mt-1.5 block text-xs leading-5 text-[#686760]">{item.description}</span>
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

function QualityChoice({
  active,
  title,
  note,
  onClick,
}: {
  active: boolean;
  title: string;
  note: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={active ? "rounded-[8px] border border-[#315c4a] bg-[#edf2ee] p-3 text-left" : "rounded-[8px] border border-[#d5d3ca] bg-white p-3 text-left hover:border-[#96948b]"}
    >
      <span className="block text-xs font-semibold">{title}</span>
      <span className="mt-1 block text-[10px] leading-4 text-[#77766f]">{note}</span>
    </button>
  );
}
