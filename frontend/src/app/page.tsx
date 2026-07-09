"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { generateWorld, getGallery, type GallerySplat, SPLAT_URL_BASE } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [recentProjects, setRecentProjects] = useState<GallerySplat[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load recent projects on mount
  useEffect(() => {
    getGallery().then(setRecentProjects).catch(() => {});
  }, []);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setSelectedImage(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleGenerate = useCallback(async () => {
    if (!selectedImage) return;
    try {
      const { job_id } = await generateWorld(selectedImage);
      sessionStorage.setItem("lucidframe-job-id", job_id);
      sessionStorage.removeItem("lucidframe-splat-url");
      sessionStorage.removeItem("lucidframe-splat-name");
      router.push("/workspace");
    } catch (err) {
      console.error("Failed to start pipeline:", err);
    }
  }, [selectedImage, router]);

  const handleReplace = useCallback(() => {
    setSelectedImage(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  return (
    <div className="h-full overflow-y-auto thin-scroll">
      <div className="mx-auto max-w-[1100px] px-8 py-16">
        {/* Heading */}
        <div className="mb-12">
          <h1 className="text-[40px] font-semibold leading-[1.1] tracking-tight text-white">
            Turn one image into a world.
          </h1>
          <p className="mt-4 text-[15px] leading-relaxed text-zinc-400">
            Upload one photograph and LucidFrame reconstructs an explorable 3D Gaussian Splat scene.
          </p>
        </div>

        {/* Upload + CTA */}
        <div className="flex gap-8">
          {/* Upload panel */}
          <div className="w-[60%]">
            {!previewUrl ? (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`flex h-[420px] cursor-pointer flex-col items-center justify-center gap-4 border transition-all duration-200 ${
                  isDragging
                    ? "border-white/20 bg-white/[0.02]"
                    : "border-white/[0.06] bg-surface hover:border-white/[0.1] hover:bg-elevated"
                }`}
              >
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white/[0.04]">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-zinc-400">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <div className="text-center">
                  <p className="text-[17px] font-medium text-white">Drag & Drop</p>
                  <p className="mt-1 text-[13px] text-zinc-500">
                    PNG, JPG, WebP — up to 20MB
                  </p>
                </div>
                <button className="mt-2 rounded-btn border border-white/[0.08] bg-elevated px-4 py-2 text-[13px] font-medium text-zinc-200 transition-colors hover:bg-white/[0.06]">
                  Browse
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFile(file);
                  }}
                />
              </div>
            ) : (
              <div className="relative h-[420px] overflow-hidden rounded-dialog border border-white/[0.06] bg-surface">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="h-full w-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <button
                  onClick={handleReplace}
                  className="absolute bottom-4 left-4 rounded-btn border border-white/[0.1] bg-black/50 px-3 py-1.5 text-[13px] font-medium text-white backdrop-blur-sm transition-colors hover:bg-black/70"
                >
                  Replace image
                </button>
              </div>
            )}
          </div>

          {/* CTA */}
          <div className="flex flex-1 flex-col justify-end pb-4">
            <button
              onClick={handleGenerate}
              disabled={!selectedImage}
              className="flex items-center justify-center gap-2 rounded-btn bg-white px-6 py-3.5 text-[15px] font-semibold text-black transition-all duration-200 hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-30"
            >
              Generate World
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </button>
            {!selectedImage && (
              <p className="mt-3 text-[13px] text-zinc-600">
                Upload an image to get started
              </p>
            )}
          </div>
        </div>

        {/* Recent Projects */}
        <div className="mt-16">
          <div className="mb-5 flex items-center justify-between">
            <h2 className="text-[15px] font-semibold text-white">Recent Projects</h2>
            {recentProjects.length > 0 && (
              <button className="text-[13px] text-zinc-500 transition-colors hover:text-zinc-300">
                View all
              </button>
            )}
          </div>

          {recentProjects.length > 0 ? (
            <div className="grid grid-cols-4 gap-4">
              {recentProjects.slice(0, 4).map((project) => (
                <ProjectCard key={project.name} project={project} />
              ))}
            </div>
          ) : (
            <div className="flex h-32 items-center justify-center rounded-card border border-white/[0.04] bg-surface">
              <p className="text-[14px] text-zinc-600">Your generated worlds will appear here.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ProjectCard({ project }: { project: GallerySplat }) {
  const router = useRouter();
  const handleClick = () => {
    const fullUrl = project.url.startsWith("http")
      ? project.url
      : `${SPLAT_URL_BASE}${project.url}`;
    sessionStorage.setItem("lucidframe-splat-url", fullUrl);
    sessionStorage.setItem("lucidframe-splat-name", project.name);
    router.push("/workspace");
  };

  return (
    <button
      onClick={handleClick}
      className="group overflow-hidden rounded-card border border-white/[0.04] bg-surface text-left transition-all duration-200 hover:border-white/[0.08] hover:bg-elevated"
    >
      <div className="relative aspect-[4/3] overflow-hidden bg-elevated">
        <div className="flex h-full items-center justify-center">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-zinc-700">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
          <span className="rounded-btn bg-white/10 px-3 py-1.5 text-[13px] font-medium text-white backdrop-blur-sm">
            Open
          </span>
        </div>
      </div>
      <div className="p-3">
        <p className="truncate text-[13px] font-medium text-white">{project.name}</p>
        <p className="mt-0.5 text-[12px] text-zinc-500">
          {(project.size_kb / 1024).toFixed(1)} MB
        </p>
      </div>
    </button>
  );
}
