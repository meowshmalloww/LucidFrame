"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getGallery, type GallerySplat, SPLAT_URL_BASE } from "@/lib/api";

export default function GalleryPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<GallerySplat[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<GallerySplat | null>(null);

  useEffect(() => {
    getGallery()
      .then(setProjects)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const filtered = projects.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase()),
  );

  const handleOpen = (project: GallerySplat) => {
    const fullUrl = project.url.startsWith("http")
      ? project.url
      : `${SPLAT_URL_BASE}${project.url}`;
    sessionStorage.setItem("lucidframe-splat-url", fullUrl);
    sessionStorage.setItem("lucidframe-splat-name", project.name);
    router.push("/workspace");
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top bar */}
        <div className="flex h-12 shrink-0 items-center justify-between border-b border-white/[0.06] bg-surface px-6">
          <div className="flex items-center gap-3">
            <h1 className="text-[15px] font-semibold text-white">Gallery</h1>
            <span className="text-[13px] text-zinc-600">{filtered.length} projects</span>
          </div>
          <div className="flex items-center gap-2">
            {/* Search */}
            <div className="relative">
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-600"
              >
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search projects..."
                className="w-[200px] border border-white/[0.06] bg-elevated py-1.5 pl-8 pr-3 text-[13px] text-zinc-300 placeholder:text-zinc-600 focus:border-white/20 focus:outline-none"
              />
            </div>
            {/* Sort */}
            <button className="flex items-center gap-1.5 rounded-btn border border-white/[0.06] bg-elevated px-3 py-1.5 text-[13px] font-medium text-zinc-400 transition-colors hover:text-zinc-200">
              Sort
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
            {/* View toggle */}
            <div className="flex items-center gap-0.5 rounded-btn border border-white/[0.06] bg-elevated p-0.5">
              <button
                onClick={() => setViewMode("grid")}
                className={`rounded-[8px] p-1.5 transition-colors ${viewMode === "grid" ? "bg-white/[0.08] text-white" : "text-zinc-500 hover:text-zinc-300"}`}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="3" y="3" width="7" height="7" />
                  <rect x="14" y="3" width="7" height="7" />
                  <rect x="3" y="14" width="7" height="7" />
                  <rect x="14" y="14" width="7" height="7" />
                </svg>
              </button>
              <button
                onClick={() => setViewMode("list")}
                className={`rounded-[8px] p-1.5 transition-colors ${viewMode === "list" ? "bg-white/[0.08] text-white" : "text-zinc-500 hover:text-zinc-300"}`}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <line x1="8" y1="6" x2="21" y2="6" />
                  <line x1="8" y1="12" x2="21" y2="12" />
                  <line x1="8" y1="18" x2="21" y2="18" />
                  <line x1="3" y1="6" x2="3.01" y2="6" />
                  <line x1="3" y1="12" x2="3.01" y2="12" />
                  <line x1="3" y1="18" x2="3.01" y2="18" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto thin-scroll p-6">
          {loading ? (
            <div className="grid grid-cols-4 gap-4">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="aspect-[4/3] rounded-card border border-white/[0.04] bg-surface shimmer" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-4">
              <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-surface">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-zinc-700">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <circle cx="9" cy="9" r="2" />
                  <path d="M21 15l-5-5L5 21" />
                </svg>
              </div>
              <p className="text-[15px] text-zinc-500">Your generated worlds will appear here.</p>
              <button
                onClick={() => router.push("/")}
                className="rounded-btn bg-white px-4 py-2 text-[13px] font-semibold text-black transition-colors hover:bg-zinc-200"
              >
                Create your first world
              </button>
            </div>
          ) : viewMode === "grid" ? (
            <div className="grid grid-cols-4 gap-4">
              {filtered.map((project) => (
                <GalleryCard
                  key={project.name}
                  project={project}
                  onOpen={() => handleOpen(project)}
                  onSelect={() => setSelected(project)}
                />
              ))}
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {filtered.map((project) => (
                <GalleryRow
                  key={project.name}
                  project={project}
                  onOpen={() => handleOpen(project)}
                  onSelect={() => setSelected(project)}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Detail side panel */}
      {selected && (
        <div className="w-[300px] shrink-0 border-l border-white/[0.06] bg-surface p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-[15px] font-semibold text-white">Details</h2>
            <button
              onClick={() => setSelected(null)}
              className="flex h-7 w-7 items-center justify-center rounded-btn text-zinc-500 transition-colors hover:bg-white/[0.06] hover:text-zinc-300"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>

          {/* Preview */}
          <div className="mb-4 aspect-[4/3] overflow-hidden rounded-card border border-white/[0.04] bg-elevated">
            <div className="flex h-full items-center justify-center">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-zinc-700">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </div>
          </div>

          {/* Metadata */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-[13px] text-zinc-500">Name</span>
              <span className="text-[13px] font-medium text-zinc-200">{selected.name}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[13px] text-zinc-500">File size</span>
              <span className="text-[13px] font-mono text-zinc-300">{(selected.size_kb / 1024).toFixed(1)} MB</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[13px] text-zinc-500">Format</span>
              <span className="text-[13px] font-mono text-zinc-300">.splat</span>
            </div>
          </div>

          {/* Actions */}
          <div className="mt-6 flex flex-col gap-2">
            <button
              onClick={() => handleOpen(selected)}
              className="rounded-btn bg-white px-4 py-2 text-[13px] font-semibold text-black transition-colors hover:bg-zinc-200"
            >
              Open
            </button>
            <a
              href={selected.url.startsWith("http") ? selected.url : `${SPLAT_URL_BASE}${selected.url}`}
              download
              className="flex items-center justify-center gap-2 rounded-btn border border-white/[0.08] bg-elevated px-4 py-2 text-[13px] font-medium text-zinc-200 transition-colors hover:bg-white/[0.06]"
            >
              Download
            </a>
            <button className="flex items-center justify-center gap-2 rounded-btn border border-white/[0.08] bg-elevated px-4 py-2 text-[13px] font-medium text-danger transition-colors hover:bg-danger/10">
              Delete
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function GalleryCard({
  project,
  onOpen,
  onSelect,
}: {
  project: GallerySplat;
  onOpen: () => void;
  onSelect: () => void;
}) {
  return (
    <div
      className="group cursor-pointer overflow-hidden rounded-card border border-white/[0.04] bg-surface transition-all duration-200 hover:-translate-y-0.5 hover:border-white/[0.08] hover:bg-elevated"
      onClick={onSelect}
    >
      <div className="relative aspect-[4/3] overflow-hidden bg-elevated">
        <div className="flex h-full items-center justify-center">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-zinc-700">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <div className="absolute inset-0 flex items-center justify-center gap-2 bg-black/40 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onOpen();
            }}
            className="rounded-btn bg-white/10 px-3 py-1.5 text-[12px] font-medium text-white backdrop-blur-sm"
          >
            Open
          </button>
          <button className="rounded-btn bg-white/10 px-2.5 py-1.5 text-[12px] font-medium text-white backdrop-blur-sm">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="1" />
              <circle cx="19" cy="12" r="1" />
              <circle cx="5" cy="12" r="1" />
            </svg>
          </button>
        </div>
      </div>
      <div className="p-3">
        <p className="truncate text-[13px] font-medium text-white">{project.name}</p>
        <div className="mt-1 flex items-center gap-2 text-[11px] text-zinc-600">
          <span>{(project.size_kb / 1024).toFixed(1)} MB</span>
        </div>
      </div>
    </div>
  );
}

function GalleryRow({
  project,
  onOpen,
  onSelect,
}: {
  project: GallerySplat;
  onOpen: () => void;
  onSelect: () => void;
}) {
  return (
    <div
      className="group flex cursor-pointer items-center gap-4 rounded-card border border-white/[0.04] bg-surface px-4 py-3 transition-all duration-200 hover:border-white/[0.08] hover:bg-elevated"
      onClick={onSelect}
    >
      <div className="flex h-12 w-16 shrink-0 items-center justify-center rounded-btn bg-elevated">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-zinc-700">
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
      </div>
      <div className="flex flex-1 items-center justify-between">
        <div>
          <p className="text-[13px] font-medium text-white">{project.name}</p>
          <p className="mt-0.5 text-[11px] text-zinc-600">{(project.size_kb / 1024).toFixed(1)} MB</p>
        </div>
        <div className="flex items-center gap-2 opacity-0 transition-opacity group-hover:opacity-100">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onOpen();
            }}
            className="rounded-btn bg-white/[0.06] px-3 py-1.5 text-[12px] font-medium text-zinc-200"
          >
            Open
          </button>
        </div>
      </div>
    </div>
  );
}
