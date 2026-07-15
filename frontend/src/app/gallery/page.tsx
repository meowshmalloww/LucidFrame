"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  deleteGalleryProjects,
  getGallery,
  type GallerySplat,
  SPLAT_URL_BASE,
} from "@/lib/api";

export default function GalleryPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<GallerySplat[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<GallerySplat | null>(null);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getGallery()
      .then(setProjects)
      .catch((caught) => setError(caught instanceof Error ? caught.message : "The library could not be loaded."))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(
    () => projects.filter((project) => project.name.toLowerCase().includes(search.trim().toLowerCase())),
    [projects, search],
  );
  const totalBytes = useMemo(() => projects.reduce((sum, project) => sum + project.size_kb * 1024, 0), [projects]);
  const selectable = filtered.flatMap((project) => project.id ? [project.id] : []);
  const allVisibleChecked = selectable.length > 0 && selectable.every((id) => checked.has(id));

  const toggle = (id: string) => {
    setChecked((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleVisible = () => {
    setChecked((current) => {
      const next = new Set(current);
      if (allVisibleChecked) selectable.forEach((id) => next.delete(id));
      else selectable.forEach((id) => next.add(id));
      return next;
    });
  };

  const remove = async (ids: string[]) => {
    if (ids.length === 0 || deleting) return;
    const label = ids.length === 1 ? "this scene" : `${ids.length} scenes`;
    if (!window.confirm(`Delete ${label} and the matching source files from this computer? This cannot be undone.`)) return;
    setDeleting(true);
    setError(null);
    setNotice(null);
    try {
      const result = await deleteGalleryProjects(ids);
      const removed = new Set(result.deleted);
      setProjects((current) => current.filter((project) => !project.id || !removed.has(project.id)));
      setChecked((current) => new Set([...current].filter((id) => !removed.has(id))));
      if (selected?.id && removed.has(selected.id)) setSelected(null);
      setNotice(`Deleted ${result.deleted.length} ${result.deleted.length === 1 ? "scene" : "scenes"} and reclaimed ${formatBytes(result.freed_bytes)}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The selected scenes could not be deleted.");
    } finally {
      setDeleting(false);
    }
  };

  const open = (project: GallerySplat) => {
    const splatUrl = absoluteUrl(project.url);
    sessionStorage.setItem("lucidframe-splat-url", splatUrl);
    sessionStorage.setItem("lucidframe-project-name", project.name);
    sessionStorage.setItem("lucidframe-provider", project.provider || "local");
    if (project.source_url) sessionStorage.setItem("lucidframe-source-preview", absoluteUrl(project.source_url));
    router.push("/workspace");
  };

  return (
    <div className="h-full overflow-y-auto bg-[#f3f2ec] text-[#1b1b18]">
      <div className="mx-auto w-full max-w-[1320px] px-6 py-9 lg:px-10 lg:py-12">
        <header className="flex flex-col gap-5 border-b border-[#d5d3ca] pb-7 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-[clamp(2rem,4vw,3.25rem)] font-medium leading-none tracking-[-0.045em]">Library</h1>
            <p className="mt-3 text-sm text-[#686760]">
              {loading ? "Loading scenes…" : `${projects.length} ${projects.length === 1 ? "scene" : "scenes"}${projects.length ? ` · ${formatBytes(totalBytes)}` : ""}`}
            </p>
          </div>
          <div className="flex w-full max-w-xl flex-col gap-2 sm:items-end">
            <label className="relative block w-full max-w-xs">
              <span className="sr-only">Search scenes</span>
              <svg viewBox="0 0 24 24" className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 fill-none stroke-[#77766f] stroke-[1.7]" aria-hidden><circle cx="11" cy="11" r="7" /><path d="m16 16 4 4" /></svg>
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search library"
                className="w-full rounded-[8px] border border-[#bbb9af] bg-[#faf9f5] py-2.5 pl-9 pr-3 text-sm outline-none focus:border-[#315c4a]"
              />
            </label>
          </div>
        </header>

        {!loading && projects.length > 0 && (
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-b border-[#d5d3ca] pb-5">
            <label className="flex cursor-pointer items-center gap-2 text-xs font-semibold text-[#5f5e58]">
              <input
                type="checkbox"
                checked={allVisibleChecked}
                onChange={toggleVisible}
                className="h-4 w-4 accent-[#315c4a]"
              />
              Select {search ? "matching" : "all"}
            </label>
            <div className="flex items-center gap-3">
              {(notice || error) && <p className={error ? "text-xs text-[#963b35]" : "text-xs text-[#315c4a]"}>{error || notice}</p>}
              {checked.size > 0 && (
                <button
                  type="button"
                  disabled={deleting}
                  onClick={() => remove([...checked])}
                  className="inline-flex items-center gap-2 rounded-[8px] border border-[#b89a95] bg-[#faf9f5] px-3 py-2 text-xs font-semibold text-[#823b36] hover:border-[#963b35] disabled:opacity-50"
                >
                  <TrashIcon />
                  {deleting ? "Deleting..." : `Delete selected (${checked.size})`}
                </button>
              )}
            </div>
          </div>
        )}

        {loading ? (
          <LibrarySkeleton />
        ) : filtered.length === 0 ? (
          <EmptyLibrary search={search} onCreate={() => router.push("/")} />
        ) : (
          <div className="mt-7 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {filtered.map((project) => (
              <article key={project.url} className="overflow-hidden rounded-[10px] border border-[#d1cfc5] bg-[#faf9f5]">
                <button type="button" onClick={() => open(project)} className="group block w-full text-left">
                  <div className="relative aspect-[16/10] overflow-hidden bg-[#dedcd3]">
                    {project.id && (
                      <span
                        role="checkbox"
                        aria-checked={checked.has(project.id)}
                        aria-label={`Select ${cleanName(project.name)}`}
                        tabIndex={0}
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          toggle(project.id!);
                        }}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            event.stopPropagation();
                            toggle(project.id!);
                          }
                        }}
                        className="absolute left-3 top-3 z-10 grid h-7 w-7 place-items-center rounded-[6px] border border-white/40 bg-[#1b1b18]/80 text-white shadow"
                      >
                        {checked.has(project.id) && <CheckIcon />}
                      </span>
                    )}
                    {project.source_url ? (
                      // eslint-disable-next-line @next/next/no-img-element -- images come from the local FastAPI origin.
                      <img src={absoluteUrl(project.source_url)} alt="" className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.015]" />
                    ) : (
                      <div className="grid h-full place-items-center">
                        <svg viewBox="0 0 48 48" className="h-12 w-12 fill-none stroke-[#77766f] stroke-[1.2]" aria-hidden><path d="M24 5 7 14l17 9 17-9-17-9Z" /><path d="m7 23 17 9 17-9M7 32l17 9 17-9" /></svg>
                      </div>
                    )}
                  </div>
                </button>
                  <div className="p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <h2 className="truncate text-sm font-semibold">{cleanName(project.name)}</h2>
                        <p className="mt-1 text-xs text-[#77766f]">{providerLabel(project.provider)} / {(project.size_kb / 1024).toFixed(1)} MB</p>
                      </div>
                      <div className="flex gap-1.5">
                        <button
                          type="button"
                          aria-label="Show scene details"
                          onClick={(event) => {
                            event.stopPropagation();
                            setSelected(project);
                          }}
                          className="rounded-[7px] border border-[#c6c3b9] px-2.5 py-1.5 text-xs font-semibold hover:border-[#77766f]"
                        >
                          Details
                        </button>
                        {project.id && (
                          <button
                            type="button"
                            aria-label={`Delete ${cleanName(project.name)}`}
                            onClick={() => remove([project.id!])}
                            className="rounded-[7px] border border-[#d0b8b4] p-1.5 text-[#823b36] hover:border-[#963b35]"
                          >
                            <TrashIcon />
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
              </article>
            ))}
          </div>
        )}
      </div>

      {selected && (
        <div className="fixed inset-0 z-40 grid place-items-center bg-[#1b1b18]/45 p-5" onMouseDown={() => setSelected(null)}>
          <section
            role="dialog"
            aria-modal="true"
            aria-label="Scene details"
            onMouseDown={(event) => event.stopPropagation()}
            className="w-full max-w-lg rounded-[12px] border border-[#cbc9bf] bg-[#faf9f5] p-5 shadow-2xl"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold">{cleanName(selected.name)}</h2>
              </div>
              <button type="button" onClick={() => setSelected(null)} className="rounded-[7px] border border-[#c6c3b9] p-2" aria-label="Close">
                <svg viewBox="0 0 24 24" className="h-4 w-4 stroke-current stroke-2" aria-hidden><path d="m6 6 12 12M18 6 6 18" /></svg>
              </button>
            </div>
            <dl className="mt-5 divide-y divide-[#dedcd3] border-y border-[#dedcd3] text-sm">
              <Detail label="Method" value={providerLabel(selected.provider)} />
              <Detail label="File size" value={(selected.size_kb / 1024).toFixed(1) + " MB"} />
              <Detail label="Coverage" value={selected.coverage || (selected.provider === "local_pano" || selected.provider === "local_world" ? "360 look-around" : "nearby view")} />
              <Detail label="Created" value={selected.created_at ? new Date(selected.created_at * 1000).toLocaleString() : "Unknown"} />
            </dl>
            <div className="mt-5 flex gap-3">
              <button type="button" onClick={() => open(selected)} className="flex-1 rounded-[8px] bg-[#1b1b18] px-4 py-3 text-sm font-semibold text-white hover:bg-[#315c4a]">Open scene</button>
              <a href={absoluteUrl(selected.url)} download className="rounded-[8px] border border-[#bbb9af] bg-white px-4 py-3 text-sm font-semibold hover:border-[#77766f]">Save .splat</a>
              {selected.id && (
                <button type="button" onClick={() => remove([selected.id!])} className="rounded-[8px] border border-[#d0b8b4] bg-white px-3 py-3 text-[#823b36] hover:border-[#963b35]" aria-label="Delete scene">
                  <TrashIcon />
                </button>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function absoluteUrl(path: string) {
  return path.startsWith("http") ? path : SPLAT_URL_BASE + path;
}

function cleanName(value: string) {
  return value.replace(/[-_]/g, " ");
}

function providerLabel(provider?: GallerySplat["provider"]) {
  if (provider === "local_world") return "Image to 360";
  if (provider === "local_pano") return "Panorama to 360";
  if (provider === "worldlabs") return "World Labs";
  return "Image to 3D";
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-5 py-3"><dt className="text-[#77766f]">{label}</dt><dd className="text-right font-medium">{value}</dd></div>;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(0, bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function TrashIcon() {
  return <svg viewBox="0 0 24 24" className="h-4 w-4 fill-none stroke-current stroke-[1.8]" aria-hidden><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5" /></svg>;
}

function CheckIcon() {
  return <svg viewBox="0 0 24 24" className="h-4 w-4 fill-none stroke-current stroke-2" aria-hidden><path d="m5 12 4 4L19 6" /></svg>;
}

function LibrarySkeleton() {
  return (
    <div className="mt-7 grid gap-4 sm:grid-cols-2 xl:grid-cols-3" aria-busy="true" aria-label="Loading scenes">
      {[0, 1, 2, 3, 4, 5].map((item) => (
        <div key={item} className="overflow-hidden rounded-[10px] border border-[#d5d3ca] bg-[#faf9f5]">
          <div className="skeleton-shimmer aspect-[16/10] bg-[#e3e1d8]" />
          <div className="space-y-2.5 p-4">
            <div className="skeleton-shimmer h-3.5 w-2/3 bg-[#e3e1d8]" />
            <div className="skeleton-shimmer h-2.5 w-2/5 bg-[#e3e1d8]" />
          </div>
        </div>
      ))}
      <span className="sr-only" role="status" aria-live="polite">Loading scenes</span>
    </div>
  );
}

function EmptyLibrary({ search, onCreate }: { search: string; onCreate: () => void }) {
  return (
    <div className="mt-7 grid min-h-[420px] place-items-center rounded-[12px] border border-dashed border-[#c6c3b9] bg-[#eeece4] p-8 text-center">
      <div>
        <svg viewBox="0 0 48 48" className="mx-auto h-12 w-12 fill-none stroke-[#77766f] stroke-[1.2]" aria-hidden><path d="M8 10h32v28H8z" /><path d="m12 33 8-9 7 7 4-4 5 6" /></svg>
        <h2 className="mt-4 text-lg font-semibold">{search ? "No matching scenes" : "Your library is empty"}</h2>
        <p className="mt-2 text-sm text-[#686760]">{search ? "Try another name." : "Completed local generations are collected here automatically."}</p>
        {!search && <button type="button" onClick={onCreate} className="mt-5 rounded-[8px] bg-[#1b1b18] px-4 py-3 text-sm font-semibold text-white">Create a scene</button>}
      </div>
    </div>
  );
}
