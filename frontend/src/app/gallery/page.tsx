"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { getGallery, type GallerySplat, SPLAT_URL_BASE } from "@/lib/api";

export default function GalleryPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<GallerySplat[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<GallerySplat | null>(null);

  useEffect(() => {
    getGallery()
      .then(setProjects)
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(
    () => projects.filter((project) => project.name.toLowerCase().includes(search.trim().toLowerCase())),
    [projects, search],
  );

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
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#77766f]">Library</p>
            <h1 className="mt-3 text-4xl font-medium tracking-[-0.045em]">Generated scenes</h1>
            <p className="mt-3 text-sm text-[#686760]">{projects.length} local splat files found</p>
          </div>
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
        </header>

        {loading ? (
          <div className="mt-7 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {[0, 1, 2, 3, 4, 5].map((item) => <div key={item} className="aspect-[4/3] rounded-[10px] border border-[#d5d3ca] bg-[#e7e5dc]" />)}
          </div>
        ) : filtered.length === 0 ? (
          <EmptyLibrary search={search} onCreate={() => router.push("/")} />
        ) : (
          <div className="mt-7 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {filtered.map((project) => (
              <article key={project.id || project.url} className="overflow-hidden rounded-[10px] border border-[#d1cfc5] bg-[#faf9f5]">
                <button type="button" onClick={() => open(project)} className="group block w-full text-left">
                  <div className="relative aspect-[16/10] overflow-hidden bg-[#dedcd3]">
                    {project.source_url ? (
                      // eslint-disable-next-line @next/next/no-img-element -- images come from the local FastAPI origin.
                      <img src={absoluteUrl(project.source_url)} alt="" className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.015]" />
                    ) : (
                      <div className="grid h-full place-items-center">
                        <svg viewBox="0 0 48 48" className="h-12 w-12 fill-none stroke-[#77766f] stroke-[1.2]" aria-hidden><path d="M24 5 7 14l17 9 17-9-17-9Z" /><path d="m7 23 17 9 17-9M7 32l17 9 17-9" /></svg>
                      </div>
                    )}
                    <span className="absolute bottom-3 right-3 rounded-[7px] bg-[#1b1b18]/85 px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-white">
                      Open scene
                    </span>
                  </div>
                </button>
                  <div className="p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <h2 className="truncate text-sm font-semibold">{cleanName(project.name)}</h2>
                        <p className="mt-1 text-xs text-[#77766f]">{providerLabel(project.provider)} / {(project.size_kb / 1024).toFixed(1)} MB</p>
                      </div>
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
                <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#77766f]">Scene details</p>
                <h2 className="mt-2 text-xl font-semibold">{cleanName(selected.name)}</h2>
              </div>
              <button type="button" onClick={() => setSelected(null)} className="rounded-[7px] border border-[#c6c3b9] p-2" aria-label="Close">
                <svg viewBox="0 0 24 24" className="h-4 w-4 stroke-current stroke-2" aria-hidden><path d="m6 6 12 12M18 6 6 18" /></svg>
              </button>
            </div>
            <dl className="mt-5 divide-y divide-[#dedcd3] border-y border-[#dedcd3] text-sm">
              <Detail label="Method" value={providerLabel(selected.provider)} />
              <Detail label="File size" value={(selected.size_kb / 1024).toFixed(1) + " MB"} />
              <Detail label="Coverage" value={selected.coverage || (selected.provider === "local_pano" ? "360 look-around" : "nearby view")} />
              <Detail label="Created" value={selected.created_at ? new Date(selected.created_at * 1000).toLocaleString() : "Unknown"} />
            </dl>
            <div className="mt-5 flex gap-3">
              <button type="button" onClick={() => open(selected)} className="flex-1 rounded-[8px] bg-[#1b1b18] px-4 py-3 text-sm font-semibold text-white hover:bg-[#315c4a]">Open scene</button>
              <a href={absoluteUrl(selected.url)} download className="rounded-[8px] border border-[#bbb9af] bg-white px-4 py-3 text-sm font-semibold hover:border-[#77766f]">Save .splat</a>
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
  if (provider === "local_pano") return "Local panorama 360";
  if (provider === "worldlabs") return "World Labs";
  return "Local image to 3D";
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-5 py-3"><dt className="text-[#77766f]">{label}</dt><dd className="text-right font-medium">{value}</dd></div>;
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
