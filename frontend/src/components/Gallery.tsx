"use client";

import { useEffect, useState } from "react";
import { getGallery, type GallerySplat, SPLAT_URL_BASE } from "@/lib/api";
import { CubeIcon, LayersIcon } from "./Icons";

interface GalleryProps {
  onSelect: (splatUrl: string) => void;
}

export default function Gallery({ onSelect }: GalleryProps) {
  const [splats, setSplats] = useState<GallerySplat[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getGallery().then((s) => {
      setSplats(s);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-gray-600 font-mono">
        <LayersIcon size={12} className="animate-subtle-pulse" />
        <span>Loading gallery...</span>
      </div>
    );
  }

  if (splats.length === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-xs font-mono text-gray-600 tracking-wider">
        <LayersIcon size={12} />
        <span>PRE-BAKED DREAMS</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {splats.map((splat) => (
          <button
            key={splat.name}
            onClick={() => onSelect(splat.url)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg border border-white/5
                     bg-panel/40 hover:bg-accent/10 hover:border-accent/20
                     transition-all group"
          >
            <CubeIcon size={14} className="text-gray-500 group-hover:text-accent transition-colors" />
            <div className="flex flex-col items-start">
              <span className="text-xs text-gray-400 group-hover:text-white transition-colors">
                {splat.name}
              </span>
              <span className="text-[10px] text-gray-600 font-mono">
                {splat.size_kb > 1024
                  ? `${(splat.size_kb / 1024).toFixed(1)} MB`
                  : `${splat.size_kb} KB`}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
