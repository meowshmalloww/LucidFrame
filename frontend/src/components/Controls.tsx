"use client";

import { useState } from "react";
import type { SplatViewerHandle } from "./SplatViewer";
import { ResetIcon, SettingsIcon } from "./Icons";

interface ControlsProps {
  viewerRef: React.RefObject<SplatViewerHandle | null>;
  visible: boolean;
}

export default function Controls({ viewerRef, visible }: ControlsProps) {
  const [fov, setFov] = useState(75);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleResetCamera = () => {
    viewerRef.current?.resetCamera();
  };

  const handleFovChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newFov = parseInt(e.target.value);
    setFov(newFov);
    viewerRef.current?.setFov(newFov);
  };

  if (!visible) return null;

  return (
    <div className="flex items-center gap-3 px-3 py-2 glass rounded-lg text-xs font-mono">
      <button
        onClick={handleResetCamera}
        className="flex items-center gap-1.5 px-2 py-1 text-gray-400 hover:text-accent transition-colors"
      >
        <ResetIcon size={12} />
        <span>RESET</span>
      </button>

      <div className="w-px h-4 bg-white/5" />

      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className={`flex items-center gap-1.5 px-2 py-1 transition-colors ${
          showAdvanced ? "text-accent" : "text-gray-400 hover:text-white"
        }`}
      >
        <SettingsIcon size={12} />
        <span>FOV</span>
      </button>

      {showAdvanced && (
        <div className="flex items-center gap-2 animate-fade-in">
          <input
            type="range"
            min="45"
            max="110"
            value={fov}
            onChange={handleFovChange}
            className="w-24 accent-accent"
          />
          <span className="text-gray-500 w-8">{fov}deg</span>
        </div>
      )}
    </div>
  );
}
