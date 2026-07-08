"use client";

import { useCallback, useRef, useState } from "react";
import { MoonIcon, ImageIcon } from "./Icons";

interface UploadDropzoneProps {
  onImageSelected: (image: File) => void;
  disabled?: boolean;
}

export default function UploadDropzone({ onImageSelected, disabled }: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      const url = URL.createObjectURL(file);
      setPreview(url);
      onImageSelected(file);
    },
    [onImageSelected],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile, disabled],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div
      className={`
        relative flex flex-col items-center justify-center
        min-h-[420px] rounded-2xl border-2 border-dashed
        transition-all duration-300 cursor-pointer
        ${isDragging ? "border-accent bg-accent/5 scale-[1.01]" : "border-secondary/40 hover:border-accent/40"}
        ${disabled ? "opacity-40 pointer-events-none" : ""}
        dropzone-glow
      `}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.heic,.heif,.avif,.webp,.tiff,.tif,.bmp,.gif,.svg"
        className="hidden"
        onChange={handleChange}
      />

      {preview ? (
        <div className="flex flex-col items-center gap-6 p-8">
          {/* eslint-disable-next-line @next/next/no-img-element -- blob URL preview can't use next/image */}
          <img
            src={preview}
            alt="Uploaded preview"
            className="max-h-[300px] rounded-xl object-contain"
            style={{ filter: "drop-shadow(0 8px 24px rgba(0,0,0,0.5))" }}
          />
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <ImageIcon size={14} />
            <span>Click to change image</span>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-6 p-8 text-center">
          <div className="relative">
            <MoonIcon size={56} className="text-accent/40 animate-pulse-slow" />
            <div className="absolute inset-0 blur-xl bg-accent/10 rounded-full" />
          </div>
          <div className="space-y-2">
            <h2 className="text-2xl font-light tracking-wide text-gray-200">
              Drop any image
            </h2>
            <p className="text-base font-light text-accent/70 tracking-wide">
              Step into the dream
            </p>
          </div>
          <p className="text-xs text-gray-600 max-w-xs leading-relaxed">
            Photos, paintings, artifacts, screenshots — anything visual.
            The AI will hallucinate the unseen world around it.
          </p>
        </div>
      )}
    </div>
  );
}
