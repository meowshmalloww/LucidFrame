"use client";

import { useCallback, useRef, useState } from "react";

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
        min-h-[400px] rounded-2xl border-2 border-dashed
        transition-all duration-300 cursor-pointer
        ${isDragging ? "border-accent bg-accent/5 scale-[1.02]" : "border-secondary/50 hover:border-accent/50"}
        ${disabled ? "opacity-50 pointer-events-none" : ""}
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
        accept="image/*"
        className="hidden"
        onChange={handleChange}
      />

      {preview ? (
        <div className="flex flex-col items-center gap-6 p-8">
          {/* eslint-disable-next-line @next/next/no-img-element -- blob URL preview can't use next/image */}
          <img
            src={preview}
            alt="Uploaded preview"
            className="max-h-[300px] rounded-lg shadow-2xl object-contain"
          />
          <p className="text-sm text-gray-400">Click to change image</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4 p-8 text-center">
          <div className="text-6xl mb-2 animate-pulse-slow">🌙</div>
          <h2 className="text-2xl font-light tracking-wide text-gray-300">
            Drop any image
          </h2>
          <p className="text-lg font-light text-accent/80">
            Step into the dream
          </p>
          <p className="text-xs text-gray-500 mt-4 max-w-xs">
            Photos, paintings, historical artifacts, screenshots — anything visual.
            The AI will hallucinate the unseen world around it.
          </p>
        </div>
      )}
    </div>
  );
}
