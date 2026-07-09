"use client";

import { useState } from "react";

type SettingsCategory = "general" | "providers" | "pipeline" | "viewer" | "advanced" | "about";

const categories: { id: SettingsCategory; label: string; icon: React.ReactNode }[] = [
  {
    id: "general",
    label: "General",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v6m0 6v6m11-7h-6m-6 0H1m16.5-5.5l-4.24 4.24m-4.52 4.52L4.5 19.5M19.5 19.5l-4.24-4.24m-4.52-4.52L4.5 4.5" />
      </svg>
    ),
  },
  {
    id: "providers",
    label: "Providers",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="2" width="20" height="8" rx="2" />
        <rect x="2" y="14" width="20" height="8" rx="2" />
        <line x1="6" y1="6" x2="6.01" y2="6" />
        <line x1="6" y1="18" x2="6.01" y2="18" />
      </svg>
    ),
  },
  {
    id: "pipeline",
    label: "Pipeline",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
  },
  {
    id: "viewer",
    label: "Viewer",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
  },
  {
    id: "advanced",
    label: "Advanced",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
      </svg>
    ),
  },
  {
    id: "about",
    label: "About",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
    ),
  },
];

export default function SettingsPage() {
  const [active, setActive] = useState<SettingsCategory>("providers");
  const [showApiKey, setShowApiKey] = useState<Record<string, boolean>>({});
  const [saved, setSaved] = useState(false);

  const toggleKey = (id: string) =>
    setShowApiKey((prev) => ({ ...prev, [id]: !prev[id] }));

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left: categories */}
      <div className="w-[200px] shrink-0 border-r border-white/[0.06] bg-surface py-4">
        <div className="px-4 pb-4">
          <h1 className="text-[18px] font-semibold tracking-tight text-white">Settings</h1>
        </div>
        <nav className="flex flex-col gap-px px-2">
          {categories.map((cat) => (
            <button
              key={cat.id}
              onClick={() => setActive(cat.id)}
              className={`flex items-center gap-2.5 px-3 py-2 text-left text-[13px] font-medium transition-colors ${
                active === cat.id
                  ? "bg-white/[0.06] text-white"
                  : "text-zinc-500 hover:bg-white/[0.02] hover:text-zinc-300"
              }`}
            >
              {cat.icon}
              {cat.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Right: settings content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Scrollable settings */}
        <div className="flex-1 overflow-y-auto thin-scroll">
          <div className="mx-auto max-w-[640px] px-8 py-10">
            {active === "general" && <GeneralSettings />}
            {active === "providers" && (
              <ProvidersSettings showApiKey={showApiKey} toggleKey={toggleKey} />
            )}
            {active === "pipeline" && <PipelineSettings />}
            {active === "viewer" && <ViewerSettings />}
            {active === "advanced" && <AdvancedSettings />}
            {active === "about" && <AboutSettings />}

            {/* Save / Restore */}
            {active !== "about" && (
              <div className="mt-10 flex items-center justify-between border-t border-white/[0.06] pt-6">
                <button className="text-[13px] font-medium text-zinc-500 transition-colors hover:text-zinc-300">
                  Restore Defaults
                </button>
                <div className="flex items-center gap-3">
                  {saved && (
                    <span className="text-[13px] text-success animate-fade-in">
                      Saved
                    </span>
                  )}
                  <button
                    onClick={handleSave}
                    className="rounded-btn bg-white px-4 py-2 text-[13px] font-semibold text-black transition-colors hover:bg-zinc-200"
                  >
                    Save Changes
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Section helpers ──────────────────────────────────────────────────────────

function SectionTitle({ title, desc }: { title: string; desc?: string }) {
  return (
    <div className="mb-6">
      <h2 className="text-[16px] font-semibold tracking-tight text-white">{title}</h2>
      {desc && <p className="mt-1 text-[13px] text-zinc-500">{desc}</p>}
    </div>
  );
}

function SettingsRow({ label, desc, value, children }: { label: string; desc?: string; value?: string; children?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-white/[0.04] py-3.5 last:border-0">
      <div className="flex flex-col">
        <span className="text-[14px] font-medium text-zinc-200">{label}</span>
        {desc && <span className="mt-0.5 text-[12px] text-zinc-600">{desc}</span>}
      </div>
      <div>{children ?? <span className="text-[13px] font-mono text-zinc-400">{value}</span>}</div>
    </div>
  );
}

function Toggle({ defaultOn = false }: { defaultOn?: boolean }) {
  const [on, setOn] = useState(defaultOn);
  return (
    <button
      onClick={() => setOn(!on)}
      className={`relative h-[22px] w-[38px] rounded-[4px] border transition-colors ${
        on
          ? "border-white bg-white"
          : "border-white/[0.12] bg-transparent"
      }`}
    >
      <span
        className={`absolute top-[2px] h-[16px] w-[16px] rounded-[2px] transition-all ${
          on ? "left-[18px] bg-black" : "left-[2px] bg-white/40"
        }`}
      />
    </button>
  );
}

function Dropdown({ options, defaultValue }: { options: string[]; defaultValue?: string }) {
  return (
    <select
      defaultValue={defaultValue}
      className="rounded-btn border border-white/[0.08] bg-elevated px-3 py-1.5 text-[13px] font-medium text-zinc-200 focus:border-white/20 focus:outline-none"
    >
      {options.map((opt) => (
        <option key={opt} value={opt} className="bg-elevated text-white">
          {opt}
        </option>
      ))}
    </select>
  );
}

function ApiKeyInput({
  id,
  placeholder,
  showApiKey,
  toggleKey,
}: {
  id: string;
  placeholder: string;
  showApiKey: Record<string, boolean>;
  toggleKey: (id: string) => void;
}) {
  const visible = showApiKey[id] ?? false;
  return (
    <div className="flex items-center gap-2">
      <input
        type={visible ? "text" : "password"}
        placeholder={placeholder}
        className="w-[200px] rounded-btn border border-white/[0.08] bg-elevated px-3 py-1.5 text-[13px] font-mono text-zinc-200 placeholder:text-zinc-600 focus:border-white/20 focus:outline-none"
      />
      <button
        onClick={() => toggleKey(id)}
        className="flex h-8 w-8 items-center justify-center text-zinc-500 transition-colors hover:text-zinc-300"
      >
        {visible ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
            <line x1="1" y1="1" x2="23" y2="23" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        )}
      </button>
    </div>
  );
}

function StatusBadge({ status }: { status: "connected" | "not-configured" }) {
  return (
    <span
      className={`flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-medium ${
        status === "connected"
          ? "text-success"
          : "text-zinc-600"
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${status === "connected" ? "bg-success" : "bg-zinc-600"}`} />
      {status === "connected" ? "Connected" : "Not configured"}
    </span>
  );
}

// ── Settings sections ────────────────────────────────────────────────────────

function GeneralSettings() {
  return (
    <div>
      <SectionTitle title="General" desc="Manage application preferences." />
      <SettingsRow label="Auto Save" desc="Automatically save generated projects">
        <Toggle defaultOn />
      </SettingsRow>
      <SettingsRow label="Recent Projects" desc="Number of projects to show on home">
        <Dropdown options={["4", "8", "12", "20"]} defaultValue="4" />
      </SettingsRow>
      <SettingsRow label="Notifications" desc="Show notifications when generation completes">
        <Toggle defaultOn />
      </SettingsRow>
      <SettingsRow label="Hardware Acceleration" desc="Use GPU for rendering">
        <Toggle defaultOn />
      </SettingsRow>
    </div>
  );
}

function ProvidersSettings({
  showApiKey,
  toggleKey,
}: {
  showApiKey: Record<string, boolean>;
  toggleKey: (id: string) => void;
}) {
  return (
    <div>
      <SectionTitle title="Providers" desc="Configure AI model providers for VLM and LLM stages." />

      {/* VLM */}
      <div className="mb-8">
        <h3 className="mb-1 text-[12px] font-semibold uppercase tracking-wider text-zinc-600">VLM Provider</h3>
        <SettingsRow label="Provider" desc="Vision language model for image analysis">
          <Dropdown
            options={["gemini", "groq", "openrouter", "nim", "mistral", "openai", "anthropic"]}
            defaultValue="gemini"
          />
        </SettingsRow>
        <SettingsRow label="API Key">
          <ApiKeyInput id="vlm-key" placeholder="Enter API key" showApiKey={showApiKey} toggleKey={toggleKey} />
        </SettingsRow>
        <div className="py-2">
          <StatusBadge status="not-configured" />
        </div>
      </div>

      {/* LLM */}
      <div className="mb-8">
        <h3 className="mb-1 text-[12px] font-semibold uppercase tracking-wider text-zinc-600">LLM Provider</h3>
        <SettingsRow label="Provider" desc="Language model for dream narrative generation">
          <Dropdown
            options={["groq", "gemini", "cerebras", "nim", "mistral", "openrouter", "openai", "anthropic"]}
            defaultValue="groq"
          />
        </SettingsRow>
        <SettingsRow label="API Key">
          <ApiKeyInput id="llm-key" placeholder="Enter API key" showApiKey={showApiKey} toggleKey={toggleKey} />
        </SettingsRow>
        <div className="py-2">
          <StatusBadge status="not-configured" />
        </div>
      </div>

      {/* Image Generation */}
      <div className="mb-8">
        <h3 className="mb-1 text-[12px] font-semibold uppercase tracking-wider text-zinc-600">Image Generation</h3>
        <SettingsRow label="Provider" desc="Optional image enhancement before pipeline">
          <Dropdown
            options={["pollinations", "cloudflare", "huggingface", "pixazo"]}
            defaultValue="pollinations"
          />
        </SettingsRow>
        <SettingsRow label="API Key" desc="Not required for Pollinations">
          <ApiKeyInput id="imgen-key" placeholder="Enter API key (if needed)" showApiKey={showApiKey} toggleKey={toggleKey} />
        </SettingsRow>
      </div>
    </div>
  );
}

function PipelineSettings() {
  return (
    <div>
      <SectionTitle title="Pipeline" desc="Configure the 3D generation pipeline." />
      <SettingsRow label="Reconstruction Model" desc="Model used for 3D Gaussian reconstruction">
        <Dropdown options={["lgm", "depth_gsplat", "splatter_image"]} defaultValue="lgm" />
      </SettingsRow>
      <SettingsRow label="Artifact Fix" desc="Difix3D+ post-processing mode">
        <Dropdown options={["structural", "full", "skip"]} defaultValue="structural" />
      </SettingsRow>
      <SettingsRow label="Stitching" desc="Expand hallucinated space with additional views">
        <Toggle />
      </SettingsRow>
      <SettingsRow label="Multi-view Steps" desc="Diffusion steps (higher = better, slower)">
        <div className="flex items-center gap-2">
          <input type="range" min="25" max="150" defaultValue="75" className="w-32 accent-white" />
          <span className="w-8 text-right text-[13px] font-mono text-zinc-300">75</span>
        </div>
      </SettingsRow>
      <SettingsRow label="IP-Adapter" desc="Style locking for multi-view consistency">
        <Toggle defaultOn />
      </SettingsRow>
      <SettingsRow label="Server URL" desc="Backend API endpoint">
        <input
          type="text"
          defaultValue="http://localhost:8000"
          className="w-[200px] rounded-btn border border-white/[0.08] bg-elevated px-3 py-1.5 text-[13px] font-mono text-zinc-200 focus:border-white/20 focus:outline-none"
        />
      </SettingsRow>
    </div>
  );
}

function ViewerSettings() {
  return (
    <div>
      <SectionTitle title="Viewer" desc="Customize the 3D viewport." />
      <SettingsRow label="Default Navigation" desc="Camera navigation mode">
        <Dropdown options={["Orbit", "FPS"]} defaultValue="Orbit" />
      </SettingsRow>
      <SettingsRow label="Default FOV" desc="Field of view in degrees">
        <div className="flex items-center gap-2">
          <input type="range" min="30" max="100" defaultValue="60" className="w-32 accent-white" />
          <span className="w-8 text-right text-[13px] font-mono text-zinc-300">60°</span>
        </div>
      </SettingsRow>
      <SettingsRow label="Auto Exposure" desc="Automatically adjust exposure">
        <Toggle defaultOn />
      </SettingsRow>
      <SettingsRow label="Theme" desc="Viewer background theme">
        <Dropdown options={["Dark", "Darker", "Black"]} defaultValue="Dark" />
      </SettingsRow>
      <SettingsRow label="Performance" desc="Render quality preset">
        <Dropdown options={["Low", "Medium", "High", "Ultra"]} defaultValue="High" />
      </SettingsRow>
    </div>
  );
}

function AdvancedSettings() {
  return (
    <div>
      <SectionTitle title="Advanced" desc="Advanced configuration options." />
      <SettingsRow label="Flash Attention 2" desc="Enable memory-efficient attention">
        <Toggle defaultOn />
      </SettingsRow>
      <SettingsRow label="CPU Offloading" desc="Move unused model layers to CPU">
        <Toggle defaultOn />
      </SettingsRow>
      <SettingsRow label="Debug Logging" desc="Verbose pipeline logs">
        <Toggle />
      </SettingsRow>
      <SettingsRow label="WebSocket Port" desc="Port for pipeline event streaming">
        <input
          type="text"
          defaultValue="8000"
          className="w-[80px] rounded-btn border border-white/[0.08] bg-elevated px-3 py-1.5 text-[13px] font-mono text-zinc-200 focus:border-white/20 focus:outline-none"
        />
      </SettingsRow>
    </div>
  );
}

function AboutSettings() {
  return (
    <div>
      <SectionTitle title="About" desc="Application information." />
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-card bg-white">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-black">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h2 className="text-[18px] font-semibold tracking-tight text-white">LucidFrame</h2>
            <p className="text-[13px] text-zinc-500">Version 1.0.0</p>
          </div>
        </div>
        <div className="rounded-card border border-white/[0.04] bg-surface p-4">
          <p className="text-[13px] leading-relaxed text-zinc-400">
            LucidFrame transforms a single image into an explorable 3D Gaussian Splat scene.
            It combines VLM analysis, LLM dream narrative, multi-view diffusion, and 3D
            reconstruction into a seamless pipeline.
          </p>
        </div>
        <div className="flex flex-col">
          <SettingsRow label="Backend" value="FastAPI + uvicorn" />
          <SettingsRow label="Frontend" value="Next.js 15 + Tailwind CSS" />
          <SettingsRow label="3D Engine" value="gsplat" />
        </div>
      </div>
    </div>
  );
}
