"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  clearWorldLabsSettings,
  getWorldLabsSettings,
  saveWorldLabsSettings,
  type WorldLabsModelOption,
  type WorldLabsSettings,
} from "@/lib/api";

export function WorldLabsSettings() {
  const [settings, setSettings] = useState<WorldLabsSettings | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState<WorldLabsModelOption["id"]>("marble-1.1");
  const [showKey, setShowKey] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getWorldLabsSettings()
      .then((value) => {
        setSettings(value);
        setModel(value.model);
      })
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Could not reach the local backend."));
  }, []);

  const selected = useMemo(
    () => settings?.model_options.find((option) => option.id === model),
    [model, settings],
  );

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setMessage(null);
    if (!apiKey.trim()) {
      setError("Paste a World Labs API key first.");
      return;
    }
    setBusy(true);
    try {
      const next = await saveWorldLabsSettings(apiKey, model);
      setSettings(next);
      setApiKey("");
      setMessage("World Labs is connected for this session.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "World Labs connection failed.");
    } finally {
      setBusy(false);
    }
  };

  const clear = async () => {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      setSettings(await clearWorldLabsSettings());
      setMessage("Session key removed.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not remove the session key.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto bg-[#f3f2ec] text-[#1b1b18]">
      <div className="mx-auto w-full max-w-[860px] px-6 py-9 lg:px-10 lg:py-12">
        <header className="flex items-end justify-between gap-5 border-b border-[#d5d3ca] pb-6">
          <div>
            <h1 className="text-[clamp(2rem,4vw,3.1rem)] font-medium leading-none tracking-[-0.045em]">Settings</h1>
            <p className="mt-3 text-sm text-[#686760]">World Labs connection</p>
          </div>
          <Status connected={settings?.available === true} />
        </header>

        <form onSubmit={save} className="mt-8">
          <section className="grid gap-6 border-b border-[#d5d3ca] pb-8 md:grid-cols-[190px_1fr]">
            <div>
              <h2 className="text-sm font-semibold">API key</h2>
              <p className="mt-2 text-xs leading-5 text-[#77766f]">Used only for Marble requests. It remains in backend memory for this session.</p>
            </div>
            <div>
              <div className="flex gap-2">
                <input
                  id="worldlabs-key"
                  aria-label="World Labs API key"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  type={showKey ? "text" : "password"}
                  autoComplete="off"
                  placeholder={settings?.available ? "Enter a replacement key" : "Paste World Labs API key"}
                  className="min-w-0 flex-1 rounded-[7px] border border-[#bbb9af] bg-[#faf9f5] px-3.5 py-3 font-mono text-sm outline-none focus:border-[#315c4a]"
                />
                <button type="button" onClick={() => setShowKey((value) => !value)} className="rounded-[7px] border border-[#bbb9af] bg-[#faf9f5] px-3 text-xs font-semibold hover:border-[#77766f]">
                  {showKey ? "Hide" : "Show"}
                </button>
              </div>
              <p className="mt-3 text-xs leading-5 text-[#686760]">{connectionText(settings)}</p>
            </div>
          </section>

          <fieldset className="grid gap-6 border-b border-[#d5d3ca] py-8 md:grid-cols-[190px_1fr]">
            <div>
              <legend className="text-sm font-semibold">Marble model</legend>
              <p className="mt-2 text-xs leading-5 text-[#77766f]">Choose the model used for paid generation.</p>
            </div>
            <div>
              <div className="divide-y divide-[#d5d3ca] border-y border-[#d5d3ca]">
                {settings?.model_options.map((option) => (
                  <ModelChoice key={option.id} option={option} selected={model === option.id} onSelect={() => setModel(option.id)} />
                )) || <SettingsSkeleton />}
              </div>
              {selected && <p className="mt-3 text-xs leading-5 text-[#675131]">Estimated charge: <strong>{selected.estimate_label}</strong>. You will confirm before generation starts.</p>}
            </div>
          </fieldset>

          {error && <p role="alert" className="mt-5 text-sm text-[#963b35]">{error}</p>}
          {message && <p className="mt-5 text-sm text-[#315c4a]">{message}</p>}

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button disabled={busy} className="rounded-[7px] bg-[#1c1c19] px-4 py-3 text-sm font-semibold text-white hover:bg-[#315c4a] disabled:bg-[#aaa89f]">
              {busy ? "Checking…" : "Save connection"}
            </button>
            {settings?.key_source === "session" && (
              <button type="button" disabled={busy} onClick={clear} className="rounded-[7px] border border-[#bbb9af] px-4 py-3 text-sm font-semibold hover:border-[#963b35]">
                Remove key
              </button>
            )}
            <a href="https://platform.worldlabs.ai/" target="_blank" rel="noreferrer" className="ml-auto inline-flex items-center gap-2 text-sm font-semibold text-[#315c4a] hover:underline">
              Open World Labs
              <svg viewBox="0 0 24 24" className="h-4 w-4 fill-none stroke-current stroke-2" aria-hidden><path d="M8 16 16 8M9 8h7v7" /></svg>
            </a>
          </div>
          <p className="mt-8 border-t border-[#d5d3ca] pt-5 text-xs text-[#77766f]">Local generation never uses this key or World Labs credits.</p>
        </form>
      </div>
    </div>
  );
}

function connectionText(settings: WorldLabsSettings | null) {
  if (!settings) return "Checking…";
  if (!settings.available) return "Not connected. Local generation is still available.";
  const source = settings.key_source === "environment" ? "backend environment" : "current backend session";
  const balance = settings.credit_balance == null ? "" : " Reported balance: " + settings.credit_balance.toLocaleString() + " credits.";
  return "Connected through the " + source + "." + balance;
}

function Status({ connected }: { connected: boolean }) {
  return (
    <span className={connected ? "rounded-[6px] border border-[#8eaa9a] bg-[#edf2ee] px-2.5 py-1.5 text-xs font-medium text-[#315c4a]" : "rounded-[6px] border border-[#c6c3b9] px-2.5 py-1.5 text-xs font-medium text-[#77766f]"}>
      {connected ? "Connected" : "Not connected"}
    </span>
  );
}

function ModelChoice({ option, selected, onSelect }: { option: WorldLabsModelOption; selected: boolean; onSelect: () => void }) {
  return (
    <button type="button" onClick={onSelect} className={selected ? "flex w-full items-start justify-between gap-4 border-l-2 border-l-[#315c4a] bg-[#e9eee9] px-4 py-3.5 text-left" : "flex w-full items-start justify-between gap-4 border-l-2 border-l-transparent px-4 py-3.5 text-left hover:bg-[#eceae2]"}>
      <span>
        <span className="block text-sm font-semibold">{option.label}</span>
        <span className="mt-1 block text-xs leading-5 text-[#686760]">{option.quality}</span>
      </span>
      <span className="shrink-0 text-right text-[11px] leading-5 text-[#5f5e58]">{option.estimate_label}</span>
    </button>
  );
}

function SettingsSkeleton() {
  return (
    <div aria-busy="true" aria-label="Loading model options">
      {[0, 1].map((item) => (
        <div key={item} className="border-b border-[#dedcd3] px-4 py-4 last:border-b-0">
          <div className="skeleton-shimmer h-3.5 w-32 bg-[#e3e1d8]" />
          <div className="skeleton-shimmer mt-2.5 h-2.5 w-3/5 bg-[#e3e1d8]" />
        </div>
      ))}
    </div>
  );
}
