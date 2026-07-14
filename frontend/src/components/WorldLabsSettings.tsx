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
      setMessage("Connection verified. The key is held only by this running backend process.");
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
      <div className="mx-auto w-full max-w-6xl px-6 py-9 lg:px-10 lg:py-12">
        <header className="border-b border-[#d5d3ca] pb-7">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#77766f]">Settings</p>
          <h1 className="mt-3 text-4xl font-medium tracking-[-0.045em]">Connections and generation cost</h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-[#686760]">
            Local generation does not need an API key. World Labs is optional and every generation uses paid credits.
          </p>
        </header>

        <div className="mt-8 grid gap-7 lg:grid-cols-[minmax(0,1fr)_310px]">
          <form onSubmit={save} className="rounded-[12px] border border-[#d1cfc5] bg-[#faf9f5] p-6">
            <div className="flex items-start justify-between gap-4 border-b border-[#dedcd3] pb-5">
              <div>
                <h2 className="text-xl font-semibold tracking-[-0.025em]">World Labs API</h2>
                <p className="mt-2 max-w-xl text-sm leading-6 text-[#686760]">
                  The browser sends the key to your local FastAPI process. It is not placed in localStorage and is forgotten when the process stops.
                </p>
              </div>
              <Status connected={settings?.available === true} />
            </div>

            <label htmlFor="worldlabs-key" className="mt-6 block text-xs font-semibold text-[#4f4e48]">API key</label>
            <div className="mt-2 flex gap-2">
              <input
                id="worldlabs-key"
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                type={showKey ? "text" : "password"}
                autoComplete="off"
                placeholder={settings?.available ? "Enter a replacement key" : "Paste World Labs API key"}
                className="min-w-0 flex-1 rounded-[8px] border border-[#bbb9af] bg-white px-3.5 py-3 font-mono text-sm outline-none focus:border-[#315c4a]"
              />
              <button
                type="button"
                onClick={() => setShowKey((value) => !value)}
                className="rounded-[8px] border border-[#bbb9af] bg-white px-3 text-xs font-semibold hover:border-[#77766f]"
              >
                {showKey ? "Hide" : "Show"}
              </button>
            </div>

            <fieldset className="mt-7">
              <legend className="text-xs font-semibold text-[#4f4e48]">Default model</legend>
              <div className="mt-3 grid gap-2">
                {settings?.model_options.map((option) => (
                  <ModelChoice
                    key={option.id}
                    option={option}
                    selected={model === option.id}
                    onSelect={() => setModel(option.id)}
                  />
                )) || <div className="h-24 rounded-[9px] bg-[#eeece4]" />}
              </div>
            </fieldset>

            {selected && (
              <div className="mt-5 border-l-2 border-[#9a6b2d] bg-[#f4efe3] px-4 py-3 text-xs leading-5 text-[#675131]">
                Estimated charge: <strong>{selected.estimate_label}</strong>. LucidFrame asks again before starting a paid request.
              </div>
            )}

            {error && <p role="alert" className="mt-4 text-sm text-[#963b35]">{error}</p>}
            {message && <p className="mt-4 text-sm text-[#315c4a]">{message}</p>}

            <div className="mt-6 flex flex-wrap gap-3">
              <button
                disabled={busy}
                className="rounded-[8px] bg-[#1c1c19] px-4 py-3 text-sm font-semibold text-white hover:bg-[#315c4a] disabled:bg-[#aaa89f]"
              >
                {busy ? "Checking..." : "Verify and keep for session"}
              </button>
              {settings?.key_source === "session" && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={clear}
                  className="rounded-[8px] border border-[#bbb9af] bg-white px-4 py-3 text-sm font-semibold hover:border-[#963b35]"
                >
                  Remove key
                </button>
              )}
            </div>
          </form>

          <aside className="space-y-4">
            <Info title="Connection" text={connectionText(settings)} />
            <Info title="Local modes" text="Image to 3D and Panorama to 360 run on your machine and do not consume World Labs credits." />
            <Info title="Privacy boundary" text="A World Labs image leaves your computer only after you select its mode and confirm the paid request." />
            <a
              href="https://platform.worldlabs.ai/"
              target="_blank"
              rel="noreferrer"
              className="flex items-center justify-between rounded-[10px] border border-[#315c4a] bg-[#edf2ee] p-4 text-sm font-semibold text-[#274c3d] hover:bg-[#e3ece6]"
            >
              World Labs platform
              <svg viewBox="0 0 24 24" className="h-4 w-4 fill-none stroke-current stroke-2" aria-hidden><path d="M8 16 16 8M9 8h7v7" /></svg>
            </a>
          </aside>
        </div>
      </div>
    </div>
  );
}

function connectionText(settings: WorldLabsSettings | null) {
  if (!settings?.available) return "Not connected. This does not affect either local reconstruction mode.";
  const source = settings.key_source === "environment" ? "backend environment" : "current backend session";
  const balance = settings.credit_balance == null ? "" : " Reported balance: " + settings.credit_balance.toLocaleString() + " credits.";
  return "Connected through the " + source + "." + balance;
}

function Status({ connected }: { connected: boolean }) {
  return (
    <span className={connected ? "rounded-[7px] border border-[#8eaa9a] bg-[#edf2ee] px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#315c4a]" : "rounded-[7px] border border-[#c6c3b9] px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#77766f]"}>
      {connected ? "Connected" : "Not connected"}
    </span>
  );
}

function ModelChoice({
  option,
  selected,
  onSelect,
}: {
  option: WorldLabsModelOption;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={selected ? "flex items-start justify-between gap-4 rounded-[9px] border border-[#315c4a] bg-[#edf2ee] p-4 text-left" : "flex items-start justify-between gap-4 rounded-[9px] border border-[#dedcd3] bg-white p-4 text-left hover:border-[#aaa89e]"}
    >
      <span>
        <span className="block text-sm font-semibold">{option.label}</span>
        <span className="mt-1 block text-xs leading-5 text-[#686760]">{option.quality}</span>
      </span>
      <span className="shrink-0 text-right text-[11px] leading-5 text-[#5f5e58]">{option.estimate_label}</span>
    </button>
  );
}

function Info({ title, text }: { title: string; text: string }) {
  return (
    <div className="rounded-[10px] border border-[#d1cfc5] bg-[#faf9f5] p-4">
      <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-[#77766f]">{title}</h2>
      <p className="mt-2 text-sm leading-6 text-[#686760]">{text}</p>
    </div>
  );
}
