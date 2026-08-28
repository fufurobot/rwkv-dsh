import { Moon, Sun, X } from "lucide-react";

import type { Settings } from "../types";

type Props = {
  settings: Settings;
  error: string;
  dark: boolean;
  onChange: (next: Settings) => void;
  onPersist: (next: Settings) => void;
  onChooseWorkspace: () => void;
  onToggleDark: () => void;
  onClose: () => void;
};

export function SettingsSheet({
  settings,
  error,
  dark,
  onChange,
  onPersist,
  onChooseWorkspace,
  onToggleDark,
  onClose,
}: Props) {
  const field =
    "mt-1.5 h-9 w-full rounded-lg border border-input bg-background px-3 text-sm text-foreground outline-none focus:ring-1 focus:ring-ring";

  return (
    <div className="absolute inset-0 z-20 flex justify-end bg-foreground/10" onClick={onClose}>
      <aside
        className="flex h-full w-88 max-w-full flex-col border-l border-border bg-card text-card-foreground shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3">
          <h2 className="text-sm font-medium">设置</h2>
          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 pb-6">
          <label className="block text-xs text-muted-foreground">
            Base URL
            <input
              className={field}
              value={settings.baseUrl}
              onChange={(event) => onChange({ ...settings, baseUrl: event.target.value })}
              onBlur={(event) => onPersist({ ...settings, baseUrl: event.target.value })}
            />
          </label>
          <label className="block text-xs text-muted-foreground">
            Password
            <input
              type="password"
              className={field}
              value={settings.password}
              onChange={(event) => onChange({ ...settings, password: event.target.value })}
              onBlur={(event) => onPersist({ ...settings, password: event.target.value })}
            />
          </label>
          <label className="block text-xs text-muted-foreground">
            Model
            <input
              className={field}
              value={settings.model}
              onChange={(event) => onChange({ ...settings, model: event.target.value })}
              onBlur={(event) => onPersist({ ...settings, model: event.target.value })}
            />
          </label>
          <label className="block text-xs text-muted-foreground">
            Workspace
            <div className="mt-1.5 flex gap-2">
              <input className={field + " mt-0"} value={settings.workspace} readOnly />
              <button
                type="button"
                onClick={onChooseWorkspace}
                className="h-9 shrink-0 rounded-lg border border-input px-3 text-sm text-foreground hover:bg-accent"
              >
                选择
              </button>
            </div>
          </label>
          <label className="block text-xs text-muted-foreground">
            Max tokens
            <input
              type="number"
              min={256}
              max={16384}
              step={256}
              className={field}
              value={settings.maxTokens}
              onChange={(event) => {
                const maxTokens = Number(event.target.value);
                onChange({ ...settings, maxTokens: Number.isFinite(maxTokens) ? maxTokens : settings.maxTokens });
              }}
              onBlur={(event) => {
                const parsed = Number(event.target.value);
                const maxTokens = Math.min(16384, Math.max(256, Number.isFinite(parsed) ? Math.round(parsed) : 8192));
                const next = { ...settings, maxTokens };
                onChange(next);
                onPersist(next);
              }}
            />
          </label>
          <label className="block text-xs text-muted-foreground">
            Thinking
            <div className={`${field} flex items-center text-foreground`}>
              off（已按后端协议强制关闭）
            </div>
          </label>
          <button
            type="button"
            onClick={onToggleDark}
            className="flex h-9 w-full items-center gap-2 rounded-lg border border-input px-3 text-sm text-foreground hover:bg-accent"
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            {dark ? "浅色外观" : "深色外观"}
          </button>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <p className="text-xs leading-5 text-muted-foreground">
            推理后端为 rwkv_lightning。Base URL 可自行填写，默认
            http://192.168.0.115:8030/v1。GET /v1/models 的 owned_by 含 cuda 走
            /v1/batch/completions，否则走 /v1/chat/completions。所有 contents
            均使用完整的 Assistant: &lt;think&gt;&lt;/think&gt; 关闭深度思考。
          </p>
        </div>
      </aside>
    </div>
  );
}
