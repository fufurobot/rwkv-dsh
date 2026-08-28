import { ArrowUp, Folder, Square } from "lucide-react";
import { useEffect, useRef } from "react";

type Props = {
  draft: string;
  busy: boolean;
  workspaceName: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
};

export function Composer({ draft, busy, workspaceName, onChange, onSend, onStop }: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 160)}px`;
  }, [draft]);

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 px-4 pb-4">
      <form
        className="pointer-events-auto mx-auto max-w-3xl"
        onSubmit={(event) => {
          event.preventDefault();
          if (!busy) onSend();
        }}
      >
        <div className="rounded-[22px] border border-border bg-composer-input shadow-[0_8px_24px_rgba(0,0,0,0.12)]">
          <div className="px-4 pt-3 pb-1">
            <textarea
              ref={textareaRef}
              value={draft}
              rows={1}
              placeholder={busy ? "正在生成…" : "问问 RWKV…"}
              disabled={busy}
              onChange={(event) => onChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  if (!busy) onSend();
                }
              }}
              className="max-h-40 min-h-8 w-full resize-none overflow-y-auto border-0 bg-transparent px-1 py-2 font-sans text-sm leading-6 text-foreground outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>
          <div className="flex items-center justify-between gap-3 px-3 pb-3">
            <div className="flex min-w-0 items-center gap-1.5 text-muted-foreground">
              <Folder className="h-3.5 w-3.5 shrink-0" strokeWidth={1.7} />
              <span className="truncate text-xs">{workspaceName || "未选择工作区"}</span>
            </div>
            {busy ? (
              <button
                type="button"
                onClick={onStop}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-foreground text-background"
                aria-label="停止"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!draft.trim()}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-foreground text-background disabled:opacity-35"
                aria-label="发送"
              >
                <ArrowUp className="h-4 w-4" strokeWidth={2.2} />
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}
