import { AlertCircle, CircleStop, RotateCcw } from "lucide-react";

import { cn } from "../../lib/cn";
import { presentChatError } from "../../lib/chatErrors";

type Props = {
  error: string;
  onRetry?: () => void;
};

export function AssistantError({ error, onRetry }: Props) {
  const presentation = presentChatError(error);
  const Icon = presentation.stopped ? CircleStop : AlertCircle;

  return (
    <div
      role={presentation.stopped ? "status" : "alert"}
      className={cn(
        "max-w-2xl rounded-xl border px-3.5 py-3",
        presentation.stopped
          ? "border-border bg-muted/45"
          : "border-destructive/20 bg-destructive/5",
      )}
    >
      <div className="flex items-start gap-2.5">
        <Icon
          className={cn(
            "mt-0.5 h-4 w-4 shrink-0",
            presentation.stopped ? "text-muted-foreground" : "text-destructive",
          )}
          strokeWidth={1.9}
        />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-foreground">{presentation.title}</p>
          <p className="mt-1 text-[13px] leading-5 text-muted-foreground">
            {presentation.description}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
            {presentation.retryable && onRetry ? (
              <button
                type="button"
                onClick={onRetry}
                className="inline-flex items-center gap-1.5 text-xs font-medium text-foreground hover:text-primary"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                重新发送
              </button>
            ) : null}
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer select-none hover:text-foreground">
                技术详情
              </summary>
              <pre className="mt-2 max-h-32 max-w-full overflow-auto whitespace-pre-wrap rounded-md bg-code-background px-2.5 py-2 font-mono text-[11px] leading-4">
                {error}
              </pre>
            </details>
          </div>
        </div>
      </div>
    </div>
  );
}
