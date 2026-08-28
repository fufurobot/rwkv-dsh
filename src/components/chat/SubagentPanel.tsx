import { Bot, Check, Loader2, X } from "lucide-react";

import { cn } from "../../lib/cn";
import type { WorkerToolTrace, WorkerTrace } from "../../types";

type Props = {
  workers: WorkerTrace[];
  running: boolean;
  selected: number;
  workspaceName?: string;
  onSelect: (index: number) => void;
};

export function SubagentPanel({ workers, running, selected, workspaceName, onSelect }: Props) {
  const active = workers[selected] ?? workers[0];
  const timeline = active ? sharedTimeline([active]) : [];
  const shared = sharedTimeline(workers);
  const latest = shared[shared.length - 1];
  const completed = workers.filter((worker) => worker.status === "done").length;
  const failed = workers.filter((worker) => worker.status === "error").length;

  return (
    <aside className="flex h-full w-88 shrink-0 flex-col border-l border-sidebar-border/70 bg-sidebar text-sidebar-foreground">
      <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-sidebar-border/60 px-4">
        <div className="flex min-w-0 items-center gap-2">
          <Bot className="h-4 w-4 shrink-0 text-tool-foreground" strokeWidth={1.8} />
          <h2 className="truncate text-sm font-medium">
            子任务{workspaceName ? ` · ${workspaceName}` : ""}
          </h2>
        </div>
        <span className="shrink-0 text-[11px] text-muted-foreground">
          {running
            ? `第 ${latest?.round || 1} 轮 · ${workers.length} 路`
            : failed > 0
              ? `${latest?.round || 0} 轮 · ${completed}/${workers.length} 路完成 · ${failed} 路失败`
              : `${latest?.round || 0} 轮 · ${workers.length} 路完成`}
        </span>
      </div>

      {workers.length === 0 ? (
        <p className="px-4 py-6 text-sm text-muted-foreground">正在拆解子任务…</p>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          <ol className="shrink-0 border-b border-sidebar-border/60 py-1.5">
            {workers.map((worker, index) => (
              <li key={worker.index}>
                <button
                  type="button"
                  onClick={() => onSelect(index)}
                  className={cn(
                    "flex w-full items-start gap-1.5 px-3 py-1.5 text-left text-[12px] leading-4",
                    index === selected
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
                  )}
                >
                  <StatusIcon status={worker.status} />
                  <span className="line-clamp-2">{worker.goal || `子任务 ${index + 1}`}</span>
                </button>
              </li>
            ))}
          </ol>

          <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
            {timeline.length === 0 ? (
              <p className="text-[12px] text-muted-foreground">正在工作…</p>
            ) : (
              <ol className="space-y-4">
                {timeline.map((group) => {
                  const failed = group.items.some((trace) => trace.ok === false);
                  const live =
                    !failed && group.items.some((trace) => trace.ok === undefined);
                  return (
                    <li key={group.round || group.label}>
                      <p className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                        {live ? (
                          <Loader2 className="h-3 w-3 shrink-0 animate-spin" />
                        ) : failed ? (
                          <X className="h-3 w-3 shrink-0 text-destructive" />
                        ) : (
                          <Check className="h-3 w-3 shrink-0 text-success" />
                        )}
                        {group.label}
                      </p>
                      {group.items.length > 0 ? (
                        <ul className="space-y-1.5 rounded-lg bg-code-background px-2.5 py-2">
                          {group.items.map((trace, index) => {
                            return (
                              <li
                                key={`${group.round}-${index}-${trace.name}`}
                                className="flex items-start gap-1.5 text-[12px] leading-4 text-muted-foreground"
                              >
                                {trace.ok === true ? (
                                  <Check className="mt-0.5 h-3 w-3 shrink-0 text-success" />
                                ) : trace.ok === undefined ? (
                                  <Loader2 className="mt-0.5 h-3 w-3 shrink-0 animate-spin" />
                                ) : (
                                  <X className="mt-0.5 h-3 w-3 shrink-0 text-destructive" />
                                )}
                                <TraceContent trace={trace} />
                              </li>
                            );
                          })}
                        </ul>
                      ) : null}
                    </li>
                  );
                })}
              </ol>
            )}
            {active?.output ? (
              <div className="mt-4 rounded-lg border border-sidebar-border/70 bg-background/70 px-3 py-2.5">
                <p className="mb-1.5 text-[11px] font-medium text-muted-foreground">
                  {active.status === "running" ? "累计总结" : "子任务结论"}
                </p>
                <p className="whitespace-pre-wrap text-[12px] leading-5 text-foreground/85">
                  {active.output}
                </p>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </aside>
  );
}

function sharedTimeline(workers: WorkerTrace[]) {
  const rounds = new Map<number, { round: number; label: string; items: WorkerToolTrace[] }>();

  for (const worker of workers) {
    for (const trace of worker.traces ?? []) {
      const round = trace.round ?? 0;
      if (trace.name === "round") {
        const current = rounds.get(round) ?? { round, label: "", items: [] };
        if (!current.label) current.label = trace.summary || `第 ${round} 轮`;
        rounds.set(round, current);
        continue;
      }
      const current = rounds.get(round) ?? {
        round,
        label: round ? `第 ${round} 轮` : "准备",
        items: [],
      };
      current.items.push(trace);
      rounds.set(round, current);
    }
  }

  return [...rounds.values()].sort((left, right) => left.round - right.round);
}

function TraceContent({ trace }: { trace: WorkerToolTrace }) {
  const summary = trace.name === "notes" || trace.name === "agent_summary";
  const label =
    trace.name === "read_file"
      ? "读取"
      : trace.name === "read_retry"
        ? "同游标重试"
      : trace.name === "read_chunk"
        ? "读取分块"
        : trace.name === "notes"
          ? "分段总结"
          : trace.name === "summary_retry"
            ? "缩小分块重试"
          : trace.name === "agent_summary_fallback"
            ? "保留完整证据账本"
          : trace.name === "round_summary"
            ? "轮次总结"
            : trace.name === "agent_summary"
              ? "子任务总结"
              : trace.name === "list_files"
                ? "扫描文件"
                : trace.name;

  return (
    <span className="min-w-0 flex-1 wrap-anywhere">
      <span className="font-medium text-foreground/80">{label}</span>
      {trace.summary ? (
        summary ? (
          <span className="mt-1 block whitespace-pre-wrap leading-5 text-foreground/85">
            {trace.summary}
          </span>
        ) : (
          ` · ${trace.summary}`
        )
      ) : trace.ok === undefined && summary ? (
        <span className="ml-1">· 正在总结…</span>
      ) : null}
    </span>
  );
}

function StatusIcon({ status }: { status: WorkerTrace["status"] }) {
  if (status === "running") {
    return <Loader2 className="mt-0.5 h-3 w-3 shrink-0 animate-spin" />;
  }
  if (status === "error") {
    return <X className="mt-0.5 h-3 w-3 shrink-0 text-destructive" />;
  }
  return <Check className="mt-0.5 h-3 w-3 shrink-0 text-success" />;
}
