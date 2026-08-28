import type { ChatMessage, ToolTrace, WorkerToolTrace, WorkerTrace } from "../types";

type WorkerWire = WorkerTrace & { ok?: boolean };

export function workersFromArguments(value: unknown): WorkerTrace[] | undefined {
  if (!value || typeof value !== "object") return undefined;
  const tasks = (value as { tasks?: unknown }).tasks;
  if (!Array.isArray(tasks) || tasks.length === 0) return undefined;
  return tasks.map((task, index) => {
    const goal =
      typeof task === "string"
        ? task
        : String((task as { goal?: string } | null)?.goal ?? "");
    return {
      index,
      goal,
      status: "running" as const,
      traces: [],
    };
  });
}

export function workersFromPayload(
  payload: unknown,
  fallback?: WorkerTrace[],
): WorkerTrace[] | undefined {
  const raw = pickWorkers(payload);
  if (!Array.isArray(raw) || raw.length === 0) return fallback;
  return raw.map((worker, index) => normalizeWorker(worker, index));
}

export function workersOfTool(tool: ToolTrace): WorkerTrace[] {
  if (tool.workers && tool.workers.length > 0) return tool.workers;
  return workersFromPayload(tool.payload) ?? [];
}

export function upsertWorkerTrace(
  traces: WorkerToolTrace[],
  incoming: WorkerToolTrace,
): WorkerToolTrace[] {
  const next = [...traces];
  let found = -1;
  let fallback = -1;
  for (let index = next.length - 1; index >= 0; index -= 1) {
    const current = next[index];
    if (
      current.name !== incoming.name ||
      (current.round ?? 0) !== (incoming.round ?? 0) ||
      current.ok !== undefined
    ) {
      continue;
    }
    const currentKey = traceOperationKey(current.summary);
    const incomingKey = traceOperationKey(incoming.summary);
    if (fallback < 0 && (incoming.ok !== undefined || !incomingKey)) fallback = index;
    if (currentKey && incomingKey && currentKey !== incomingKey) continue;
    found = index;
    break;
  }
  if (found < 0) found = fallback;
  if (found < 0) return [...next, incoming];

  const current = next[found];
  next[found] = {
    ...current,
    ...incoming,
    summary: incoming.summary ?? current.summary,
    round: incoming.round ?? current.round,
  };
  return next;
}

function traceOperationKey(summary?: string) {
  if (!summary) return undefined;
  const normalized = summary.replace(/\s+/g, " ").trim();
  const match = normalized.match(/第\s*(\d+)\s*段/);
  if (!match || match.index === undefined) return undefined;
  const path = normalized
    .slice(0, match.index)
    .replace(/^正在总结\s*/, "")
    .replace(/[·:：\s]+$/, "")
    .trim();
  return path ? `${path}|${match[1]}` : `|${match[1]}`;
}

export function latestRunAgents(
  messages: ChatMessage[],
): { message: ChatMessage; tool: ToolTrace } | undefined {
  const message = [...messages].reverse().find((item) => item.role === "assistant");
  if (!message) return undefined;
  for (let index = message.tools.length - 1; index >= 0; index -= 1) {
    const tool = message.tools[index];
    if (tool.name !== "run_agents") continue;
    if (workersOfTool(tool).length > 0 || tool.ok === undefined) {
      return { message, tool };
    }
  }
  return undefined;
}

function pickWorkers(payload: unknown): unknown {
  if (!payload || typeof payload !== "object") return undefined;
  const root = payload as { workers?: unknown; result?: { workers?: unknown } };
  if (Array.isArray(root.workers)) return root.workers;
  if (Array.isArray(root.result?.workers)) return root.result.workers;
  return undefined;
}

function normalizeWorker(value: WorkerWire, index: number): WorkerTrace {
  const status =
    value.status === "done" || value.status === "error" || value.status === "running"
      ? value.status
      : value.ok === false
        ? "error"
        : "done";
  return {
    index: value.index ?? index,
    goal: value.goal ?? "",
    status,
    output: value.output,
    traces: value.traces ?? [],
  };
}
