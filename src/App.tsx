import { FormEvent, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type EventCallback, type UnlistenFn } from "@tauri-apps/api/event";
import { PanelLeft } from "lucide-react";

import { AppSidebar } from "./components/Sidebar";
import { CallLogPage } from "./components/logs/CallLogPage";
import { SettingsSheet } from "./components/SettingsSheet";
import { Composer } from "./components/chat/Composer";
import { EmptyHome } from "./components/chat/EmptyHome";
import { MessageList } from "./components/chat/MessageList";
import { SubagentPanel } from "./components/chat/SubagentPanel";
import {
  latestRunAgents,
  upsertWorkerTrace,
  workersFromArguments,
  workersFromPayload,
  workersOfTool,
} from "./lib/workers";
import { makeCallLogEntry } from "./lib/callLogs";
import type {
  ChatMessage,
  CallLogEntry,
  CallLogKind,
  EventPayload,
  SessionSnap,
  Settings,
  SidebarData,
  ToolTrace,
} from "./types";

function uid() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

type ListenerGate = {
  promise: Promise<boolean>;
  resolve: (ready: boolean) => void;
};

const LISTENER_SETUP_TIMEOUT_MS = 10_000;

function createListenerGate(): ListenerGate {
  let resolve!: (ready: boolean) => void;
  const promise = new Promise<boolean>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

function durable(messages: ChatMessage[]): ChatMessage[] {
    return messages
    .filter(
      (message) =>
        !message.streaming || message.content || message.error || message.tools.length > 0,
    )
    .map((message) => sanitizeMessage({
      ...message,
      streaming: undefined,
      tools: message.tools.map((tool) => ({ ...tool, open: false })),
    }));
}

function protocolText(text: string) {
  const lower = text.toLowerCase();
  const wrapped = [
    "<think",
    "</think",
    "<tool_call",
    "```json",
    "function output:",
  ].some((marker) => lower.includes(marker));
  return wrapped &&
    (lower.includes('"name"') ||
      lower.includes('"function"') ||
      lower.includes('"arguments"'));
}

function sanitizeMessage(message: ChatMessage): ChatMessage {
  if (message.role !== "assistant" || !protocolText(message.content)) return message;
  return {
    ...message,
    content: "",
    error: message.error || "历史回答包含未完成的工具协议，原始协议已隐藏；请重新发送原问题。",
  };
}

function readDark() {
  return localStorage.getItem("rwkv-theme") === "dark";
}

function applyDark(dark: boolean) {
  document.documentElement.classList.toggle("dark", dark);
  localStorage.setItem("rwkv-theme", dark ? "dark" : "light");
}

function lastAssistant(messages: ChatMessage[], streamingOnly = false) {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "assistant") continue;
    if (streamingOnly && !message.streaming) continue;
    return message;
  }
  return undefined;
}

function hasPendingTool(message: ChatMessage, toolName: string, step?: number) {
  return pendingToolIndex(message, toolName, step) >= 0;
}

function pendingToolIndex(message: ChatMessage, toolName: string, step?: number) {
  for (let index = message.tools.length - 1; index >= 0; index -= 1) {
    const tool = message.tools[index];
    if (
      tool.name === toolName &&
      tool.ok === undefined &&
      (step === undefined || tool.step === step)
    ) {
      return index;
    }
  }
  return -1;
}

function pendingToolAssistantId(
  messages: ChatMessage[],
  toolName: string,
  step?: number,
  preferredId?: string | null,
) {
  if (preferredId) {
    const preferred = messages.find((message) => message.id === preferredId);
    return preferred &&
      preferred.role === "assistant" &&
      hasPendingTool(preferred, toolName, step)
      ? preferredId
      : undefined;
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && hasPendingTool(message, toolName, step)) {
      return message.id;
    }
  }
  return undefined;
}

function failPendingTools(tools: ToolTrace[], error: string): ToolTrace[] {
  return tools.map((tool) => {
    if (tool.ok !== undefined) return tool;
    const workers = tool.workers?.map((worker) => {
      if (worker.status !== "running") return worker;
      return {
        ...worker,
        status: "error" as const,
        output: worker.output || error,
        traces: worker.traces?.map((trace) =>
          trace.ok === undefined ? { ...trace, ok: false } : trace,
        ),
      };
    });
    return {
      ...tool,
      ok: false,
      workers,
      open: tool.name === "run_agents" ? true : tool.open,
    };
  });
}

export default function App() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [sidebar, setSidebar] = useState<SidebarData>({
    activeSessionId: "",
    activeProjectId: "",
    projects: [],
  });
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("idle");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsError, setSettingsError] = useState("");
  const [collapsed, setCollapsed] = useState(false);
  const [dark, setDark] = useState(readDark);
  const [selectedWorker, setSelectedWorker] = useState(0);
  const [activeView, setActiveView] = useState<"chat" | "logs">("chat");
  const [callLogs, setCallLogs] = useState<CallLogEntry[]>([]);
  const assistantId = useRef<string | null>(null);
  const activeTurnId = useRef<string | null>(null);
  const agentsCollecting = useRef(false);
  const cancelRequestedRef = useRef(false);
  const busyRef = useRef(false);
  const messagesRef = useRef<ChatMessage[]>([]);
  const commitQueue = useRef<Promise<void>>(Promise.resolve());
  const listenerGeneration = useRef(0);
  const callLogSequence = useRef(0);
  const listenerGateRef = useRef<ListenerGate | null>(null);
  if (!listenerGateRef.current) listenerGateRef.current = createListenerGate();

  function appendCallLog(kind: CallLogKind, payload: EventPayload) {
    const entry = makeCallLogEntry(kind, payload, callLogSequence.current + 1);
    callLogSequence.current += 1;
    setCallLogs((previous) => {
      let next = [...previous, entry];
      if (kind === "model_response" && entry.callId) {
        next = next.map((item) => {
          if (item.kind !== "model_request" || item.callId !== entry.callId) return item;
          if (
            entry.payload.attempt !== undefined &&
            item.payload.attempt !== entry.payload.attempt
          ) {
            return item;
          }
          return {
            ...item,
            status: entry.status,
            elapsedMs: item.elapsedMs ?? entry.elapsedMs,
          };
        });
      }
      if (kind === "turn_done" || kind === "turn_error") {
        const terminalStatus = kind === "turn_done" ? "success" : "error";
        next = next.map((item) =>
          item.kind === "model_request" && item.status === "running"
            ? { ...item, status: terminalStatus }
            : item,
        );
      }
      return next;
    });
  }

  function replaceMessages(next: ChatMessage[]) {
    messagesRef.current = next;
    setMessages(next);
    return next;
  }

  function updateMessages(updater: (previous: ChatMessage[]) => ChatMessage[]) {
    return replaceMessages(updater(messagesRef.current));
  }

  function setTurnBusy(next: boolean) {
    busyRef.current = next;
    setBusy(next);
  }

  function isCurrentTurn(turnId?: string, generation?: number) {
    return Boolean(
      turnId &&
        activeTurnId.current === turnId &&
        (generation === undefined || listenerGeneration.current === generation),
    );
  }

  function applySnap(snap: SessionSnap) {
    agentsCollecting.current = false;
    cancelRequestedRef.current = false;
    assistantId.current = null;
    activeTurnId.current = null;
    callLogSequence.current = 0;
    setCallLogs([]);
    setActiveView("chat");
    setSidebar(snap.sidebar);
    replaceMessages(snap.messages.map(sanitizeMessage));
    setDraft("");
    setSettings((current) =>
      current ? { ...current, workspace: snap.workspace } : current,
    );
  }

  useEffect(() => {
    applyDark(dark);
  }, [dark]);

  useEffect(() => {
    Promise.all([invoke<Settings>("get_settings"), invoke<SessionSnap>("get_sessions")])
      .then(([nextSettings, snap]) => {
        setSettings(nextSettings);
        applySnap(snap);
      })
      .catch((err) => setSettingsError(String(err)));
  }, []);

  useEffect(() => {
    let cancelled = false;
    const generation = listenerGeneration.current + 1;
    listenerGeneration.current = generation;
    const previousGate = listenerGateRef.current;
    const gate = createListenerGate();
    listenerGateRef.current = gate;
    previousGate?.resolve(false);
    const unlisteners: UnlistenFn[] = [];
    const register = (event: string, handler: EventCallback<EventPayload>) =>
      Promise.resolve().then(() => listen<EventPayload>(event, handler));
    const close = (fns: UnlistenFn[]) => {
      fns.forEach((fn) => {
        try {
          fn();
        } catch {}
      });
    };
    const fulfilledListeners = (results: PromiseSettledResult<UnlistenFn>[]) =>
      results.flatMap((result) =>
        result.status === "fulfilled" ? [result.value] : [],
      );
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    const bind = async () => {
      const registrations = Promise.allSettled([
        register("intent_resolved", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          appendCallLog("intent_resolved", event.payload);
          if (event.payload.intent) {
            setStatus(`已识别意图：${event.payload.intent}`);
          }
        }),
        register("model_trace", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          const kind = event.payload.kind;
          if (kind === "model_request" || kind === "model_response" || kind === "transport") {
            appendCallLog(kind, event.payload);
          }
        }),
        register("status", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          appendCallLog("status", event.payload);
          if (event.payload.text) setStatus(event.payload.text);
        }),
        register("token_delta", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          const text = event.payload.text ?? "";
          if (!text || agentsCollecting.current || protocolText(text)) return;
          const preferredAssistantId = assistantId.current;
          updateMessages((prev) => {
            const activeId = preferredAssistantId;
            const id = activeId
              ? prev.some(
                  (message) =>
                    message.id === activeId && message.role === "assistant" && message.streaming,
                )
                ? activeId
                : undefined
              : lastAssistant(prev, true)?.id;
            if (!id) return prev;
            return prev.map((message) =>
              message.id === id ? { ...message, content: message.content + text } : message,
            );
          });
        }),
        register("tool_start", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          if (!event.payload.tool) return;
          appendCallLog("tool_start", event.payload);
          const activeAssistantId = assistantId.current;
          if (event.payload.tool === "run_agents") agentsCollecting.current = true;
          const trace: ToolTrace = {
            id: uid(),
            step: event.payload.step ?? 0,
            name: event.payload.tool,
            arguments: event.payload.arguments,
            open: event.payload.tool === "run_agents",
            workers: workersFromArguments(event.payload.arguments),
          };
          updateMessages((prev) => {
            const target = activeAssistantId
              ? prev.find((message) => message.id === activeAssistantId)
              : lastAssistant(prev, true);
            if (!target) return prev;
            return prev.map((message) =>
              message.id === target.id
                ? {
                    ...message,
                    content: trace.name === "run_agents" ? "" : message.content,
                    tools: [...message.tools, trace],
                  }
                : message,
            );
          });
        }),
        register("subagent", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          const index = event.payload.index;
          if (index === undefined) return;
          appendCallLog("subagent", event.payload);
          const preferredAssistantId = assistantId.current;
          updateMessages((prev) => {
            const targetId = pendingToolAssistantId(
              prev,
              "run_agents",
              event.payload.step,
              preferredAssistantId,
            );
            if (!targetId) return prev;
            return prev.map((message) => {
              if (message.id !== targetId) return message;
              const targetToolIndex = pendingToolIndex(
                message,
                "run_agents",
                event.payload.step,
              );
              if (targetToolIndex < 0) return message;
              const tools = message.tools.map((tool, toolIndex) => {
                if (toolIndex !== targetToolIndex) return tool;
                const workers = [...(tool.workers ?? [])];
                const current = workers[index] ?? {
                  index,
                  goal: event.payload.goal ?? "",
                  status: "running" as const,
                };
                workers[index] = {
                  ...current,
                  goal: event.payload.goal || current.goal,
                  status:
                    event.payload.status === "done" || event.payload.status === "error"
                      ? event.payload.status
                      : "running",
                  output: event.payload.output ?? current.output,
                  traces: current.traces ?? [],
                };
                return { ...tool, workers, open: true };
              });
              return { ...message, tools };
            });
          });
        }),
        register("subagent_tool", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          const index = event.payload.index;
          const toolName = event.payload.tool;
          if (index === undefined || !toolName) return;
          appendCallLog("subagent_tool", event.payload);
          const preferredAssistantId = assistantId.current;
          updateMessages((prev) => {
            const targetId = pendingToolAssistantId(
              prev,
              "run_agents",
              event.payload.step,
              preferredAssistantId,
            );
            if (!targetId) return prev;
            return prev.map((message) => {
              if (message.id !== targetId) return message;
              const targetToolIndex = pendingToolIndex(
                message,
                "run_agents",
                event.payload.step,
              );
              if (targetToolIndex < 0) return message;
              const tools = message.tools.map((tool, toolIndex) => {
                if (toolIndex !== targetToolIndex) return tool;
                const workers = [...(tool.workers ?? [])];
                const current = workers[index] ?? {
                  index,
                  goal: "",
                  status: "running" as const,
                  traces: [],
                };
                const traces = upsertWorkerTrace(current.traces ?? [], {
                  name: toolName,
                  ok: event.payload.ok,
                  summary: event.payload.summary,
                  round: event.payload.round,
                });
                workers[index] = { ...current, traces };
                return { ...tool, workers, open: true };
              });
              return { ...message, tools };
            });
          });
        }),
        register("tool_done", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          const toolName = event.payload.tool;
          const preferredAssistantId = assistantId.current;
          if (!toolName) return;
          appendCallLog("tool_done", event.payload);
          updateMessages((prev) => {
            const targetId = pendingToolAssistantId(
              prev,
              toolName,
              event.payload.step,
              preferredAssistantId,
            );
            if (targetId && toolName === "run_agents") agentsCollecting.current = false;
            return prev.map((message) => {
              if (message.id !== targetId) return message;
              const tools = [...message.tools];
              for (let i = tools.length - 1; i >= 0; i -= 1) {
                if (
                  tools[i].name === toolName &&
                  tools[i].ok === undefined &&
                  (event.payload.step === undefined || tools[i].step === event.payload.step)
                ) {
                  const workers = workersFromPayload(event.payload.payload, tools[i].workers);
                  tools[i] = {
                    ...tools[i],
                    ok: event.payload.ok,
                    payload: event.payload.payload,
                    workers,
                    open: toolName === "run_agents" ? true : tools[i].open,
                  };
                  break;
                }
              }
              return { ...message, tools };
            });
          });
        }),
        register("retry", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          appendCallLog("retry", event.payload);
          setStatus(event.payload.error ? `retry · ${event.payload.error}` : "retry");
        }),
        register("step_mark", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          appendCallLog("step_mark", event.payload);
        }),
        register("turn_done", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          appendCallLog("turn_done", event.payload);
          cancelRequestedRef.current = false;
          setTurnBusy(false);
          setStatus("idle");
          agentsCollecting.current = false;
          const targetId = assistantId.current;
          assistantId.current = null;
          activeTurnId.current = null;
          const next = updateMessages((prev) => {
            const target = targetId
              ? prev.find((message) => message.id === targetId && message.role === "assistant")
              : lastAssistant(prev, true);
            return prev.map((message) => {
              if (message.id !== target?.id) return message;
              return sanitizeMessage({
                ...message,
                streaming: false,
                content: event.payload.output || message.content,
              });
            });
          });
          void commit(next).catch(() => undefined);
        }),
        register("turn_error", (event) => {
          if (!isCurrentTurn(event.payload.turnId, generation)) return;
          appendCallLog("turn_error", event.payload);
          cancelRequestedRef.current = false;
          setTurnBusy(false);
          setStatus("idle");
          agentsCollecting.current = false;
          const targetId = assistantId.current;
          assistantId.current = null;
          activeTurnId.current = null;
          const error = event.payload.message || "turn failed";
          const next = updateMessages((prev) => {
            const target = targetId
              ? prev.find((message) => message.id === targetId && message.role === "assistant")
              : lastAssistant(prev, true);
            return prev.map((message) =>
              message.id === target?.id
                ? {
                    ...message,
                    streaming: false,
                    error,
                    tools: failPendingTools(message.tools, error),
                  }
                : message,
            );
          });
          void commit(next).catch(() => undefined);
        }),
      ]);
      const timeout = new Promise<"timeout">((resolve) => {
        timeoutHandle = setTimeout(() => resolve("timeout"), LISTENER_SETUP_TIMEOUT_MS);
      });
      const result = await Promise.race([registrations, timeout]);
      if (timeoutHandle !== undefined) {
        clearTimeout(timeoutHandle);
        timeoutHandle = undefined;
      }
      if (result === "timeout") {
        const stillCurrent = !cancelled && listenerGeneration.current === generation;
        cancelled = true;
        if (listenerGeneration.current === generation) listenerGeneration.current += 1;
        gate.resolve(false);
        void registrations.then((lateResults) => close(fulfilledListeners(lateResults)));
        if (stillCurrent) {
          setSettingsError("消息事件监听初始化超时，请重新打开应用后重试。");
        }
        return;
      }
      const results = result;
      const fns = fulfilledListeners(results);
      const failed = results.some((registration) => registration.status === "rejected");
      if (cancelled || listenerGeneration.current !== generation) {
        close(fns);
        gate.resolve(false);
        return;
      }
      if (failed) {
        close(fns);
        gate.resolve(false);
        setSettingsError("消息事件监听初始化失败，请重新打开应用后重试。");
        return;
      }
      unlisteners.push(...fns);
      gate.resolve(true);
    };
    void bind();
    return () => {
      cancelled = true;
      cancelRequestedRef.current = true;
      if (listenerGeneration.current === generation) listenerGeneration.current += 1;
      if (timeoutHandle !== undefined) {
        clearTimeout(timeoutHandle);
        timeoutHandle = undefined;
      }
      close(unlisteners.splice(0));
      gate.resolve(false);
    };
  }, []);

  async function commit(next: ChatMessage[]) {
    const messagesToSave = durable(next);
    const save = commitQueue.current
      .catch(() => undefined)
      .then(() =>
        invoke<SessionSnap>("save_transcript", {
          messages: messagesToSave,
        }),
      );
    commitQueue.current = save.then(
      () => undefined,
      () => undefined,
    );
    const snap = await save;
    setSidebar(snap.sidebar);
  }

  async function sendPrompt(text: string) {
    if (!settings || busyRef.current) return;
    const prompt = text.trim();
    if (!prompt) return;
    cancelRequestedRef.current = false;
    setTurnBusy(true);
    setStatus("connecting");
    let gate = listenerGateRef.current;
    let listenersReady = true;
    while (gate) {
      listenersReady = await gate.promise;
      const latestGate = listenerGateRef.current;
      if (latestGate && latestGate !== gate) {
        gate = latestGate;
        continue;
      }
      break;
    }
    if (cancelRequestedRef.current) {
      cancelRequestedRef.current = false;
      setTurnBusy(false);
      setStatus("idle");
      return;
    }
    if (!listenersReady) {
      cancelRequestedRef.current = false;
      setTurnBusy(false);
      setStatus("idle");
      setSettingsError("消息事件监听不可用，请重新打开应用后重试。");
      return;
    }
    const user: ChatMessage = { id: uid(), role: "user", content: prompt, tools: [] };
    const assistant: ChatMessage = {
      id: uid(),
      role: "assistant",
      content: "",
      tools: [],
      streaming: true,
    };
    const turnId = uid();
    assistantId.current = assistant.id;
    activeTurnId.current = turnId;
    agentsCollecting.current = false;
    appendCallLog("turn_start", {
      turnId,
      prompt,
      workspace: settings.workspace,
      model: settings.model,
    });
    setDraft("");
    setStatus("generating");
    setSelectedWorker(0);
    const next = [...messagesRef.current, user, assistant];
    replaceMessages(next);
    void commit([...next.slice(0, -1)]).catch(() => undefined);
    try {
      await invoke("chat_send", { prompt, turnId });
    } catch (err) {
      if (activeTurnId.current !== turnId) return;
      cancelRequestedRef.current = false;
      agentsCollecting.current = false;
      const error = String(err);
      appendCallLog("turn_error", {
        turnId,
        message: error,
        source: "frontend",
        phase: "agent",
      });
      assistantId.current = null;
      activeTurnId.current = null;
      setTurnBusy(false);
      setStatus("idle");
      const failed = updateMessages((prev) =>
        prev.map((message) =>
          message.id === assistant.id
            ? {
                ...message,
                streaming: false,
                error,
                tools: failPendingTools(message.tools, error),
              }
            : message,
        ),
      );
      void commit(failed).catch(() => undefined);
    }
  }

  async function onSend(event?: FormEvent) {
    event?.preventDefault();
    await sendPrompt(draft);
  }

  function onStop() {
    if (!busyRef.current) return;
    cancelRequestedRef.current = true;
    void invoke("chat_cancel").catch(() => undefined);
  }

  async function onRetry(messageId: string) {
    if (busyRef.current) return;
    const currentMessages = messagesRef.current;
    const failedIndex = currentMessages.findIndex((message) => message.id === messageId);
    if (failedIndex < 0) return;
    for (let index = failedIndex - 1; index >= 0; index -= 1) {
      const message = currentMessages[index];
      if (message.role !== "user") continue;
      await sendPrompt(message.content);
      return;
    }
  }

  async function onNewSession(projectId?: string) {
    const id = projectId || sidebar.activeProjectId || sidebar.projects[0]?.id;
    if (busyRef.current || !id) return;
    await commit(messagesRef.current);
    applySnap(await invoke<SessionSnap>("new_session", { projectId: id }));
  }

  async function onOpenProject(id: string) {
    if (busyRef.current) return;
    await commit(messagesRef.current);
    applySnap(await invoke<SessionSnap>("open_project", { id }));
  }

  async function onOpenSession(id: string) {
    if (busyRef.current || id === sidebar.activeSessionId) return;
    await commit(messagesRef.current);
    applySnap(await invoke<SessionSnap>("open_session", { id }));
  }

  async function onAddProject() {
    if (busyRef.current) return;
    await commit(messagesRef.current);
    applySnap(await invoke<SessionSnap>("add_project"));
  }

  async function persist(next: Settings) {
    setSettingsError("");
    try {
      setSettings(await invoke<Settings>("save_settings", { settings: next }));
    } catch (err) {
      setSettingsError(String(err));
    }
  }

  const activeProject = sidebar.projects.find((project) => project.id === sidebar.activeProjectId);
  const workspaceName =
    activeProject?.name ||
    settings?.workspace.split("/").filter(Boolean).slice(-1)[0] ||
    "";
  const runAgents = latestRunAgents(messages);
  const workers = runAgents ? workersOfTool(runAgents.tool) : [];
  const selected = Math.min(selectedWorker, Math.max(workers.length - 1, 0));
  const visibleMessages =
    runAgents && runAgents.message.streaming && !runAgents.message.content
      ? messages.filter((message) => message.id !== runAgents.message.id)
      : messages;

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <AppSidebar
        collapsed={collapsed}
        busy={busy}
        data={sidebar}
        onNewSession={onNewSession}
        onAddProject={onAddProject}
        onOpenProject={onOpenProject}
        onOpenSession={onOpenSession}
        onPinSession={async (id) => {
          setSidebar((await invoke<SessionSnap>("pin_session", { id })).sidebar);
        }}
        onDeleteSession={async (id) => {
          if (busyRef.current) return;
          if (id !== sidebar.activeSessionId) await commit(messagesRef.current);
          applySnap(await invoke<SessionSnap>("delete_session", { id }));
        }}
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenCallLogs={() => setActiveView("logs")}
        callLogCount={callLogs.length}
        callLogsOpen={activeView === "logs"}
      />

      <main className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-background">
        <button
          type="button"
          onClick={() => setCollapsed((value) => !value)}
          className="absolute top-3 left-3 z-10 flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground"
          aria-label={collapsed ? "展开侧栏" : "收起侧栏"}
        >
          <PanelLeft className="h-4 w-4" />
        </button>

        {activeView === "logs" ? (
          <CallLogPage
            entries={callLogs}
            workspaceName={workspaceName}
            busy={busy}
            onOpenChat={() => setActiveView("chat")}
            onClear={() => {
              callLogSequence.current = 0;
              setCallLogs([]);
            }}
          />
        ) : (
          <>
            {messages.length === 0 ? (
              <EmptyHome projectName={workspaceName} onSelect={setDraft} />
            ) : (
              <MessageList
                messages={visibleMessages}
                status={status}
                onRetry={(messageId) => void onRetry(messageId)}
                onToggleTool={(messageId, toolId) => {
                  updateMessages((prev) =>
                    prev.map((message) =>
                      message.id === messageId
                        ? {
                            ...message,
                            tools: message.tools.map((tool) =>
                              tool.id === toolId ? { ...tool, open: !tool.open } : tool,
                            ),
                          }
                        : message,
                    ),
                  );
                }}
              />
            )}

            <Composer
              draft={draft}
              busy={busy}
              workspaceName={workspaceName}
              onChange={setDraft}
              onSend={() => void onSend()}
              onStop={onStop}
            />
          </>
        )}

        {settingsOpen && settings && (
          <SettingsSheet
            settings={settings}
            error={settingsError}
            dark={dark}
            onChange={setSettings}
            onPersist={persist}
            onChooseWorkspace={onAddProject}
            onToggleDark={() => setDark((value) => !value)}
            onClose={() => setSettingsOpen(false)}
          />
        )}
      </main>
      {activeView === "chat" && runAgents ? (
        <SubagentPanel
          workers={workers}
          running={runAgents.tool.ok === undefined}
          selected={selected}
          workspaceName={workspaceName}
          onSelect={setSelectedWorker}
        />
      ) : null}
    </div>
  );
}
