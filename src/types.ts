export type Settings = {
  baseUrl: string;
  password: string;
  model: string;
  workspace: string;
  thinking: string;
  maxTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  alphaPresence: number;
  alphaFrequency: number;
  alphaDecay: number;
};

export type WorkerToolTrace = {
  name: string;
  ok?: boolean;
  summary?: string;
  round?: number;
};

export type WorkerTrace = {
  index: number;
  goal: string;
  status: "running" | "done" | "error";
  output?: string;
  traces?: WorkerToolTrace[];
};

export type ToolTrace = {
  id: string;
  step: number;
  name: string;
  arguments: unknown;
  ok?: boolean;
  payload?: unknown;
  open: boolean;
  workers?: WorkerTrace[];
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  tools: ToolTrace[];
  streaming?: boolean;
  error?: string;
};

export type EventPayload = {
  turnId?: string;
  prompt?: string;
  workspace?: string;
  model?: string;
  text?: string;
  step?: number;
  kind?: string;
  tool?: string;
  endpoint?: string;
  path?: string;
  contentType?: string;
  bytes?: number;
  statusCode?: number;
  cuda?: boolean;
  stream?: boolean;
  arguments?: unknown;
  ok?: boolean;
  payload?: unknown;
  error?: string;
  message?: string;
  output?: string;
  index?: number;
  goal?: string;
  status?: string | number;
  summary?: string;
  round?: number;
  intent?: string;
  source?: string;
  event?: string;
  detail?: unknown;
  callId?: string;
  phase?: string;
  attempt?: number;
  modelAttempt?: number;
  batchSize?: number;
  request?: unknown;
  response?: unknown;
  elapsedMs?: number;
  traceId?: number;
};

export type CallLogKind =
  | "turn_start"
  | "intent_resolved"
  | "status"
  | "step_mark"
  | "model_request"
  | "model_response"
  | "transport"
  | "tool_start"
  | "tool_done"
  | "retry"
  | "subagent"
  | "subagent_tool"
  | "turn_done"
  | "turn_error";

export type CallLogStatus = "running" | "success" | "error" | "info";

export type CallLogEntry = {
  id: string;
  seq: number;
  receivedAt: number;
  turnId?: string;
  kind: CallLogKind;
  status: CallLogStatus;
  title: string;
  subtitle?: string;
  phase?: string;
  step?: number;
  index?: number;
  round?: number;
  callId?: string;
  elapsedMs?: number;
  payload: EventPayload;
};

export type SessionMeta = {
  id: string;
  title: string;
  updatedAt: number;
  pinned: boolean;
};

export type ProjectView = {
  id: string;
  path: string;
  name: string;
  sessions: SessionMeta[];
};

export type SidebarData = {
  activeSessionId: string;
  activeProjectId: string;
  projects: ProjectView[];
};

export type SessionSnap = {
  sidebar: SidebarData;
  messages: ChatMessage[];
  workspace: string;
};
