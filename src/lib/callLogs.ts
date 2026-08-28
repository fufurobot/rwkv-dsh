import type { CallLogEntry, CallLogKind, CallLogStatus, EventPayload } from "../types";

export function makeCallLogEntry(
  kind: CallLogKind,
  payload: EventPayload,
  seq: number,
  receivedAt = Date.now(),
): CallLogEntry {
  const status = statusFor(kind, payload);
  const title = titleFor(kind, payload);
  const subtitle = subtitleFor(kind, payload);
  return {
    id: `${receivedAt.toString(36)}-${seq.toString(36)}`,
    seq,
    receivedAt,
    turnId: payload.turnId,
    kind,
    status,
    title,
    subtitle,
    phase: payload.phase,
    step: payload.step,
    index: payload.index,
    round: payload.round,
    callId: payload.callId,
    elapsedMs: payload.elapsedMs,
    payload,
  };
}

export function phaseLabel(phase?: string) {
  switch (phase) {
    case "intent":
      return "意图识别";
    case "planner":
      return "并发规划";
    case "agent":
      return "主代理";
    case "chunk_summary":
      return "分块总结";
    case "agent_summary":
      return "子任务总结";
    case "synthesis":
      return "最终综合";
    default:
      return phase || "系统";
  }
}

export function kindLabel(kind: CallLogKind) {
  switch (kind) {
    case "turn_start":
      return "任务开始";
    case "intent_resolved":
      return "意图识别";
    case "status":
      return "状态";
    case "step_mark":
      return "流程步骤";
    case "model_request":
      return "模型请求";
    case "model_response":
      return "模型返回";
    case "transport":
      return "网络细节";
    case "tool_start":
      return "工具调用";
    case "tool_done":
      return "工具结果";
    case "retry":
      return "重试";
    case "subagent":
      return "子任务";
    case "subagent_tool":
      return "子任务动作";
    case "turn_done":
      return "任务完成";
    case "turn_error":
      return "任务失败";
  }
}

export type ModelExchange = {
  id: string;
  callId?: string;
  attempt?: number;
  request: CallLogEntry;
  response?: CallLogEntry;
  transports: CallLogEntry[];
};

export function formatJson(value: unknown) {
  if (value === undefined || value === null) return "（无）";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function asRecord(value: unknown): Record<string, any> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, any>)
    : {};
}

export function promptTexts(payload: EventPayload): string[] {
  const request = asRecord(payload.request);
  if (Array.isArray(request.prompts)) {
    return request.prompts.filter((value): value is string => typeof value === "string");
  }
  if (typeof request.prompt === "string") return [request.prompt];
  if (typeof payload.prompt === "string") return [payload.prompt];
  return [];
}

export function responseTexts(payload: EventPayload): string[] {
  if (typeof payload.error === "string" && payload.ok === false) return [payload.error];
  const response = asRecord(payload.response);
  if (Array.isArray(response.items) && response.items.length > 0) {
    return response.items.map(modelOutputText);
  }
  const single = modelOutputText(Object.keys(response).length ? response : payload);
  return single ? [single] : [];
}

export function lastUserQuestion(prompt: string): string {
  const trimmed = prompt.trim();
  if (!trimmed) return "";
  const chunks = trimmed.split(/(?:^|\n)User:\s*/);
  if (chunks.length >= 2) {
    return chunks[chunks.length - 1]
      .replace(/\n+Assistant:\s*```(?:json)?\s*$/i, "")
      .replace(/\n+Assistant:\s*$/, "")
      .trim();
  }
  return "";
}

export function modelOutputText(value: unknown): string {
  if (typeof value === "string") return value;
  const record = asRecord(value);
  for (const key of ["text", "content", "output", "preview"]) {
    if (typeof record[key] === "string" && record[key].trim()) return record[key];
  }
  if (typeof record.message === "string") return record.message;
  if (record.message && typeof record.message === "object") {
    const message = asRecord(record.message);
    if (typeof message.content === "string") return message.content;
  }
  const choices = Array.isArray(record.choices) ? record.choices : [];
  if (choices.length > 0) {
    const choiceText = choices.map(modelOutputText).filter(Boolean).join("\n\n");
    if (choiceText) return choiceText;
  }
  const formatted = formatJson(value);
  return formatted === "{}" || formatted === "[]" || formatted === "（无）" ? "" : formatted;
}

export function modelFinishReason(value: unknown): string {
  const record = asRecord(value);
  const reason = record.finishReason ?? record.finish_reason;
  return reason === undefined || reason === null ? "" : String(reason);
}

export function requestModelName(payload: EventPayload): string {
  const request = asRecord(payload.request);
  if (typeof request.model === "string" && request.model.trim()) return request.model;
  if (typeof payload.model === "string" && payload.model.trim()) return payload.model;
  return "";
}

export function requestParamRecord(payload: EventPayload): Record<string, unknown> {
  const request = asRecord(payload.request);
  const params = { ...request };
  delete params.prompt;
  delete params.prompts;
  return params;
}

export function collectModelExchanges(entries: CallLogEntry[]): ModelExchange[] {
  const usedResponses = new Set<string>();
  return entries
    .filter((entry) => entry.kind === "model_request")
    .map((request) => {
      const response = entries.find((entry) => {
        if (entry.kind !== "model_response" || usedResponses.has(entry.id)) return false;
        if (!request.callId || entry.callId !== request.callId) return false;
        if (
          request.payload.attempt !== undefined &&
          entry.payload.attempt !== undefined &&
          request.payload.attempt !== entry.payload.attempt
        ) {
          return false;
        }
        return true;
      });
      if (response) usedResponses.add(response.id);
      return {
        id: request.id,
        callId: request.callId,
        attempt: request.payload.attempt,
        request,
        response,
        transports: entries.filter(
          (entry) =>
            entry.kind === "transport" &&
            Boolean(request.callId) &&
            entry.callId === request.callId &&
            (request.payload.attempt === undefined ||
              entry.payload.attempt === undefined ||
              entry.payload.attempt === request.payload.attempt),
        ),
      };
    });
}

export function exchangeSearchText(exchange: ModelExchange) {
  return [
    eventSearchText(exchange.request),
    exchange.response ? eventSearchText(exchange.response) : "",
    ...exchange.transports.map(eventSearchText),
  ]
    .join(" ")
    .toLowerCase();
}

export function eventSearchText(entry: CallLogEntry) {
  const prompts = promptTexts(entry.payload);
  const outputs = responseTexts(entry.payload);
  return [
    entry.title,
    entry.subtitle,
    entry.kind,
    entry.phase,
    entry.payload.tool,
    entry.payload.intent,
    entry.payload.text,
    entry.payload.error,
    entry.payload.message,
    entry.payload.output,
    entry.payload.goal,
    entry.payload.summary,
    entry.payload.prompt,
    ...prompts,
    ...outputs,
    ...prompts.map(lastUserQuestion),
    usefulJson(entry.payload.arguments),
    usefulJson(entry.payload.payload),
    usefulJson(entry.payload.request),
    usefulJson(entry.payload.response),
    entry.payload.event,
    entry.payload.callId,
    entry.payload.statusCode,
    entry.payload.status,
    entry.payload.traceId,
    entry.payload.modelAttempt,
    usefulJson(entry.payload.detail),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export function entryPreview(entry: CallLogEntry) {
  const payload = entry.payload;
  const prompts = promptTexts(payload);
  const outputs = responseTexts(payload);
  const question = prompts.map(lastUserQuestion).find(Boolean) ?? "";
  const candidates = [
    typeof payload.prompt === "string" ? payload.prompt : "",
    question,
    outputs[0] ?? "",
    typeof payload.output === "string" ? payload.output : "",
    typeof payload.summary === "string" ? payload.summary : "",
    typeof payload.goal === "string" ? payload.goal : "",
    typeof payload.error === "string" ? payload.error : "",
    typeof payload.message === "string" ? payload.message : "",
    typeof payload.text === "string" ? payload.text : "",
    typeof payload.arguments === "string" ? payload.arguments : usefulJson(payload.arguments),
    typeof payload.payload === "string" ? payload.payload : usefulJson(payload.payload),
    prompts[0] ?? "",
    typeof payload.event === "string" ? payload.event : "",
    typeof payload.detail === "string" ? payload.detail : usefulJson(payload.detail),
  ];
  const value = candidates.find((candidate) => candidate.trim());
  if (!value) return "";
  const oneLine = value.replace(/\s+/g, " ").trim();
  return oneLine.length > 170 ? `${oneLine.slice(0, 170)}…` : oneLine;
}

function usefulJson(value: unknown) {
  if (value === undefined || value === null) return "";
  const formatted = formatJson(value);
  return formatted === "（无）" || formatted === "{}" || formatted === "[]" ? "" : formatted;
}

function statusFor(kind: CallLogKind, payload: EventPayload): CallLogStatus {
  if (kind === "transport") {
    if (payload.event?.includes("error") || payload.event === "response_error" || payload.event === "batch_response_error") return "error";
    if (payload.event?.includes("retry")) return "info";
    return "info";
  }
  if (kind === "model_response" || kind === "tool_done") {
    return payload.ok === false ? "error" : "success";
  }
  if (kind === "turn_error") return "error";
  if (kind === "subagent") {
    return payload.status === "error"
      ? "error"
      : payload.status === "done"
        ? "success"
        : "running";
  }
  if (kind === "subagent_tool" || kind === "tool_start" || kind === "model_request") {
    if (payload.ok === false) return "error";
    if (payload.ok === true) return "success";
    return "running";
  }
  if (kind === "retry") return "info";
  if (kind === "turn_done" || kind === "intent_resolved") return "success";
  if (kind === "turn_start") return "running";
  return "info";
}

function titleFor(kind: CallLogKind, payload: EventPayload) {
  switch (kind) {
    case "turn_start":
      return payload.prompt ? `提问 · ${clip(payload.prompt, 28)}` : "开始处理请求";
    case "intent_resolved":
      return `识别为 ${intentLabel(payload.intent)}`;
    case "status":
      return payload.text || "流程状态更新";
    case "step_mark":
      return `进入第 ${payload.step ?? "?"} 步`;
    case "model_request":
      return questionTitle(payload, "发给模型");
    case "model_response":
      return payload.ok === false
        ? "模型返回错误"
        : responseTitle(payload, "模型返回");
    case "transport":
      return transportTitle(payload.event);
    case "tool_start":
      return `调用 ${payload.tool || "工具"}`;
    case "tool_done":
      return `${payload.tool || "工具"} 返回${payload.ok === false ? "失败" : "成功"}`;
    case "retry":
      return `第 ${payload.step ?? "?"} 步准备重试`;
    case "subagent":
      return `子任务 ${formatIndex(payload.index)} ${statusLabel(payload.status)}`;
    case "subagent_tool":
      return `子任务 ${formatIndex(payload.index)} · ${payload.tool || "动作"}`;
    case "turn_done":
      return "任务完成，输出已提交";
    case "turn_error":
      return "任务失败";
  }
}

function subtitleFor(kind: CallLogKind, payload: EventPayload) {
  const bits: string[] = [];
  if (payload.phase) bits.push(phaseLabel(payload.phase));
  if (payload.step !== undefined) bits.push(`步骤 ${payload.step}`);
  if (payload.round !== undefined) bits.push(`第 ${payload.round} 轮`);
  if (payload.attempt !== undefined) bits.push(`尝试 ${payload.attempt}`);
  if (payload.modelAttempt !== undefined && payload.modelAttempt !== payload.attempt) {
    bits.push(`模型尝试 ${payload.modelAttempt}`);
  }
  if (payload.batchSize !== undefined && payload.batchSize > 1) {
    bits.push(`${payload.batchSize} 路 batch`);
  }
  if (payload.elapsedMs !== undefined) bits.push(`${payload.elapsedMs} ms`);
  if (kind === "retry" && payload.error) bits.push(payload.error);
  if (kind === "transport" && payload.endpoint) bits.push(payload.endpoint);
  if (kind === "transport" && payload.statusCode !== undefined) {
    bits.push(`HTTP ${payload.statusCode}`);
  } else if (kind === "transport" && payload.status !== undefined) {
    bits.push(`HTTP ${payload.status}`);
  }
  if (kind === "status" && payload.text) return bits.filter((bit) => bit !== payload.text).join(" · ");
  if (kind === "subagent_tool" && payload.summary) bits.push(payload.summary);
  if (kind === "transport" && payload.detail !== undefined) {
    bits.push(typeof payload.detail === "string" ? payload.detail : formatJson(payload.detail));
  }
  if (kind === "subagent" && payload.goal) bits.push(payload.goal);
  if (kind === "turn_start" && payload.prompt) bits.push(payload.prompt);
  if (kind === "turn_done" && payload.output) bits.push(String(payload.output));
  if (kind === "model_request") {
    const question = promptTexts(payload).map(lastUserQuestion).find(Boolean);
    if (question) bits.push(question);
  }
  if (kind === "model_response") {
    const output = responseTexts(payload)[0];
    if (output) bits.push(output);
  }
  return bits.join(" · ");
}

function questionTitle(payload: EventPayload, fallback: string) {
  const question = promptTexts(payload).map(lastUserQuestion).find(Boolean);
  return question ? `${fallback} · ${clip(question, 24)}` : fallback;
}

function responseTitle(payload: EventPayload, fallback: string) {
  const output = responseTexts(payload)[0];
  return output ? `${fallback} · ${clip(output, 24)}` : fallback;
}

function clip(value: string, max: number) {
  const oneLine = value.replace(/\s+/g, " ").trim();
  return oneLine.length > max ? `${oneLine.slice(0, max)}…` : oneLine;
}

function intentLabel(intent?: string) {
  switch (intent) {
    case "chat":
      return "普通对话";
    case "capability":
      return "能力咨询";
    case "arithmetic":
      return "计算";
    case "inspect_single":
      return "单文件查看";
    case "inspect_project":
      return "项目分析";
    case "parallel_lookup":
      return "并发查找";
    case "edit_small":
      return "小范围修改";
    case "unknown":
      return "待确认";
    default:
      return intent || "未知意图";
  }
}

function formatIndex(index?: number) {
  return index === undefined ? "?" : String(index + 1).padStart(2, "0");
}

function statusLabel(status?: string | number) {
  switch (status) {
    case "done":
      return "完成";
    case "error":
      return "失败";
    case "running":
      return "进行中";
    default:
      return status === undefined ? "更新" : String(status);
  }
}

function transportTitle(event?: string) {
  switch (event) {
    case "backend_detect":
      return "检测模型后端";
    case "route_resolved":
      return "确定请求路由";
    case "request":
    case "batch_request":
      return "发起 HTTP 请求";
    case "response_headers":
    case "batch_response_headers":
      return "收到响应头";
    case "response_error":
    case "batch_response_error":
      return "HTTP 响应错误";
    case "endpoint_fallback":
      return "切换备用端点";
    case "request_transport_retry":
    case "batch_transport_retry":
      return "传输层重试";
    case "read_path":
      return "解析响应流";
    case "sse_first_delta":
      return "收到首个流片段";
    case "sse_done":
      return "流式响应结束";
    case "buffered_body":
    case "batch_buffered_body":
      return "读取缓冲响应";
    case "buffered_text":
      return "提取缓冲文本";
    default:
      return event ? `网络事件 · ${event}` : "网络事件";
  }
}
