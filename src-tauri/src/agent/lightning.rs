use futures_util::{future::join_all, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;

use super::protocol::FinishReason;

const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const MIN_TEMPERATURE: f32 = 0.001;
const THINKING_DISABLED_PREFIX: &str = "<think></think";
const TRANSPORT_RETRY_DELAYS_MS: [u64; 3] = [250, 750, 1_500];

pub type TraceSink = Arc<dyn Fn(Value) + Send + Sync>;

static TRACE_SINK: OnceLock<Mutex<Option<TraceSink>>> = OnceLock::new();
static TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(1);

struct TraceContext {
    call_id: String,
    phase: &'static str,
    attempt: AtomicU32,
}

tokio::task_local! {
    static ACTIVE_TRACE_CONTEXT: Arc<TraceContext>;
}

pub struct TraceGuard {
    previous: Option<TraceSink>,
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        if let Ok(mut slot) = trace_sink_slot().lock() {
            *slot = self.previous.take();
        }
    }
}

pub fn install_trace_sink(sink: TraceSink) -> TraceGuard {
    let mut slot = trace_sink_slot().lock().expect("trace sink lock");
    let previous = slot.replace(sink);
    TraceGuard { previous }
}

fn trace_sink_slot() -> &'static Mutex<Option<TraceSink>> {
    TRACE_SINK.get_or_init(|| Mutex::new(None))
}

fn emit_trace(mut payload: Value) {
    let sink = trace_sink_slot()
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(Arc::clone));
    let Some(sink) = sink else {
        return;
    };
    if let Some(object) = payload.as_object_mut() {
        object
            .entry("traceId")
            .or_insert_with(|| json!(TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed)));
    }
    sink(payload);
}

fn infer_trace_phase(prompt: &str) -> &'static str {
    let lower = prompt.to_lowercase();
    if lower.contains("choose only the distinct read-only investigations warranted by this request")
        || lower.contains("parallel planner")
        || lower.contains("并发规划")
    {
        "planner"
    } else if lower.contains("one evidence summary") || lower.contains("项目证据") {
        "chunk_summary"
    } else if lower.contains("sub-agent producing their final report")
        || lower.contains("子任务结论")
    {
        "agent_summary"
    } else if lower.contains("coordinating agent") || lower.contains("综合分析") {
        "synthesis"
    } else if (lower.contains("system: tools:")
        && lower.contains("exact tool names only. return only one json function call"))
        || (lower.contains("route") && lower.contains("inspect_project"))
    {
        "intent"
    } else {
        "agent"
    }
}

fn trace_request(
    call_id: &str,
    phase: &str,
    attempt: u32,
    request: &ContinueRequest,
    endpoint: &str,
    stream: bool,
    batch_size: usize,
) {
    emit_trace(json!({
        "kind": "model_request",
        "callId": call_id,
        "phase": phase,
        "attempt": attempt,
        "batchSize": batch_size,
        "request": {
            "endpoint": endpoint,
            "stream": stream,
            "model": request.model,
            "prompt": request.prompt,
            "maxTokens": request.max_output_tokens,
            "temperature": request.temperature,
            "topK": request.top_k,
            "topP": request.top_p,
            "alphaPresence": request.alpha_presence,
            "alphaFrequency": request.alpha_frequency,
            "alphaDecay": request.alpha_decay,
            "stops": request.stops,
        }
    }));
}

fn trace_response(
    call_id: &str,
    phase: &str,
    attempt: u32,
    elapsed_ms: u128,
    result: Result<&ContinueResult, &str>,
) {
    match result {
        Ok(result) => emit_trace(json!({
            "kind": "model_response",
            "callId": call_id,
            "phase": phase,
            "attempt": attempt,
            "ok": true,
            "elapsedMs": elapsed_ms,
            "response": {
                "text": result.text,
                "finishReason": result.finish_reason,
            }
        })),
        Err(error) => emit_trace(json!({
            "kind": "model_response",
            "callId": call_id,
            "phase": phase,
            "attempt": attempt,
            "ok": false,
            "elapsedMs": elapsed_ms,
            "error": error,
        })),
    }
}

fn trace_batch_request(
    call_id: &str,
    phase: &str,
    attempt: u32,
    request: &ContinueRequest,
    endpoint: &str,
    stream: bool,
    prompts: &[String],
) {
    emit_trace(json!({
        "kind": "model_request",
        "callId": call_id,
        "phase": phase,
        "attempt": attempt,
        "batchSize": prompts.len(),
        "request": {
            "endpoint": endpoint,
            "stream": stream,
            "model": request.model,
            "prompts": prompts,
            "maxTokens": request.max_output_tokens,
            "temperature": request.temperature,
            "topK": request.top_k,
            "topP": request.top_p,
            "alphaPresence": request.alpha_presence,
            "alphaFrequency": request.alpha_frequency,
            "alphaDecay": request.alpha_decay,
            "stops": request.stops,
        }
    }));
}

fn trace_batch_response(
    call_id: &str,
    phase: &str,
    attempt: u32,
    elapsed_ms: u128,
    result: Result<&[ContinueResult], &str>,
) {
    match result {
        Ok(results) => emit_trace(json!({
            "kind": "model_response",
            "callId": call_id,
            "phase": phase,
            "attempt": attempt,
            "ok": true,
            "elapsedMs": elapsed_ms,
            "response": {
                "items": results.iter().map(|item| json!({
                    "text": item.text,
                    "finishReason": item.finish_reason,
                })).collect::<Vec<_>>(),
            }
        })),
        Err(error) => emit_trace(json!({
            "kind": "model_response",
            "callId": call_id,
            "phase": phase,
            "attempt": attempt,
            "ok": false,
            "elapsedMs": elapsed_ms,
            "error": error,
        })),
    }
}

#[derive(Debug, Clone)]
pub struct ContinueRequest {
    pub prompt: String,
    pub model: String,
    pub max_output_tokens: u32,
    pub stops: Vec<String>,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub alpha_presence: f32,
    pub alpha_frequency: f32,
    pub alpha_decay: f32,
}

#[derive(Debug, Clone)]
pub struct ContinueResult {
    pub text: String,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone)]
pub struct LightningRoute {
    pub endpoint: String,
    pub cuda: bool,
}

fn log_event(event: &str, extra: Value) {
    let mut record = match extra {
        Value::Object(map) => map,
        other => {
            let mut map = serde_json::Map::new();
            map.insert("detail".into(), other);
            map
        }
    };
    static EXECUTABLE: OnceLock<String> = OnceLock::new();
    let executable = EXECUTABLE.get_or_init(|| {
        std::env::current_exe()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "unknown".into())
    });
    record.insert("event".into(), json!(event));
    let _ = ACTIVE_TRACE_CONTEXT.try_with(|context| {
        let model_attempt = context.attempt.load(Ordering::Relaxed);
        record
            .entry("callId")
            .or_insert_with(|| json!(context.call_id.clone()));
        record
            .entry("phase")
            .or_insert_with(|| json!(context.phase));
        record
            .entry("modelAttempt")
            .or_insert_with(|| json!(model_attempt));
        record
            .entry("attempt")
            .or_insert_with(|| json!(model_attempt));
    });
    record.insert("pid".into(), json!(std::process::id()));
    record.insert("exe".into(), json!(executable));
    record.insert(
        "t".into(),
        json!(SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_millis())
            .unwrap_or(0)),
    );
    let mut trace = Value::Object(record.clone());
    if let Some(object) = trace.as_object_mut() {
        object.insert("kind".into(), json!("transport"));
    }
    emit_trace(trace);
    let line = Value::Object(record).to_string();
    let path = std::env::temp_dir().join("rwkv-agent-lightning.jsonl");
    static LOG_PATH: std::sync::Once = std::sync::Once::new();
    LOG_PATH.call_once(|| {
        eprintln!("[lightning] writing request log to {}", path.display());
    });
    eprintln!("[lightning] {line}");
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(file, "{line}");
    }
}

fn preview(text: &str, max_chars: usize) -> String {
    let trimmed = text.replace('\n', "\\n");
    let count = trimmed.chars().count();
    if count <= max_chars {
        return trimmed;
    }
    format!(
        "{}…(+{} chars)",
        trimmed.chars().take(max_chars).collect::<String>(),
        count.saturating_sub(max_chars)
    )
}

fn preview_tail(text: &str, max_chars: usize) -> String {
    let normalized = text.replace('\n', "\\n");
    let count = normalized.chars().count();
    if count <= max_chars {
        return normalized;
    }
    format!(
        "…{}",
        normalized
            .chars()
            .skip(count.saturating_sub(max_chars))
            .collect::<String>()
    )
}

/// Pick `/v1/batch/completions` (CUDA) vs `/v1/chat/completions` (PyTorch).
///
/// `GET /v1/models` `owned_by` containing `"cuda"` → CUDA. Otherwise PyTorch.
/// Probe failure defaults to PyTorch; a 404 then flips once.
pub async fn resolve_route(client: &reqwest::Client, base_url: &str) -> LightningRoute {
    let root = v1_root(base_url);
    let cuda = detect_cuda_backend(client, &root).await.unwrap_or(false);
    let route = LightningRoute {
        endpoint: completions_path(&root, cuda),
        cuda,
    };
    log_event(
        "route_resolved",
        json!({
            "baseUrl": base_url,
            "endpoint": route.endpoint,
            "cuda": route.cuda,
        }),
    );
    route
}

fn completions_path(root: &str, cuda: bool) -> String {
    if cuda {
        format!("{root}/batch/completions")
    } else {
        format!("{root}/chat/completions")
    }
}

async fn detect_cuda_backend(client: &reqwest::Client, root: &str) -> Option<bool> {
    let url = format!("{root}/models");
    let response = client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let payload: Value = response.json().await.ok()?;
    let owned = payload
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("owned_by").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let cuda = owned.contains("cuda");
    log_event(
        "backend_detect",
        json!({
            "url": url,
            "ownedBy": owned,
            "cuda": cuda,
        }),
    );
    Some(cuda)
}

fn alternate_endpoint(endpoint: &str) -> Option<String> {
    if let Some(prefix) = endpoint.strip_suffix("/batch/completions") {
        return Some(format!("{prefix}/chat/completions"));
    }
    if let Some(prefix) = endpoint.strip_suffix("/chat/completions") {
        return Some(format!("{prefix}/batch/completions"));
    }
    None
}

impl LightningRoute {
    fn flip_endpoint(&mut self) -> bool {
        let Some(alt) = alternate_endpoint(&self.endpoint) else {
            return false;
        };
        self.endpoint = alt;
        self.cuda = self.endpoint.ends_with("/batch/completions");
        true
    }
}

fn v1_root(base_url: &str) -> String {
    let mut base = base_url.trim().trim_end_matches('/').to_string();
    if let Some(stripped) = base.strip_suffix("/openai/v1/chat/completions") {
        base = stripped.trim_end_matches('/').to_string();
    } else {
        for suffix in [
            "/chat/completions",
            "/batch/completions",
            "/big_batch/completions",
        ] {
            if let Some(stripped) = base.strip_suffix(suffix) {
                base = stripped.trim_end_matches('/').to_string();
                break;
            }
        }
    }
    if let Some(stripped) = base.strip_suffix("/openai/v1") {
        base = stripped.trim_end_matches('/').to_string();
    }
    if base.ends_with("/v1") {
        base
    } else {
        format!("{base}/v1")
    }
}

#[derive(Debug, Deserialize)]
struct StreamBody {
    choices: Option<Vec<StreamChoice>>,
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    index: Option<i64>,
    delta: Option<Delta>,
    message: Option<Delta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    content: Option<String>,
}

enum SseLine {
    Ignore,
    Done,
    Json(StreamBody),
}

fn parse_sse_line(line: &str) -> Result<SseLine, String> {
    let line = line.trim();
    if line == "[DONE]" {
        return Ok(SseLine::Done);
    }
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(SseLine::Ignore);
    };
    let data = data.trim();
    if data.is_empty() {
        return Ok(SseLine::Ignore);
    }
    if data == "[DONE]" {
        return Ok(SseLine::Done);
    }
    serde_json::from_str(data)
        .map(SseLine::Json)
        .map_err(|err| format!("decode stream chunk: {err}"))
}

pub async fn continue_prompt(
    client: &reqwest::Client,
    route: &mut LightningRoute,
    password: &str,
    request: ContinueRequest,
    cancel: watch::Receiver<bool>,
    on_delta: impl FnMut(&str) + Send,
) -> Result<ContinueResult, String> {
    let phase = infer_trace_phase(&request.prompt);
    continue_prompt_with_phase(client, route, password, request, phase, cancel, on_delta).await
}

pub async fn continue_prompt_with_phase(
    client: &reqwest::Client,
    route: &mut LightningRoute,
    password: &str,
    request: ContinueRequest,
    phase: &'static str,
    cancel: watch::Receiver<bool>,
    mut on_delta: impl FnMut(&str) + Send,
) -> Result<ContinueResult, String> {
    if request.prompt.trim().is_empty() {
        return Err("prompt is required".into());
    }
    if *cancel.borrow() {
        return Err("cancelled".into());
    }

    let call_id = format!("llm-{}", TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed));
    let trace_context = Arc::new(TraceContext {
        call_id: call_id.clone(),
        phase,
        attempt: AtomicU32::new(0),
    });
    ACTIVE_TRACE_CONTEXT
        .scope(Arc::clone(&trace_context), async move {
    let mut stream = true;
    let mut flipped = false;
    let mut empty_response_retry_index = 0;
    let mut attempt = 0u32;
    loop {
        if *cancel.borrow() {
            return Err("cancelled".into());
        }
        let body = completion_request_body(
            &request,
            std::slice::from_ref(&request.prompt),
            route.cuda,
            stream,
            password,
        );

        attempt += 1;
        trace_context.attempt.store(attempt, Ordering::Relaxed);
        trace_request(
            &call_id,
            phase,
            attempt,
            &request,
            &route.endpoint,
            stream,
            1,
        );
        let started = Instant::now();
        log_event(
            "request",
            json!({
                "endpoint": route.endpoint,
                "cuda": route.cuda,
                "stream": stream,
                "model": request.model,
                "maxTokens": request.max_output_tokens,
                "promptChars": request.prompt.chars().count(),
                "promptTail": preview(request.prompt.rsplit_once("User:").map(|(_, rest)| rest).unwrap_or(&request.prompt), 120),
                "promptEnd": preview_tail(&request.prompt, 360),
                "clientStops": request.stops,
            }),
        );

        let response = match send_with_transport_retries(
            client,
            &route.endpoint,
            password,
            &body,
            cancel.clone(),
            "request_transport_retry",
            json!({"stream": stream}),
        )
        .await
        {
            Ok(response) => response,
            Err(error) => {
                trace_response(
                    &call_id,
                    phase,
                    attempt,
                    started.elapsed().as_millis(),
                    Err(error.as_str()),
                );
                return Err(error);
            }
        };

        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        log_event(
            "response_headers",
            json!({
                "endpoint": route.endpoint,
                "stream": stream,
                "status": status.as_u16(),
                "statusCode": status.as_u16(),
                "contentType": content_type,
                "elapsedMs": started.elapsed().as_millis(),
            }),
        );

        if !status.is_success() {
            let raw = response.text().await.unwrap_or_default();
            let detail = redact(&raw, password);
            trace_response(
                &call_id,
                phase,
                attempt,
                started.elapsed().as_millis(),
                Err(detail.as_str()),
            );
            log_event(
                "response_error",
                json!({
                    "status": status.as_u16(),
                    "statusCode": status.as_u16(),
                    "stream": stream,
                    "bodyPreview": preview(&detail, 240),
                    "willRetryUnstreamed": status.as_u16() == 500 && route.cuda && stream,
                    "willFlipEndpoint": status.as_u16() == 404 && !flipped,
                }),
            );
            if status.as_u16() == 404 && !flipped && route.flip_endpoint() {
                flipped = true;
                log_event(
                    "endpoint_fallback",
                    json!({
                        "endpoint": route.endpoint,
                        "cuda": route.cuda,
                    }),
                );
                continue;
            }
            if status.as_u16() == 500 && route.cuda && stream {
                stream = false;
                continue;
            }
            if status.as_u16() == 500 && route.cuda {
                return Err(format!(
                    "HTTP 500: {detail} (cuda batch {} )",
                    route.endpoint
                ));
            }
            return Err(format!("HTTP {}: {}", status.as_u16(), detail));
        }

        let mut delta_adapter = ThinkingDeltaAdapter::new(&mut on_delta);
        let result = if content_type.contains("event-stream")
            || content_type.contains("text/event-stream")
        {
            log_event(
                "read_path",
                json!({ "path": "sse", "elapsedMs": started.elapsed().as_millis() }),
            );
            read_stream(
                response,
                &request.stops,
                password,
                cancel.clone(),
                &mut |delta| delta_adapter.push(delta),
            )
            .await
        } else {
            log_event(
                "read_path",
                json!({ "path": "buffered", "elapsedMs": started.elapsed().as_millis() }),
            );
            read_buffered(
                response,
                &request.stops,
                password,
                cancel.clone(),
                &mut |delta| delta_adapter.push(delta),
            )
            .await
        };
        delta_adapter.finish();
        let result = result.map(reconstruct_thinking_result);
        let result = result.and_then(|generated| {
            if generated.text.trim().is_empty() {
                Err("response contains no text".into())
            } else {
                Ok(generated)
            }
        });
        match &result {
            Ok(generated) => trace_response(
                &call_id,
                phase,
                attempt,
                started.elapsed().as_millis(),
                Ok(generated),
            ),
            Err(error) => trace_response(
                &call_id,
                phase,
                attempt,
                started.elapsed().as_millis(),
                Err(error.as_str()),
            ),
        }
        match result {
            Err(error) if is_retryable_empty_response(&error) => {
                let was_stream = stream;
                if stream {
                    stream = false;
                    log_event(
                        "empty_response_fallback_buffered",
                        json!({
                            "endpoint": route.endpoint,
                            "reason": error.clone(),
                        }),
                    );
                }
                let Some(delay) = transport_retry_delay(empty_response_retry_index) else {
                    return Err(error);
                };
                empty_response_retry_index += 1;
                log_event(
                    "empty_response_retry",
                    json!({
                        "endpoint": route.endpoint,
                        "stream": stream,
                        "previousStream": was_stream,
                        "attempt": empty_response_retry_index,
                        "delayMs": delay.as_millis(),
                    }),
                );
                tokio::select! {
                    biased;
                    _ = wait_cancel(cancel.clone()) => return Err("cancelled".into()),
                    _ = tokio::time::sleep(delay) => {}
                }
            }
            result => return result,
        }
    }
        })
        .await
}

fn is_retryable_empty_response(error: &str) -> bool {
    matches!(
        error,
        "response has no choices" | "response contains no text"
    )
}

pub async fn continue_batch(
    client: &reqwest::Client,
    route: &mut LightningRoute,
    password: &str,
    template: &ContinueRequest,
    prompts: &[String],
    cancel: watch::Receiver<bool>,
) -> Result<Vec<ContinueResult>, String> {
    let phase = infer_trace_phase(prompts.first().map(String::as_str).unwrap_or_default());
    continue_batch_with_phase(client, route, password, template, prompts, phase, cancel).await
}

pub async fn continue_batch_with_phase(
    client: &reqwest::Client,
    route: &mut LightningRoute,
    password: &str,
    template: &ContinueRequest,
    prompts: &[String],
    phase: &'static str,
    cancel: watch::Receiver<bool>,
) -> Result<Vec<ContinueResult>, String> {
    if prompts.is_empty() {
        return Err("batch prompts are required".into());
    }
    if prompts.iter().any(|prompt| prompt.trim().is_empty()) {
        return Err("batch prompt is empty".into());
    }
    if *cancel.borrow() {
        return Err("cancelled".into());
    }
    let call_id = format!("llm-{}", TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed));
    let trace_context = Arc::new(TraceContext {
        call_id: call_id.clone(),
        phase,
        attempt: AtomicU32::new(0),
    });
    ACTIVE_TRACE_CONTEXT
        .scope(Arc::clone(&trace_context), async move {
    let mut flipped = false;
    let mut stream = false;
    let mut parallel_fallback = false;
    let mut attempt = 0u32;
    loop {
        if parallel_fallback {
            log_event(
                "batch_parallel_fallback",
                json!({ "batchSize": prompts.len() }),
            );
            return continue_batch_parallel(
                client, route, password, template, prompts, phase, cancel,
            )
            .await;
        }
        let body = completion_request_body(template, prompts, route.cuda, stream, password);

        attempt += 1;
        trace_context.attempt.store(attempt, Ordering::Relaxed);
        trace_batch_request(
            &call_id,
            phase,
            attempt,
            template,
            &route.endpoint,
            stream,
            prompts,
        );
        let started = Instant::now();
        log_event(
            "batch_request",
            json!({
                "endpoint": route.endpoint,
                "cuda": route.cuda,
                "stream": stream,
                "model": template.model,
                "maxTokens": template.max_output_tokens,
                "batchSize": prompts.len(),
                "promptChars": prompts.iter().map(|prompt| prompt.chars().count()).sum::<usize>(),
            }),
        );

        let response = match send_with_transport_retries(
            client,
            &route.endpoint,
            password,
            &body,
            cancel.clone(),
            "batch_transport_retry",
            json!({"stream": stream, "batchSize": prompts.len()}),
        )
        .await
        {
            Ok(response) => response,
            Err(error) => {
                trace_batch_response(
                    &call_id,
                    phase,
                    attempt,
                    started.elapsed().as_millis(),
                    Err(error.as_str()),
                );
                return Err(error);
            }
        };

        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        log_event(
            "batch_response_headers",
            json!({
                "endpoint": route.endpoint,
                "status": status.as_u16(),
                "statusCode": status.as_u16(),
                "contentType": content_type,
                "elapsedMs": started.elapsed().as_millis(),
                "batchSize": prompts.len(),
                "stream": stream,
            }),
        );

        if !status.is_success() {
            let raw = response.text().await.unwrap_or_default();
            let detail = redact(&raw, password);
            trace_batch_response(
                &call_id,
                phase,
                attempt,
                started.elapsed().as_millis(),
                Err(detail.as_str()),
            );
            log_event(
                "batch_response_error",
                json!({
                    "status": status.as_u16(),
                    "statusCode": status.as_u16(),
                    "stream": stream,
                    "bodyPreview": preview(&detail, 240),
                    "promptChars": prompts.iter().map(|prompt| prompt.chars().count()).sum::<usize>(),
                }),
            );
            if status.as_u16() == 404 && !flipped && route.flip_endpoint() {
                flipped = true;
                log_event(
                    "endpoint_fallback",
                    json!({
                        "endpoint": route.endpoint,
                        "cuda": route.cuda,
                        "batch": true,
                    }),
                );
                continue;
            }
            if status.as_u16() == 500 && route.cuda && stream {
                stream = false;
                continue;
            }
            if status.as_u16() == 500 && prompts.len() > 1 {
                parallel_fallback = true;
                continue;
            }
            return Err(format!("HTTP {}: {}", status.as_u16(), detail));
        }

        let results = if content_type.contains("event-stream")
            || content_type.contains("text/event-stream")
        {
            read_stream_batch(response, prompts.len(), &template.stops, password, cancel).await
        } else {
            read_buffered_batch(response, prompts.len(), &template.stops, password).await
        };
        let results = match results {
            Ok(results) => results,
            Err(error) => {
                trace_batch_response(
                    &call_id,
                    phase,
                    attempt,
                    started.elapsed().as_millis(),
                    Err(error.as_str()),
                );
                return Err(error);
            }
        };
        let results = results
            .into_iter()
            .map(reconstruct_thinking_result)
            .collect::<Vec<_>>();
        trace_batch_response(
            &call_id,
            phase,
            attempt,
            started.elapsed().as_millis(),
            Ok(&results),
        );
        return Ok(results);
    }
        })
        .await
}

async fn continue_batch_parallel(
    client: &reqwest::Client,
    route: &mut LightningRoute,
    password: &str,
    template: &ContinueRequest,
    prompts: &[String],
    phase: &'static str,
    cancel: watch::Receiver<bool>,
) -> Result<Vec<ContinueResult>, String> {
    if *cancel.borrow() {
        return Err("cancelled".into());
    }
    let base_route = route.clone();
    let jobs = prompts.iter().map(|prompt| {
        let mut candidate_route = base_route.clone();
        let mut request = template.clone();
        request.prompt = prompt.clone();
        let cancel = cancel.clone();
        async move {
            let result = continue_prompt_with_phase(
                client,
                &mut candidate_route,
                password,
                request,
                phase,
                cancel,
                |_| {},
            )
            .await;
            (candidate_route, result)
        }
    });
    let outcomes = join_all(jobs).await;
    if let Some(successful_route) = outcomes
        .iter()
        .find_map(|(candidate, result)| result.is_ok().then(|| candidate.clone()))
    {
        *route = successful_route;
    }
    outcomes.into_iter().map(|(_, result)| result).collect()
}

async fn wait_cancel(mut cancel: watch::Receiver<bool>) {
    loop {
        if *cancel.borrow() {
            return;
        }
        if cancel.changed().await.is_err() {
            return;
        }
    }
}

async fn send_with_transport_retries(
    client: &reqwest::Client,
    endpoint: &str,
    password: &str,
    body: &Value,
    cancel: watch::Receiver<bool>,
    retry_event: &str,
    retry_context: Value,
) -> Result<reqwest::Response, String> {
    let mut retry_index = 0;
    loop {
        let mut http = client
            .post(endpoint)
            .header("Content-Type", "application/json");
        if !password.is_empty() {
            http = http.header("Authorization", format!("Bearer {password}"));
        }
        let result = tokio::select! {
            biased;
            _ = wait_cancel(cancel.clone()) => return Err("cancelled".into()),
            result = http.json(body).send() => result,
        };
        match result {
            Ok(response) => return Ok(response),
            Err(error) => {
                let detail = reqwest_error_detail(&error, password);
                let delay = is_retryable_transport_error(&error)
                    .then(|| transport_retry_delay(retry_index))
                    .flatten();
                let Some(delay) = delay else {
                    return Err(detail);
                };
                retry_index += 1;
                let mut fields = match retry_context.clone() {
                    Value::Object(fields) => fields,
                    _ => serde_json::Map::new(),
                };
                fields.insert("endpoint".into(), json!(endpoint));
                fields.insert("attempt".into(), json!(retry_index));
                fields.insert("delayMs".into(), json!(delay.as_millis()));
                fields.insert("error".into(), json!(detail));
                log_event(retry_event, Value::Object(fields));
                tokio::select! {
                    biased;
                    _ = wait_cancel(cancel.clone()) => return Err("cancelled".into()),
                    _ = tokio::time::sleep(delay) => {}
                }
            }
        }
    }
}

fn reqwest_error_detail(error: &reqwest::Error, password: &str) -> String {
    let mut chain = vec![error.to_string()];
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        let detail = cause.to_string();
        if chain.last() != Some(&detail) {
            chain.push(detail);
        }
        source = std::error::Error::source(cause);
    }
    redact(&chain.join(": "), password)
}

fn is_retryable_transport_error(error: &reqwest::Error) -> bool {
    if error.is_timeout() {
        return false;
    }
    if error.is_connect() {
        return true;
    }
    let mut source = std::error::Error::source(error);
    while let Some(error) = source {
        if let Some(io_error) = error.downcast_ref::<std::io::Error>() {
            return is_retryable_io_kind(io_error.kind());
        }
        source = error.source();
    }
    false
}

fn is_retryable_io_kind(kind: std::io::ErrorKind) -> bool {
    matches!(
        kind,
        std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::NotConnected
            | std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::UnexpectedEof
    )
}

fn transport_retry_delay(retry_index: usize) -> Option<Duration> {
    TRANSPORT_RETRY_DELAYS_MS
        .get(retry_index)
        .copied()
        .map(Duration::from_millis)
}

async fn emit_paced(text: &str, cancel: watch::Receiver<bool>, on_delta: &mut impl FnMut(&str)) {
    let chars: Vec<char> = text.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if *cancel.borrow() {
            return;
        }
        let end = (index + 12).min(chars.len());
        let chunk: String = chars[index..end].iter().collect();
        on_delta(&chunk);
        index = end;
        tokio::select! {
            biased;
            _ = wait_cancel(cancel.clone()) => return,
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
    }
}

struct ThinkingDeltaAdapter<'a> {
    callback: &'a mut (dyn FnMut(&str) + Send),
    pending: String,
    decided: bool,
}

impl<'a> ThinkingDeltaAdapter<'a> {
    fn new(callback: &'a mut (dyn FnMut(&str) + Send)) -> Self {
        Self {
            callback,
            pending: String::new(),
            decided: false,
        }
    }

    fn push(&mut self, delta: &str) {
        if self.decided {
            (self.callback)(delta);
            return;
        }
        self.pending.push_str(delta);
        let trimmed = self.pending.trim_start();
        if trimmed.is_empty() {
            return;
        }
        if trimmed.starts_with('>') {
            let leading_len = self.pending.len() - trimmed.len();
            let mut reconstructed =
                String::with_capacity(self.pending.len() + THINKING_DISABLED_PREFIX.len());
            reconstructed.push_str(&self.pending[..leading_len]);
            reconstructed.push_str(THINKING_DISABLED_PREFIX);
            reconstructed.push_str(trimmed);
            (self.callback)(&reconstructed);
        } else {
            (self.callback)(&self.pending);
        }
        self.pending.clear();
        self.decided = true;
    }

    fn finish(&mut self) {
        if !self.decided && !self.pending.is_empty() {
            (self.callback)(&self.pending);
            self.pending.clear();
            self.decided = true;
        }
    }
}

pub fn reconstruct_thinking_output(raw: &str) -> String {
    let trimmed = raw.trim_start();
    if trimmed.starts_with('>') {
        let leading_len = raw.len() - trimmed.len();
        let mut reconstructed = String::with_capacity(raw.len() + THINKING_DISABLED_PREFIX.len());
        reconstructed.push_str(&raw[..leading_len]);
        reconstructed.push_str(THINKING_DISABLED_PREFIX);
        reconstructed.push_str(trimmed);
        return reconstructed;
    }
    raw.to_string()
}

fn reconstruct_thinking_result(mut result: ContinueResult) -> ContinueResult {
    result.text = reconstruct_thinking_output(&result.text);
    result
}

async fn read_buffered(
    response: reqwest::Response,
    stops: &[String],
    password: &str,
    cancel: watch::Receiver<bool>,
    on_delta: &mut impl FnMut(&str),
) -> Result<ContinueResult, String> {
    let payload = response
        .text()
        .await
        .map_err(|err| redact(&err.to_string(), password))?;
    log_event(
        "buffered_body",
        json!({
            "bytes": payload.len(),
            "preview": preview(&payload, 240),
        }),
    );
    if payload.len() > MAX_RESPONSE_BYTES {
        return Err("response exceeded size limit".into());
    }
    let parsed: StreamBody =
        serde_json::from_str(&payload).map_err(|err| redact(&err.to_string(), password))?;
    if let Some(error) = parsed.error.filter(|value| !value.is_null()) {
        return Err(format!(
            "response error: {}",
            redact(&error.to_string(), password)
        ));
    }
    let mut text = String::new();
    let mut finish = FinishReason::Unknown;
    let mut found = false;
    for choice in parsed.choices.unwrap_or_default() {
        if choice.index.unwrap_or(0) != 0 {
            continue;
        }
        found = true;
        text = choice
            .message
            .as_ref()
            .and_then(|delta| delta.content.clone())
            .or_else(|| choice.delta.and_then(|delta| delta.content))
            .unwrap_or_default();
        if let Some(reason) = choice.finish_reason {
            finish = FinishReason::from_wire(&reason);
        }
        break;
    }
    if !found {
        return Err("response has no choices".into());
    }
    let (truncated, stopped) = truncate_at_stop(&text, stops);
    if stopped {
        text = truncated;
        finish = FinishReason::Stop;
    }
    if !text.is_empty() {
        log_event(
            "buffered_text",
            json!({
                "chars": text.chars().count(),
                "preview": preview(&text, 160),
            }),
        );
        emit_paced(&text, cancel, on_delta).await;
    }
    Ok(ContinueResult {
        text,
        finish_reason: finish,
    })
}

async fn read_stream(
    response: reqwest::Response,
    stops: &[String],
    password: &str,
    cancel: watch::Receiver<bool>,
    on_delta: &mut impl FnMut(&str),
) -> Result<ContinueResult, String> {
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut pending = String::new();
    let mut result = String::new();
    let mut finish = FinishReason::Unknown;
    let mut saw_choice = false;
    let mut saw_done = false;
    let mut response_bytes = 0usize;
    let mut first_delta = false;
    let started = Instant::now();
    let cancelled = Arc::new(AtomicBool::new(false));

    loop {
        if *cancel.borrow() {
            cancelled.store(true, Ordering::SeqCst);
            break;
        }
        let chunk = tokio::select! {
            biased;
            _ = wait_cancel(cancel.clone()) => {
                cancelled.store(true, Ordering::SeqCst);
                None
            }
            next = stream.next() => next,
        };
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|err| redact(&err.to_string(), password))?;
        response_bytes += chunk.len();
        if response_bytes > MAX_RESPONSE_BYTES {
            return Err("response exceeded size limit".into());
        }
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(idx) = buffer.find('\n') {
            let line = buffer[..idx].to_string();
            buffer.drain(..=idx);
            match process_single_sse_line(
                &line,
                stops,
                password,
                &mut pending,
                &mut result,
                &mut finish,
                &mut saw_choice,
                &mut first_delta,
                started,
                on_delta,
            )? {
                SingleSseLineOutcome::Continue => {}
                SingleSseLineOutcome::Done => {
                    saw_done = true;
                    break;
                }
                SingleSseLineOutcome::Stopped => {
                    return Ok(ContinueResult {
                        text: result,
                        finish_reason: FinishReason::Stop,
                    });
                }
            }
        }
        if saw_done {
            break;
        }
    }

    if !cancelled.load(Ordering::SeqCst) && !saw_done && !buffer.trim().is_empty() {
        let line = std::mem::take(&mut buffer);
        log_event(
            "sse_eof_flush",
            json!({
                "bytes": response_bytes,
                "preview": preview(line.trim(), 160),
            }),
        );
        match process_single_sse_line(
            &line,
            stops,
            password,
            &mut pending,
            &mut result,
            &mut finish,
            &mut saw_choice,
            &mut first_delta,
            started,
            on_delta,
        )? {
            SingleSseLineOutcome::Continue => {}
            SingleSseLineOutcome::Done => saw_done = true,
            SingleSseLineOutcome::Stopped => {
                return Ok(ContinueResult {
                    text: result,
                    finish_reason: FinishReason::Stop,
                });
            }
        }
    }

    if cancelled.load(Ordering::SeqCst) {
        log_event(
            "sse_cancelled",
            json!({ "elapsedMs": started.elapsed().as_millis() }),
        );
        return Err("cancelled".into());
    }
    if !pending.is_empty() {
        result.push_str(&pending);
        on_delta(&pending);
    }
    if !saw_choice && result.is_empty() {
        log_event(
            "sse_empty",
            json!({
                "elapsedMs": started.elapsed().as_millis(),
                "bytes": response_bytes,
            }),
        );
        return Err("response has no choices".into());
    }
    log_event(
        "sse_done",
        json!({
            "elapsedMs": started.elapsed().as_millis(),
            "chars": result.chars().count(),
            "bytes": response_bytes,
            "sawDone": saw_done,
            "preview": preview(&result, 120),
        }),
    );
    if saw_done && finish == FinishReason::Unknown {
        finish = FinishReason::Stop;
    }
    Ok(ContinueResult {
        text: result,
        finish_reason: finish,
    })
}

fn choice_text(choice: &StreamChoice) -> String {
    choice
        .message
        .as_ref()
        .and_then(|delta| delta.content.clone())
        .or_else(|| {
            choice
                .delta
                .as_ref()
                .and_then(|delta| delta.content.clone())
        })
        .unwrap_or_default()
}

enum SingleSseLineOutcome {
    Continue,
    Done,
    Stopped,
}

fn process_single_sse_line(
    line: &str,
    stops: &[String],
    password: &str,
    pending: &mut String,
    result: &mut String,
    finish: &mut FinishReason,
    saw_choice: &mut bool,
    first_delta: &mut bool,
    started: Instant,
    on_delta: &mut impl FnMut(&str),
) -> Result<SingleSseLineOutcome, String> {
    let parsed = match parse_sse_line(line)? {
        SseLine::Ignore => return Ok(SingleSseLineOutcome::Continue),
        SseLine::Done => return Ok(SingleSseLineOutcome::Done),
        SseLine::Json(parsed) => parsed,
    };
    if let Some(error) = parsed.error.filter(|value| !value.is_null()) {
        return Err(format!(
            "stream error: {}",
            redact(&error.to_string(), password)
        ));
    }
    for choice in parsed.choices.unwrap_or_default() {
        if choice.index.unwrap_or(0) != 0 {
            continue;
        }
        *saw_choice = true;
        if let Some(reason) = choice.finish_reason.as_deref() {
            *finish = FinishReason::from_wire(reason);
        }
        let content = choice_text(&choice);
        if content.is_empty() {
            continue;
        }
        pending.push_str(&content);
        let (text, tail, stopped) = split_at_stop(pending, stops);
        if !text.is_empty() {
            if !*first_delta {
                *first_delta = true;
                log_event(
                    "sse_first_delta",
                    json!({
                        "elapsedMs": started.elapsed().as_millis(),
                        "preview": preview(&text, 80),
                    }),
                );
            }
            result.push_str(&text);
            on_delta(&text);
        }
        *pending = tail;
        if stopped {
            return Ok(SingleSseLineOutcome::Stopped);
        }
    }
    Ok(SingleSseLineOutcome::Continue)
}

struct BatchAccumulator {
    texts: Vec<String>,
    finishes: Vec<FinishReason>,
    seen: Vec<bool>,
}

impl BatchAccumulator {
    fn new(batch_size: usize) -> Self {
        Self {
            texts: vec![String::new(); batch_size],
            finishes: vec![FinishReason::Unknown; batch_size],
            seen: vec![false; batch_size],
        }
    }

    fn push(&mut self, choice: StreamChoice) {
        let Some(raw_index) = choice.index else {
            log_event(
                "batch_choice_ignored",
                json!({
                    "reason": "missing_index",
                    "batchSize": self.texts.len(),
                }),
            );
            return;
        };
        if raw_index < 0 || raw_index as usize >= self.texts.len() {
            log_event(
                "batch_choice_ignored",
                json!({
                    "reason": "out_of_range_index",
                    "index": raw_index,
                    "batchSize": self.texts.len(),
                }),
            );
            return;
        }

        let index = raw_index as usize;
        self.seen[index] = true;
        if let Some(reason) = choice.finish_reason.as_deref() {
            self.finishes[index] = FinishReason::from_wire(reason);
        }
        self.texts[index].push_str(&choice_text(&choice));
    }

    fn finish(self, stops: &[String], path: &str) -> Result<Vec<ContinueResult>, String> {
        let missing = self
            .seen
            .iter()
            .enumerate()
            .filter_map(|(index, seen)| (!seen).then_some(index))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            log_event(
                "batch_choices_missing",
                json!({
                    "path": path,
                    "missingIndices": missing,
                    "batchSize": self.texts.len(),
                }),
            );
            return Err(format!(
                "batch response missing choice indices: {missing:?}"
            ));
        }

        let results = self
            .texts
            .into_iter()
            .zip(self.finishes)
            .map(|(text, finish)| finalize_batch_text(text, finish, stops))
            .collect::<Vec<_>>();
        log_batch_results(path, &results);
        Ok(results)
    }
}

fn finalize_batch_text(text: String, finish: FinishReason, stops: &[String]) -> ContinueResult {
    let (truncated, stopped) = truncate_at_stop(&text, stops);
    let (text, finish) = if stopped {
        (truncated, FinishReason::Stop)
    } else if finish != FinishReason::Unknown {
        (text, finish)
    } else {
        (text, finish)
    };
    ContinueResult {
        text,
        finish_reason: finish,
    }
}

fn log_batch_results(path: &str, results: &[ContinueResult]) {
    for (index, result) in results.iter().enumerate() {
        log_event(
            "batch_choice_done",
            json!({
                "path": path,
                "index": index,
                "chars": result.text.chars().count(),
                "finishReason": result.finish_reason,
                "preview": preview(&result.text, 64),
            }),
        );
    }
}

async fn read_buffered_batch(
    response: reqwest::Response,
    batch_size: usize,
    stops: &[String],
    password: &str,
) -> Result<Vec<ContinueResult>, String> {
    let payload = response
        .text()
        .await
        .map_err(|err| redact(&err.to_string(), password))?;
    log_event(
        "batch_buffered_body",
        json!({
            "bytes": payload.len(),
            "batchSize": batch_size,
        }),
    );
    if payload.len() > MAX_RESPONSE_BYTES {
        return Err("response exceeded size limit".into());
    }
    let parsed: StreamBody =
        serde_json::from_str(&payload).map_err(|err| redact(&err.to_string(), password))?;
    if let Some(error) = parsed.error.filter(|value| !value.is_null()) {
        return Err(format!(
            "response error: {}",
            redact(&error.to_string(), password)
        ));
    }
    collect_batch_choices(parsed.choices.unwrap_or_default(), batch_size, stops)
}

async fn read_stream_batch(
    response: reqwest::Response,
    batch_size: usize,
    stops: &[String],
    password: &str,
    cancel: watch::Receiver<bool>,
) -> Result<Vec<ContinueResult>, String> {
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut choices = BatchAccumulator::new(batch_size);
    let mut saw_done = false;
    let mut response_bytes = 0usize;

    loop {
        if *cancel.borrow() {
            return Err("cancelled".into());
        }
        let chunk = tokio::select! {
            biased;
            _ = wait_cancel(cancel.clone()) => return Err("cancelled".into()),
            next = stream.next() => next,
        };
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|err| redact(&err.to_string(), password))?;
        response_bytes += chunk.len();
        if response_bytes > MAX_RESPONSE_BYTES {
            return Err("response exceeded size limit".into());
        }
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(idx) = buffer.find('\n') {
            let line = buffer[..idx].to_string();
            buffer.drain(..=idx);
            match parse_sse_line(&line)? {
                SseLine::Ignore => {}
                SseLine::Done => {
                    saw_done = true;
                    break;
                }
                SseLine::Json(parsed) => {
                    if let Some(error) = parsed.error.filter(|value| !value.is_null()) {
                        return Err(format!(
                            "stream error: {}",
                            redact(&error.to_string(), password)
                        ));
                    }
                    for choice in parsed.choices.unwrap_or_default() {
                        choices.push(choice);
                    }
                }
            }
        }
        if saw_done {
            break;
        }
    }

    if !saw_done && !buffer.trim().is_empty() {
        let line = std::mem::take(&mut buffer);
        log_event(
            "batch_sse_eof_flush",
            json!({
                "bytes": response_bytes,
                "batchSize": batch_size,
                "preview": preview(line.trim(), 160),
            }),
        );
        match parse_sse_line(&line)? {
            SseLine::Ignore => {}
            SseLine::Done => {}
            SseLine::Json(parsed) => {
                if let Some(error) = parsed.error.filter(|value| !value.is_null()) {
                    return Err(format!(
                        "stream error: {}",
                        redact(&error.to_string(), password)
                    ));
                }
                for choice in parsed.choices.unwrap_or_default() {
                    choices.push(choice);
                }
            }
        }
    }

    choices.finish(stops, "sse")
}

fn collect_batch_choices(
    choices: Vec<StreamChoice>,
    batch_size: usize,
    stops: &[String],
) -> Result<Vec<ContinueResult>, String> {
    let mut collected = BatchAccumulator::new(batch_size);
    for choice in choices {
        collected.push(choice);
    }
    collected.finish(stops, "buffered")
}

fn truncate_at_stop(value: &str, stops: &[String]) -> (String, bool) {
    let mut cut = value.len();
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(index) = value.find(stop.as_str()) {
            cut = cut.min(index);
        }
    }
    if cut == value.len() {
        (value.to_string(), false)
    } else {
        (value[..cut].to_string(), true)
    }
}

fn split_at_stop(value: &str, stops: &[String]) -> (String, String, bool) {
    let mut stop_index = value.len();
    for stop in stops {
        if let Some(index) = value.find(stop.as_str()) {
            stop_index = stop_index.min(index);
        }
    }
    if stop_index < value.len() {
        return (value[..stop_index].to_string(), String::new(), true);
    }
    let mut tail_bytes = 0usize;
    for stop in stops {
        let limit = value.len().min(stop.len().saturating_sub(1));
        for length in 1..=limit {
            if length > tail_bytes && value.ends_with(&stop[..length]) {
                tail_bytes = length;
            }
        }
    }
    let safe = value.len() - tail_bytes;
    (value[..safe].to_string(), value[safe..].to_string(), false)
}

fn redact(value: &str, secret: &str) -> String {
    let mut cleaned: String = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if !secret.is_empty() {
        cleaned = cleaned.replace(secret, "[REDACTED]");
    }
    if cleaned.len() > 512 {
        cleaned.truncate(512);
        cleaned.push_str("...");
    }
    if cleaned.is_empty() {
        "empty response".into()
    } else {
        cleaned
    }
}

fn completion_request_body(
    request: &ContinueRequest,
    contents: &[String],
    cuda: bool,
    stream: bool,
    password: &str,
) -> Value {
    let contents = contents
        .iter()
        .map(|prompt| prompt_with_thinking_disabled(prompt))
        .collect::<Vec<_>>();
    json!({
        "model": request.model,
        "contents": contents,
        "max_tokens": request.max_output_tokens,
        "temperature": request.temperature.max(MIN_TEMPERATURE),
        "top_k": request.top_k,
        "top_p": request.top_p,
        "alpha_presence": request.alpha_presence,
        "alpha_frequency": request.alpha_frequency,
        "alpha_decay": request.alpha_decay,
        "stream": stream,
        "chunk_size": 1,
        "enable_think": false,
        "stop_tokens": wire_stop_tokens(cuda, &request.stops),
        "password": password,
    })
}

fn prompt_with_thinking_disabled(prompt: &str) -> String {
    const ASSISTANT_MARKER: &str = "Assistant:";

    let Some(marker_index) = prompt.rfind(ASSISTANT_MARKER) else {
        let separator = if prompt.ends_with('\n') { "\n" } else { "\n\n" };
        return format!("{prompt}{separator}{ASSISTANT_MARKER} {THINKING_DISABLED_PREFIX}");
    };
    let suffix_index = marker_index + ASSISTANT_MARKER.len();
    let suffix = visible_suffix_after_thinking(&prompt[suffix_index..]);
    if is_explicit_protocol_prefix(&suffix) {
        return prompt.to_string();
    }
    if suffix.trim().is_empty() {
        return format!("{} {THINKING_DISABLED_PREFIX}", &prompt[..suffix_index]);
    }
    let suffix = suffix.trim_start();
    let separator = if suffix.ends_with('\n') || suffix.ends_with('\r') {
        ""
    } else {
        "\n"
    };
    format!(
        "{} {suffix}{separator}{THINKING_DISABLED_PREFIX}",
        &prompt[..suffix_index],
    )
}

fn is_explicit_protocol_prefix(suffix: &str) -> bool {
    let suffix = suffix.trim_start();
    ["<tool_call", "<tool_calls", "<answer", "<route"]
        .iter()
        .any(|prefix| suffix.starts_with(prefix))
}

fn visible_suffix_after_thinking(suffix: &str) -> String {
    let mut value = suffix.trim_start().to_string();
    while let Some(index) = value.find(THINKING_DISABLED_PREFIX) {
        let mut end = index + THINKING_DISABLED_PREFIX.len();
        if value[end..].starts_with('>') {
            end += 1;
        }
        end = end.saturating_add(leading_line_break_len(&value[end..]));
        value.replace_range(index..end, "");
    }
    loop {
        let Some(index) = value.find("<think>") else {
            break;
        };
        let content_start = index + "<think>".len();
        let Some(close_offset) = value[content_start..].find("</think>") else {
            if index == 0 {
                return String::new();
            }
            break;
        };
        let mut end = content_start + close_offset + "</think>".len();
        end = end.saturating_add(leading_line_break_len(&value[end..]));
        value.replace_range(index..end, "");
    }
    if value.starts_with("<think") {
        return String::new();
    }
    value.trim_start().to_string()
}

fn leading_line_break_len(value: &str) -> usize {
    if value.starts_with("\r\n") {
        2
    } else if value.starts_with('\n') || value.starts_with('\r') {
        1
    } else {
        0
    }
}

fn wire_stop_tokens(cuda: bool, stops: &[String]) -> Value {
    if !cuda {
        return json!(stops);
    }
    let mut ids = vec![0i64];
    for stop in stops {
        let mapped: &[i64] = match stop.as_str() {
            "<tool_call>"
            | "</tool_call>"
            | "<tool_calls>"
            | "</tool_calls>"
            | "**Tool Call:**"
            | "**Tool Calls:**"
            | "</think>"
            | "\n\nUser:"
            | "\nUser:"
            | "\n\nUser: Function output:"
            | "\n\n" => &[],
            "\n```" | "```" => &[6884],
            "</s>" => &[0],
            _ => &[],
        };
        for token_id in mapped {
            if !ids.contains(token_id) {
                ids.push(*token_id);
            }
        }
    }
    json!(ids)
}

#[cfg(test)]
mod tests {
    use super::{
        alternate_endpoint, choice_text, collect_batch_choices, completion_request_body,
        completions_path, continue_prompt, continue_prompt_with_phase, install_trace_sink,
        is_retryable_io_kind, parse_sse_line, prompt_with_thinking_disabled, read_stream,
        read_stream_batch, reconstruct_thinking_output, send_with_transport_retries, split_at_stop,
        transport_retry_delay, v1_root, wire_stop_tokens, BatchAccumulator, ContinueRequest, Delta,
        LightningRoute, SseLine, StreamChoice, ThinkingDeltaAdapter,
    };
    use crate::agent::protocol::FinishReason;
    use serde_json::Value;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;
    use tokio::sync::watch;

    async fn request_sse(body: &str) -> (reqwest::Response, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind SSE test server");
        let endpoint = format!(
            "http://{}",
            listener.local_addr().expect("SSE test server address")
        );
        let body = body.to_string();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept SSE test request");
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .expect("set SSE test request timeout");
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request).expect("read SSE test request");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write SSE test response");
        });
        let response = reqwest::Client::new()
            .get(endpoint)
            .send()
            .await
            .expect("request SSE test response");
        (response, server)
    }

    #[test]
    fn sse_line_parser_accepts_done_without_data_prefix() {
        assert!(matches!(parse_sse_line("[DONE]"), Ok(SseLine::Done)));
        assert!(matches!(parse_sse_line("data: [DONE]"), Ok(SseLine::Done)));
    }

    #[test]
    fn single_choice_text_reads_message_then_delta_content() {
        let message = StreamChoice {
            index: Some(0),
            delta: None,
            message: Some(Delta {
                content: Some("message".into()),
            }),
            finish_reason: None,
        };
        let delta = StreamChoice {
            index: Some(0),
            delta: Some(Delta {
                content: Some("delta".into()),
            }),
            message: None,
            finish_reason: None,
        };
        assert_eq!(choice_text(&message), "message");
        assert_eq!(choice_text(&delta), "delta");
    }

    #[tokio::test]
    async fn sse_eof_flushes_unterminated_message_content() {
        let (response, server) = request_sse(
            r#"data: {"choices":[{"index":0,"message":{"content":"message-content"},"finish_reason":"stop"}]}"#,
        )
        .await;
        let (_cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let mut streamed = String::new();
        let result = read_stream(response, &[], "", cancel_rx, &mut |delta| {
            streamed.push_str(delta)
        })
        .await
        .expect("unterminated SSE data should flush at EOF");
        server.join().expect("join SSE test server");

        assert_eq!(result.text, "message-content");
        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert_eq!(streamed, "message-content");
    }

    #[tokio::test]
    async fn sse_done_only_without_newline_is_handled_as_empty_response() {
        let (response, server) = request_sse("data: [DONE]").await;
        let (_cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let mut streamed = String::new();
        let error = read_stream(response, &[], "", cancel_rx, &mut |delta| {
            streamed.push_str(delta)
        })
        .await
        .expect_err("DONE without a choice should remain an empty response");
        server.join().expect("join SSE test server");

        assert_eq!(error, "response has no choices");
        assert!(streamed.is_empty());
    }

    #[tokio::test]
    async fn batch_sse_eof_flushes_unterminated_data() {
        let (response, server) = request_sse(
            r#"data: {"choices":[{"index":0,"message":{"content":"batch-content"},"finish_reason":"stop"}]}"#,
        )
        .await;
        let (_cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let results = read_stream_batch(response, 1, &[], "", cancel_rx)
            .await
            .expect("batch unterminated SSE data should flush at EOF");
        server.join().expect("join SSE test server");

        assert_eq!(results[0].text, "batch-content");
        assert_eq!(results[0].finish_reason, FinishReason::Stop);
    }

    #[test]
    fn splits_on_tool_close() {
        let (text, tail, stopped) = split_at_stop(
            r#"{"name":"read_file","arguments":{"path":"README.md"}}</tool_call> leftover"#,
            &["</tool_call>".into()],
        );
        assert!(stopped);
        assert!(text.ends_with("README.md\"}}"));
        assert!(tail.is_empty());
    }

    #[test]
    fn holds_partial_stop_suffix() {
        let (text, tail, stopped) = split_at_stop("hello</tool", &["</tool_call>".into()]);
        assert!(!stopped);
        assert_eq!(text, "hello");
        assert_eq!(tail, "</tool");
    }

    #[test]
    fn lightning_uses_raw_chat_completions() {
        let root = v1_root("http://192.168.0.115:8030/v1");
        assert_eq!(root, "http://192.168.0.115:8030/v1");
        assert_eq!(
            completions_path(&root, false),
            "http://192.168.0.115:8030/v1/chat/completions"
        );
        assert_eq!(
            completions_path(&root, true),
            "http://192.168.0.115:8030/v1/batch/completions"
        );
    }

    #[tokio::test]
    async fn explicit_phase_is_preserved_in_model_trace() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind phase test server");
        let endpoint = format!(
            "http://{}",
            listener.local_addr().expect("phase test server address")
        );
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept phase test request");
            let mut request = [0u8; 8192];
            let _ = stream.read(&mut request).expect("read phase test request");
            let body = r#"{"choices":[{"index":0,"message":{"content":"phase-ok"},"finish_reason":"stop"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write phase test response");
        });

        let traces = Arc::new(Mutex::new(Vec::<Value>::new()));
        let captured = Arc::clone(&traces);
        let _guard = install_trace_sink(Arc::new(move |payload| {
            captured.lock().expect("phase trace lock").push(payload);
        }));
        let mut route = LightningRoute {
            endpoint,
            cuda: false,
        };
        let request = ContinueRequest {
            prompt: "route marker-free prompt".into(),
            model: "test-model".into(),
            max_output_tokens: 32,
            stops: vec![],
            temperature: 0.001,
            top_k: 50,
            top_p: 1.0,
            alpha_presence: 0.0,
            alpha_frequency: 0.0,
            alpha_decay: 0.99,
        };
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let result = continue_prompt_with_phase(
            &reqwest::Client::new(),
            &mut route,
            "",
            request,
            "intent",
            cancel_rx,
            |_| {},
        )
        .await
        .expect("phase test completion");
        assert_eq!(result.text, "phase-ok");

        let traces = traces.lock().expect("phase trace lock");
        let test_call_id = traces
            .iter()
            .find(|payload| {
                payload.get("kind").and_then(Value::as_str) == Some("model_request")
                    && payload.pointer("/request/model").and_then(Value::as_str)
                        == Some("test-model")
            })
            .and_then(|payload| payload.get("callId").and_then(Value::as_str))
            .expect("phase test model request")
            .to_string();
        let model_requests = traces
            .iter()
            .filter(|payload| {
                payload.get("kind").and_then(Value::as_str) == Some("model_request")
                    && payload.get("callId").and_then(Value::as_str) == Some(test_call_id.as_str())
            })
            .collect::<Vec<_>>();
        let model_responses = traces
            .iter()
            .filter(|payload| {
                payload.get("kind").and_then(Value::as_str) == Some("model_response")
                    && payload.get("callId").and_then(Value::as_str) == Some(test_call_id.as_str())
            })
            .collect::<Vec<_>>();
        assert_eq!(model_requests.len(), 1);
        assert_eq!(model_responses.len(), 1);
        assert_eq!(model_requests[0]["phase"], "intent");
        assert_eq!(model_responses[0]["phase"], "intent");
        let transport = traces
            .iter()
            .filter(|payload| {
                payload.get("kind").and_then(Value::as_str) == Some("transport")
                    && payload.get("callId").and_then(Value::as_str) == Some(test_call_id.as_str())
            })
            .collect::<Vec<_>>();
        assert!(!transport.is_empty());
        assert!(transport.iter().all(|payload| {
            payload.get("phase").and_then(Value::as_str) == Some("intent")
                && payload.get("attempt").and_then(Value::as_u64) == Some(1)
        }));
        drop(traces);
        drop(_guard);
        server.join().expect("join phase test server");
    }

    #[test]
    fn transport_retry_schedule_is_bounded() {
        assert_eq!(
            transport_retry_delay(0).map(|delay| delay.as_millis()),
            Some(250)
        );
        assert_eq!(
            transport_retry_delay(1).map(|delay| delay.as_millis()),
            Some(750)
        );
        assert_eq!(
            transport_retry_delay(2).map(|delay| delay.as_millis()),
            Some(1_500)
        );
        assert_eq!(transport_retry_delay(3), None);
    }

    #[test]
    fn transport_retry_only_accepts_transient_disconnects() {
        assert!(is_retryable_io_kind(std::io::ErrorKind::ConnectionRefused));
        assert!(is_retryable_io_kind(std::io::ErrorKind::ConnectionReset));
        assert!(is_retryable_io_kind(std::io::ErrorKind::BrokenPipe));
        assert!(!is_retryable_io_kind(std::io::ErrorKind::TimedOut));
        assert!(!is_retryable_io_kind(std::io::ErrorKind::InvalidData));
    }

    #[tokio::test]
    async fn transport_retry_backoff_is_cancellable() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind unused local port");
        let endpoint = format!("http://{}", listener.local_addr().expect("local address"));
        drop(listener);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let _ = cancel_tx.send(true);
        });

        let started = std::time::Instant::now();
        let result = send_with_transport_retries(
            &reqwest::Client::new(),
            &endpoint,
            "",
            &serde_json::json!({"contents": ["test"]}),
            cancel_rx,
            "test_transport_retry",
            serde_json::json!({}),
        )
        .await;

        assert_eq!(
            result.expect_err("request should be cancelled"),
            "cancelled"
        );
        assert!(started.elapsed() < std::time::Duration::from_millis(500));
    }

    #[tokio::test]
    async fn empty_stream_falls_back_to_buffered_response() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let endpoint = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().expect("test server address")
        );
        let server = std::thread::spawn(move || {
            let responses = [
                (
                    "text/event-stream",
                    "data: [DONE]\n\n".to_string(),
                ),
                (
                    "application/json",
                    "{\"choices\":[{\"index\":0,\"message\":{\"content\":\"planned\"},\"finish_reason\":\"stop\"}]}".to_string(),
                ),
            ];
            let mut request_bodies = Vec::new();
            for (content_type, response_body) in responses {
                let (mut stream, _) = listener.accept().expect("accept test request");
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .expect("set test request timeout");
                let mut reader = BufReader::new(stream.try_clone().expect("clone test stream"));
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).expect("read request header");
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line
                        .to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(str::trim)
                    {
                        content_length = value.parse().expect("parse content length");
                    }
                }
                let mut body = vec![0u8; content_length];
                reader.read_exact(&mut body).expect("read request body");
                request_bodies.push(body);

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write test response");
            }
            request_bodies
        });

        let request = ContinueRequest {
            prompt: "User: inspect dynamically\n\nAssistant: ```json\n".into(),
            model: "rwkv-test".into(),
            max_output_tokens: 64,
            stops: vec!["\n```".into(), "\n\nUser:".into(), "</s>".into()],
            temperature: 0.5,
            top_k: 10,
            top_p: 0.75,
            alpha_presence: 0.25,
            alpha_frequency: 0.5,
            alpha_decay: 0.75,
        };
        let mut route = LightningRoute {
            endpoint,
            cuda: false,
        };
        let (_cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let mut streamed = String::new();
        let generated = continue_prompt(
            &reqwest::Client::new(),
            &mut route,
            "",
            request,
            cancel_rx,
            |delta| streamed.push_str(delta),
        )
        .await
        .expect("retry empty completions");

        assert_eq!(generated.text, "planned");
        assert_eq!(generated.finish_reason, FinishReason::Stop);
        assert_eq!(streamed, "planned");
        let request_bodies = server.join().expect("join test server");
        assert_eq!(request_bodies.len(), 2);
        let first: Value = serde_json::from_slice(&request_bodies[0]).expect("first request JSON");
        let second: Value =
            serde_json::from_slice(&request_bodies[1]).expect("second request JSON");
        assert_eq!(first["stream"], Value::Bool(true));
        assert_eq!(second["stream"], Value::Bool(false));
        assert_eq!(first["contents"], second["contents"]);
    }

    #[test]
    fn flips_404_between_batch_and_chat() {
        assert_eq!(
            alternate_endpoint("http://host/v1/batch/completions").as_deref(),
            Some("http://host/v1/chat/completions")
        );
        assert_eq!(
            alternate_endpoint("http://host/v1/chat/completions").as_deref(),
            Some("http://host/v1/batch/completions")
        );
        assert_eq!(
            v1_root("http://host/openai/v1/chat/completions"),
            "http://host/v1"
        );
    }

    #[test]
    fn stop_tokens_match_each_lightning_backend() {
        let stops = vec!["\n```".into(), "\n\nUser:".into(), "</s>".into()];
        assert_eq!(wire_stop_tokens(false, &stops), serde_json::json!(stops));
        assert_eq!(wire_stop_tokens(true, &stops), serde_json::json!([0, 6884]));
        let tool_marker = vec!["<tool_call>".into()];
        assert_eq!(wire_stop_tokens(true, &tool_marker), serde_json::json!([0]));
    }

    #[test]
    fn cuda_boundary_stops_are_client_side_sequences() {
        assert_eq!(
            wire_stop_tokens(true, &["\n\nUser:".into(), "\nUser:".into()]),
            serde_json::json!([0])
        );
        assert_eq!(
            wire_stop_tokens(true, &["\n\nUser: Function output:".into()]),
            serde_json::json!([0])
        );
        assert_eq!(
            wire_stop_tokens(true, &["\n```".into(), "\n\nUser:".into()]),
            serde_json::json!([0, 6884])
        );
    }

    #[test]
    fn request_body_carries_contents_and_backend_stop_tokens() {
        let request = ContinueRequest {
            prompt: "unused single prompt".into(),
            model: "rwkv-test".into(),
            max_output_tokens: 64,
            stops: vec!["\n```".into(), "\n\nUser:".into(), "</s>".into()],
            temperature: 0.5,
            top_k: 10,
            top_p: 0.75,
            alpha_presence: 0.25,
            alpha_frequency: 0.5,
            alpha_decay: 0.75,
        };
        let contents = vec![
            "User: first prompt\n\nAssistant:".into(),
            "User: second prompt\n\nAssistant: ```json\n".into(),
        ];

        let raw = completion_request_body(&request, &contents, false, false, "secret");
        assert_eq!(
            raw,
            serde_json::json!({
                "model": "rwkv-test",
                "contents": [
                    "User: first prompt\n\nAssistant: <think></think",
                    "User: second prompt\n\nAssistant: ```json\n<think></think"
                ],
                "max_tokens": 64,
                "temperature": 0.5,
                "top_k": 10,
                "top_p": 0.75,
                "alpha_presence": 0.25,
                "alpha_frequency": 0.5,
                "alpha_decay": 0.75,
                "stream": false,
                "chunk_size": 1,
                "enable_think": false,
                "stop_tokens": ["\n```", "\n\nUser:", "</s>"],
                "password": "secret",
            })
        );

        let cuda = completion_request_body(&request, &contents, true, true, "secret");
        assert_eq!(
            cuda["contents"],
            serde_json::json!([
                "User: first prompt\n\nAssistant: <think></think",
                "User: second prompt\n\nAssistant: ```json\n<think></think"
            ])
        );
        assert_eq!(cuda["enable_think"], serde_json::json!(false));
        assert_eq!(cuda["stop_tokens"], serde_json::json!([0, 6884]));
    }

    #[test]
    fn raw_prompts_disable_thinking_at_the_final_token_boundary() {
        assert_eq!(
            prompt_with_thinking_disabled("User: hello\n\nAssistant:"),
            "User: hello\n\nAssistant: <think></think"
        );
        assert_eq!(
            prompt_with_thinking_disabled("User: inspect\n\nAssistant: ```json\n"),
            "User: inspect\n\nAssistant: ```json\n<think></think"
        );
        assert_eq!(
            prompt_with_thinking_disabled(
                "User: hello\n\nAssistant: <think>\nold reasoning\n</think>\nanswer:"
            ),
            "User: hello\n\nAssistant: answer:\n<think></think"
        );
        assert_eq!(
            prompt_with_thinking_disabled("User: hello\n\nAssistant: <think></think"),
            "User: hello\n\nAssistant: <think></think"
        );
        assert_eq!(
            prompt_with_thinking_disabled("User: inspect\n\nAssistant: <think></think```json\n"),
            "User: inspect\n\nAssistant: ```json\n<think></think"
        );
        for unsafe_prompt in [
            "User: hello\n\nAssistant: <think",
            "User: hello\n\nAssistant: <think>",
            "User: hello\n\nAssistant: <think>unfinished reasoning",
        ] {
            assert_eq!(
                prompt_with_thinking_disabled(unsafe_prompt),
                "User: hello\n\nAssistant: <think></think"
            );
        }
        assert_eq!(
            prompt_with_thinking_disabled("User: hello"),
            "User: hello\n\nAssistant: <think></think"
        );
        assert_eq!(
            prompt_with_thinking_disabled("User: inspect\n\nAssistant: <tool_call>"),
            "User: inspect\n\nAssistant: <tool_call>"
        );
        assert_eq!(
            prompt_with_thinking_disabled("User: answer\n\nAssistant: <answer>"),
            "User: answer\n\nAssistant: <answer>"
        );
        assert_eq!(
            prompt_with_thinking_disabled("User: inspect\n\nAssistant: ```json\n<think></think"),
            "User: inspect\n\nAssistant: ```json\n<think></think"
        );
        assert_eq!(
            prompt_with_thinking_disabled("User: inspect\n\nAssistant: ```json\n<think></think>\n"),
            "User: inspect\n\nAssistant: ```json\n<think></think"
        );
        assert_eq!(
            prompt_with_thinking_disabled("User: inspect\n\nAssistant: <think></think><tool_call>"),
            "User: inspect\n\nAssistant: <think></think><tool_call>"
        );
    }

    #[test]
    fn reconstructs_only_the_withheld_think_close_and_preserves_plain_text() {
        assert_eq!(
            reconstruct_thinking_output(">\n{\"function\":\"list_files\"}"),
            "<think></think>\n{\"function\":\"list_files\"}"
        );
        assert_eq!(
            reconstruct_thinking_output("  >\n{\"name\":\"list_files\"}"),
            "  <think></think>\n{\"name\":\"list_files\"}"
        );
        assert_eq!(reconstruct_thinking_output("plain answer"), "plain answer");
        assert_eq!(
            reconstruct_thinking_output("<think></think>plain answer"),
            "<think></think>plain answer"
        );
    }

    #[test]
    fn streaming_reconstruction_hides_withheld_marker_from_consumers() {
        let mut streamed = String::new();
        let mut callback = |delta: &str| streamed.push_str(delta);
        let mut adapter = ThinkingDeltaAdapter::new(&mut callback);
        adapter.push("  ");
        adapter.push(">\n{");
        adapter.push("\"function\":\"list_files\"}");
        adapter.finish();
        assert_eq!(streamed, "  <think></think>\n{\"function\":\"list_files\"}");
    }

    #[test]
    fn batch_choices_are_demultiplexed_by_out_of_order_indices() {
        let choice = |index, text: &str| StreamChoice {
            index: Some(index),
            delta: None,
            message: Some(Delta {
                content: Some(text.into()),
            }),
            finish_reason: Some("stop".into()),
        };
        let results = collect_batch_choices(
            vec![choice(2, "third"), choice(0, "first"), choice(1, "second")],
            3,
            &[],
        )
        .expect("collect indexed choices");

        assert_eq!(
            results
                .into_iter()
                .map(|result| result.text)
                .collect::<Vec<_>>(),
            ["first", "second", "third"]
        );
    }

    #[test]
    fn batch_stream_chunks_are_accumulated_by_index() {
        let delta = |index, text: &str, finish_reason: Option<&str>| StreamChoice {
            index: Some(index),
            delta: Some(Delta {
                content: Some(text.into()),
            }),
            message: None,
            finish_reason: finish_reason.map(str::to_string),
        };
        let mut choices = BatchAccumulator::new(2);
        choices.push(delta(1, "second-", None));
        choices.push(delta(0, "first-", None));
        choices.push(delta(1, "done", Some("stop")));
        choices.push(delta(0, "done", Some("length")));

        let results = choices.finish(&[], "test").expect("collect SSE chunks");
        assert_eq!(results[0].text, "first-done");
        assert_eq!(results[0].finish_reason, FinishReason::Length);
        assert_eq!(results[1].text, "second-done");
        assert_eq!(results[1].finish_reason, FinishReason::Stop);
    }

    #[test]
    fn batch_choices_reject_missing_expected_index() {
        let results = collect_batch_choices(
            vec![StreamChoice {
                index: Some(1),
                delta: None,
                message: Some(Delta {
                    content: Some("second".into()),
                }),
                finish_reason: Some("stop".into()),
            }],
            3,
            &[],
        );

        assert_eq!(
            results.expect_err("missing indices must fail"),
            "batch response missing choice indices: [0, 2]"
        );
    }

    #[test]
    fn buffered_batch_preserves_each_finish_reason() {
        let choice = |index, text: &str, finish_reason: &str| StreamChoice {
            index: Some(index),
            delta: None,
            message: Some(Delta {
                content: Some(text.into()),
            }),
            finish_reason: Some(finish_reason.into()),
        };
        let results = collect_batch_choices(
            vec![choice(1, "second", "stop"), choice(0, "first", "length")],
            2,
            &[],
        )
        .expect("collect buffered choices");

        assert_eq!(results[0].finish_reason, FinishReason::Length);
        assert_eq!(results[1].finish_reason, FinishReason::Stop);
    }
}
