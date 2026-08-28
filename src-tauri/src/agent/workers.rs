use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::watch;

use super::config::Settings;
use super::lightning::{self, ContinueRequest};
use super::protocol::{self, ActionKind, FinishReason, GenerationStage, Message, Role};
use super::renderer;
use super::runner::AgentEvent;
use super::tools::ToolSet;

const MAX_PARALLEL_WAVES: usize = 8;
const MAX_QUEUE_FILES: usize = 24;
const WORKER_MAX_TOKENS: u32 = 512;
const ROUND_MAX_TOKENS: u32 = 384;
const FILE_EXCERPT_CHARS: usize = 4000;
const NOTE_CHARS: usize = 900;

#[derive(Debug, Clone, Serialize)]
pub struct WorkerToolTrace {
    pub name: String,
    pub ok: bool,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub round: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerReport {
    pub index: u32,
    pub goal: String,
    pub ok: bool,
    pub status: String,
    pub output: String,
    pub tools: Vec<String>,
    pub traces: Vec<WorkerToolTrace>,
}

#[derive(Debug, Clone)]
pub struct RoundSection {
    pub title: String,
    pub body: String,
}

pub struct Investigation {
    pub reports: Vec<WorkerReport>,
    pub rounds: Vec<RoundSection>,
}

impl Investigation {
    pub fn briefing(&self) -> String {
        if self.rounds.is_empty() {
            return briefing(&self.reports);
        }
        let mut out = String::from("Round briefings:\n");
        for round in &self.rounds {
            out.push_str(&format!("\n## {}\n{}\n", round.title, round.body.trim()));
        }
        out
    }

    pub fn synthesis_title(&self) -> String {
        format!("第 {} 轮 · 综合", self.rounds.len() + 1)
    }
}

struct Worker {
    goal: String,
    notes: String,
    pending_reads: Vec<String>,
    current_path: Option<String>,
    current_excerpt: Option<String>,
    done: bool,
    ok: bool,
    output: String,
    tools: Vec<String>,
    traces: Vec<WorkerToolTrace>,
}

pub async fn run_workers(
    client: &reqwest::Client,
    settings: &Settings,
    server: &mut lightning::LightningRoute,
    tools: &ToolSet,
    goals: Vec<String>,
    task: &str,
    parent_step: u32,
    cancel: watch::Receiver<bool>,
    emit: &impl Fn(AgentEvent),
) -> Result<Investigation, String> {
    let thinking_fast = settings.thinking_fast();
    let mut workers: Vec<Worker> = goals
        .into_iter()
        .map(|goal| Worker {
            notes: String::new(),
            pending_reads: Vec::new(),
            current_path: None,
            current_excerpt: None,
            done: false,
            ok: false,
            output: String::new(),
            tools: Vec::new(),
            traces: Vec::new(),
            goal,
        })
        .collect();

    for (index, worker) in workers.iter().enumerate() {
        emit(AgentEvent::Subagent {
            step: parent_step,
            index: index as u32,
            goal: worker.goal.clone(),
            status: "running".into(),
            output: None,
        });
    }

    let mut rounds = Vec::new();
    let intent = intent_brief(task, &workers);
    publish_round(
        &mut workers,
        &mut rounds,
        parent_step,
        1,
        "第 1 轮 · 意图识别",
        &intent,
        true,
        emit,
    );

    let (list_ok, list_payload) = execute_tool(
        tools,
        "list_files",
        json!({ "path": ".", "max_depth": 4, "max_results": 80 }),
    );
    let list_summary = summarize_tool("list_files", list_ok, &list_payload);
    for (index, worker) in workers.iter_mut().enumerate() {
        worker.tools.push("list_files".into());
        worker.traces.push(WorkerToolTrace {
            name: "list_files".into(),
            ok: list_ok,
            summary: list_summary.clone(),
            round: None,
        });
        emit(AgentEvent::SubagentTool {
            step: parent_step,
            index: index as u32,
            tool: "list_files".into(),
            ok: Some(list_ok),
            summary: Some(list_summary.clone()),
            round: None,
        });
    }

    let mut queue = listing_files(&list_payload);
    let mentioned: Vec<String> = workers
        .iter()
        .flat_map(|worker| files_mentioned(&worker.goal, &queue))
        .collect();
    for path in mentioned.into_iter().rev() {
        if let Some(index) = queue.iter().position(|item| item == &path) {
            queue.remove(index);
            queue.insert(0, path);
        }
    }
    let lanes = workers.len().max(1);
    top_up_queue(tools, &mut queue, MAX_QUEUE_FILES);
    queue.truncate(MAX_QUEUE_FILES);

    let digest_template = ContinueRequest {
        prompt: String::new(),
        model: settings.model.clone(),
        max_output_tokens: WORKER_MAX_TOKENS.min(settings.max_tokens).max(1),
        stops: protocol::stops(GenerationStage::Answer),
        temperature: settings.temperature,
        top_k: settings.top_k,
        top_p: settings.top_p,
        alpha_presence: settings.alpha_presence,
        alpha_frequency: settings.alpha_frequency,
        alpha_decay: settings.alpha_decay,
    };

    let mut stage: u32 = 2;
    let mut wave = 0usize;
    while !queue.is_empty() && wave < MAX_PARALLEL_WAVES {
        if *cancel.borrow() {
            return Err("cancelled".into());
        }
        wave += 1;
        let mut batch = Vec::new();
        for index in 0..lanes {
            if workers[index].done {
                continue;
            }
            let Some(path) = queue.first().cloned() else {
                break;
            };
            queue.remove(0);
            batch.push((index, path));
        }
        if batch.is_empty() {
            break;
        }

        let read_round = stage;
        stage += 1;
        let names: Vec<&str> = batch.iter().map(|(_, path)| path.as_str()).collect();
        publish_round(
            &mut workers,
            &mut rounds,
            parent_step,
            read_round,
            &format!("第 {read_round} 轮 · 并行阅读"),
            &format!("{} 路同时阅读：{}", batch.len(), names.join("、")),
            true,
            emit,
        );
        emit(AgentEvent::Status {
            text: format!("第 {read_round} 轮 · {} 路并行", batch.len()),
        });

        let mut failed_bits: Vec<(String, String)> = Vec::new();
        for (index, path) in &batch {
            emit(AgentEvent::SubagentTool {
                step: parent_step,
                index: *index as u32,
                tool: "read_file".into(),
                ok: None,
                summary: Some(format!("第 {read_round} 轮 · {path}")),
                round: Some(read_round),
            });
            let (ok, mut payload) = execute_tool(tools, "read_file", json!({ "path": path }));
            let summary = format!(
                "第 {read_round} 轮 · {}",
                summarize_tool("read_file", ok, &payload)
            );
            workers[*index].traces.push(WorkerToolTrace {
                name: "read_file".into(),
                ok,
                summary: summary.clone(),
                round: Some(read_round),
            });
            if ok {
                workers[*index].tools.push("read_file".into());
            }
            emit(AgentEvent::SubagentTool {
                step: parent_step,
                index: *index as u32,
                tool: "read_file".into(),
                ok: Some(ok),
                summary: Some(summary),
                round: Some(read_round),
            });
            compact_worker_payload(&mut payload);
            if ok {
                let excerpt = payload
                    .pointer("/result/content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                workers[*index].current_path = Some(path.clone());
                workers[*index].current_excerpt = Some(excerpt);
            } else {
                let error = payload
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("read failed")
                    .to_string();
                append_note(&mut workers[*index], path, &error);
                failed_bits.push((path.clone(), error));
            }
        }

        let mut batch_index = Vec::new();
        let mut prompts = Vec::new();
        for (index, _) in &batch {
            if workers[*index].current_excerpt.is_none() {
                continue;
            }
            let prompt = digest_prompt(&workers[*index], thinking_fast)?;
            batch_index.push(*index);
            prompts.push(prompt);
        }

        let mut wave_bits = failed_bits;
        if !prompts.is_empty() {
            match lightning::continue_batch_with_phase(
                client,
                server,
                &settings.password,
                &digest_template,
                &prompts,
                "chunk_summary",
                cancel.clone(),
            )
            .await
            {
                Ok(generated) => {
                    for (slot, worker_index) in batch_index.into_iter().enumerate() {
                        let worker = &mut workers[worker_index];
                        let path = worker.current_path.clone().unwrap_or_else(|| "file".into());
                        let digest = generated
                            .get(slot)
                            .map(|result| {
                                digest_text(&result.text, thinking_fast, result.finish_reason)
                            })
                            .unwrap_or_default();
                        let body = if digest.trim().is_empty() {
                            clip(worker.current_excerpt.as_deref().unwrap_or(""), 600)
                        } else {
                            clip(&digest, NOTE_CHARS)
                        };
                        append_note(worker, &path, &body);
                        wave_bits.push((path.clone(), body.clone()));
                        worker.traces.push(WorkerToolTrace {
                            name: "notes".into(),
                            ok: true,
                            summary: format!("第 {read_round} 轮 · {}", clip(&body, 80)),
                            round: Some(read_round),
                        });
                        emit(AgentEvent::SubagentTool {
                            step: parent_step,
                            index: worker_index as u32,
                            tool: "notes".into(),
                            ok: Some(true),
                            summary: Some(format!("第 {read_round} 轮 · {}", clip(&body, 80))),
                            round: Some(read_round),
                        });
                        worker.current_path = None;
                        worker.current_excerpt = None;
                    }
                }
                Err(_) => {
                    for index in &batch_index {
                        let worker = &mut workers[*index];
                        let path = worker.current_path.clone().unwrap_or_else(|| "file".into());
                        let salvage = clip(worker.current_excerpt.as_deref().unwrap_or(""), 600);
                        append_note(worker, &path, &salvage);
                        wave_bits.push((path, salvage));
                        worker.current_path = None;
                        worker.current_excerpt = None;
                    }
                }
            }
        }

        let sum_round = stage;
        stage += 1;
        let summary = if wave_bits.is_empty() {
            "本轮没有读到可用文本。".into()
        } else {
            write_round_summary(
                client,
                server,
                settings,
                sum_round,
                &wave_bits,
                thinking_fast,
                cancel.clone(),
                emit,
            )
            .await?
        };
        publish_round(
            &mut workers,
            &mut rounds,
            parent_step,
            sum_round,
            &format!("第 {sum_round} 轮 · 总结"),
            &summary,
            true,
            emit,
        );
    }

    if !queue.is_empty() {
        let n = workers.len().max(1);
        for (offset, path) in queue.into_iter().enumerate() {
            workers[offset % n].pending_reads.push(path);
        }
    }

    for (index, worker) in workers.iter_mut().enumerate() {
        if !worker.done {
            finish_worker(
                worker,
                !worker.notes.trim().is_empty(),
                final_output(worker),
                parent_step,
                index,
                emit,
            );
        }
    }

    Ok(Investigation {
        reports: workers
            .into_iter()
            .enumerate()
            .map(|(index, worker)| WorkerReport {
                index: index as u32,
                goal: worker.goal,
                ok: worker.ok,
                status: if worker.ok {
                    "done".into()
                } else {
                    "error".into()
                },
                output: worker.output,
                tools: worker.tools,
                traces: worker.traces,
            })
            .collect(),
        rounds,
    })
}

fn intent_brief(task: &str, workers: &[Worker]) -> String {
    let mut out = format!("用户意图：{}\n", task.trim());
    out.push_str("\n并行子任务：\n");
    for (index, worker) in workers.iter().enumerate() {
        out.push_str(&format!("{}. {}\n", index + 1, worker.goal));
    }
    out.push_str(
        "\n计划：按当前问题分轮并行阅读，每轮读完立刻总结；还有未读文件就继续下一轮，全部结束后再综合。",
    );
    out
}

fn publish_round(
    workers: &mut [Worker],
    rounds: &mut Vec<RoundSection>,
    parent_step: u32,
    round: u32,
    title: &str,
    body: &str,
    stream: bool,
    emit: &impl Fn(AgentEvent),
) {
    if stream {
        emit(AgentEvent::Token {
            text: format!("\n\n## {title}\n\n{}\n", body.trim()),
        });
    }
    rounds.push(RoundSection {
        title: title.to_string(),
        body: body.trim().to_string(),
    });
    for (index, worker) in workers.iter_mut().enumerate() {
        worker.traces.push(WorkerToolTrace {
            name: "round".into(),
            ok: true,
            summary: title.to_string(),
            round: Some(round),
        });
        emit(AgentEvent::SubagentTool {
            step: parent_step,
            index: index as u32,
            tool: "round".into(),
            ok: Some(true),
            summary: Some(title.to_string()),
            round: Some(round),
        });
    }
}

fn top_up_queue(tools: &ToolSet, queue: &mut Vec<String>, cap: usize) {
    if queue.len() >= cap {
        return;
    }
    for path in ["tests", "src", "scripts", "lib", "agent_cases"] {
        if queue.len() >= cap {
            break;
        }
        let depth = if path == "agent_cases" { 1 } else { 2 };
        let (ok, payload) = execute_tool(
            tools,
            "list_files",
            json!({
                "path": path,
                "max_depth": depth,
                "max_results": 40,
            }),
        );
        if !ok {
            continue;
        }
        for file in listing_files(&payload) {
            if looks_binary(&file) || queue.iter().any(|seen| seen == &file) {
                continue;
            }
            queue.push(file);
            if queue.len() >= cap {
                break;
            }
        }
    }
}

fn digest_prompt(worker: &Worker, thinking_fast: bool) -> Result<String, String> {
    let path = worker.current_path.as_deref().unwrap_or("file");
    let excerpt = worker.current_excerpt.as_deref().unwrap_or("").trim();
    let excerpt = if excerpt.is_empty() {
        "(empty file)"
    } else {
        excerpt
    };
    let messages = vec![
        Message {
            role: Role::System,
            content: "Write 4-8 short Chinese bullets about this file. Factual. No tools. No JSON."
                .into(),
        },
        Message {
            role: Role::User,
            content: format!("FILE: {path}\n\n{excerpt}"),
        },
    ];
    renderer::render(&messages, thinking_fast)
}

fn digest_text(raw: &str, thinking_fast: bool, finish: FinishReason) -> String {
    let reconstructed = renderer::reconstruct_output(raw, thinking_fast);
    let text = match protocol::parse_action(&reconstructed, finish) {
        Ok(action) if action.kind == ActionKind::Final => action.content,
        Ok(action) if action.kind == ActionKind::Tool => String::new(),
        Ok(action) if !action.content.trim().is_empty() => action.content,
        _ => reconstructed,
    };
    strip_toolish(&text)
}

async fn write_round_summary(
    client: &reqwest::Client,
    server: &mut lightning::LightningRoute,
    settings: &Settings,
    round: u32,
    bits: &[(String, String)],
    thinking_fast: bool,
    cancel: watch::Receiver<bool>,
    emit: &impl Fn(AgentEvent),
) -> Result<String, String> {
    emit(AgentEvent::Status {
        text: format!("第 {round} 轮总结"),
    });
    let names: Vec<&str> = bits.iter().map(|(path, _)| path.as_str()).collect();
    let mut notes = String::new();
    for (path, body) in bits {
        notes.push_str(&format!("## {path}\n{}\n\n", clip(body, 700)));
    }
    let fallback = compose_round_from_bits(bits);
    let messages = vec![
        Message {
            role: Role::System,
            content: "Write short Chinese bullets about these files. Factual. No tools. No JSON. Do not mention round numbers."
                .into(),
        },
        Message {
            role: Role::User,
            content: format!(
                "并行阅读：{}\n\n笔记:\n{}",
                names.join("、"),
                clip(&notes, 3500)
            ),
        },
    ];
    let rendered = renderer::render(&messages, thinking_fast)?;
    let request = ContinueRequest {
        prompt: rendered,
        model: settings.model.clone(),
        max_output_tokens: ROUND_MAX_TOKENS.min(settings.max_tokens).max(1),
        stops: protocol::stops(GenerationStage::Answer),
        temperature: settings.temperature,
        top_k: settings.top_k,
        top_p: settings.top_p,
        alpha_presence: settings.alpha_presence,
        alpha_frequency: settings.alpha_frequency,
        alpha_decay: settings.alpha_decay,
    };
    let body = match lightning::continue_prompt_with_phase(
        client,
        server,
        &settings.password,
        request,
        "agent_summary",
        cancel,
        |_| {},
    )
    .await
    {
        Ok(generated) if generated.finish_reason == FinishReason::Cancelled => {
            return Err("cancelled".into());
        }
        Ok(generated) => {
            let cleaned = strip_toolish(&generated.text);
            if cleaned.trim().is_empty() || looks_like_instruction_echo(&cleaned) {
                fallback
            } else {
                cleaned
            }
        }
        Err(err) if err == "cancelled" => return Err(err),
        Err(_) => fallback,
    };
    Ok(body.trim().to_string())
}

fn compose_round_from_bits(bits: &[(String, String)]) -> String {
    let mut out = String::new();
    for (path, body) in bits {
        let line = clip(body, 280).replace('\n', " ");
        out.push_str(&format!("- **{path}**：{line}\n"));
    }
    if out.trim().is_empty() {
        "本轮没有读到可用文本。".into()
    } else {
        out
    }
}

fn strip_toolish(text: &str) -> String {
    if looks_like_tool_dump(text) {
        return String::new();
    }
    let mut value = text.replace("<tool_call>", "").replace("</tool_call>", "");
    value = value
        .replace("<tool_calls>", "")
        .replace("</tool_calls>", "");
    value = value.replace("<answer>", "").replace("</answer>", "");
    value = value.replace("<think></think>", "");
    value = value.replace("<think></think", "");
    value = value.replace("<think>", "").replace("</think>", "");
    while let Some(start) = value.find("```json") {
        let after = start + 7;
        if let Some(rel) = value[after..].find("```") {
            value.replace_range(start..after + rel + 3, "");
        } else {
            value.truncate(start);
            break;
        }
    }
    let trimmed = value.trim();
    if looks_like_tool_dump(trimmed) {
        return String::new();
    }
    trimmed.to_string()
}

fn looks_like_tool_dump(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    t.contains("<tool_call")
        || t.contains("tool_calls")
        || t.contains("</tool_call")
        || t.contains("```json")
        || t.contains("<read")
        || (t.contains("\"arguments\"") && t.contains("\"path\""))
}

fn looks_like_instruction_echo(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    t.contains("short chinese")
        || t.contains("markdown briefing")
        || t.contains("the round is the current round")
        || t.contains("do not mention round")
        || t.contains("no tools. no json")
}

fn execute_tool(tools: &ToolSet, name: &str, arguments: Value) -> (bool, Value) {
    match tools.get(name) {
        None => (
            false,
            json!({
                "ok": false,
                "tool": name,
                "error": format!("unknown tool \"{name}\""),
            }),
        ),
        Some(tool) => match tool.execute(&super::tools::normalize_arguments(&arguments)) {
            Ok(result) => (
                true,
                json!({
                    "ok": true,
                    "tool": name,
                    "result": result,
                }),
            ),
            Err(err) => (
                false,
                json!({
                    "ok": false,
                    "tool": name,
                    "error": err.to_string(),
                }),
            ),
        },
    }
}

fn listing_files(payload: &Value) -> Vec<String> {
    let empty = Vec::new();
    let entries = payload
        .pointer("/result/entries")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let mut files = protocol::preferred_read_paths(entries);
    files.retain(|path| !looks_binary(path));
    for entry in entries {
        if entry.get("type").and_then(Value::as_str) != Some("file") {
            continue;
        }
        let Some(path) = entry.get("path").and_then(Value::as_str) else {
            continue;
        };
        let name = path.rsplit('/').next().unwrap_or(path);
        if name.starts_with('.') || looks_binary(path) || files.iter().any(|seen| seen == path) {
            continue;
        }
        files.push(path.to_string());
    }
    files.retain(|path| !looks_binary(path));
    files
}

fn files_mentioned(goal: &str, files: &[String]) -> Vec<String> {
    files
        .iter()
        .filter(|path| {
            goal.contains(path.as_str())
                || path
                    .rsplit('/')
                    .next()
                    .is_some_and(|name| goal.contains(name))
        })
        .cloned()
        .collect()
}

fn looks_binary(path: &str) -> bool {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "png"
            | "jpg"
            | "jpeg"
            | "gif"
            | "webp"
            | "ico"
            | "bmp"
            | "tif"
            | "tiff"
            | "woff"
            | "woff2"
            | "ttf"
            | "otf"
            | "eot"
            | "pdf"
            | "zip"
            | "gz"
            | "tgz"
            | "bz2"
            | "xz"
            | "7z"
            | "rar"
            | "pth"
            | "pt"
            | "onnx"
            | "bin"
            | "safetensors"
            | "gguf"
            | "so"
            | "dylib"
            | "dll"
            | "a"
            | "o"
            | "wasm"
            | "mp3"
            | "wav"
            | "ogg"
            | "mp4"
            | "webm"
            | "mov"
            | "pyc"
            | "pyo"
            | "class"
            | "jar"
    )
}

fn append_note(worker: &mut Worker, path: &str, body: &str) {
    if !worker.notes.is_empty() {
        worker.notes.push('\n');
    }
    worker
        .notes
        .push_str(&format!("## {path}\n{}\n", body.trim()));
}

fn final_output(worker: &Worker) -> String {
    let mut out = worker.notes.trim().to_string();
    if !worker.pending_reads.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("Unread: {}", worker.pending_reads.join(", ")));
    }
    if out.is_empty() {
        traces_summary(&worker.traces)
    } else {
        out
    }
}

fn finish_worker(
    worker: &mut Worker,
    ok: bool,
    output: String,
    parent_step: u32,
    index: usize,
    emit: &impl Fn(AgentEvent),
) {
    worker.done = true;
    worker.ok = ok;
    worker.output = output.clone();
    emit(AgentEvent::Subagent {
        step: parent_step,
        index: index as u32,
        goal: worker.goal.clone(),
        status: if ok { "done".into() } else { "error".into() },
        output: Some(output),
    });
}

fn summarize_tool(name: &str, ok: bool, payload: &Value) -> String {
    if !ok {
        return clip(
            payload
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("failed"),
            120,
        );
    }
    let result = payload.get("result").unwrap_or(payload);
    match name {
        "list_files" => {
            let empty = Vec::new();
            let entries = result
                .get("entries")
                .and_then(Value::as_array)
                .unwrap_or(&empty);
            let names: Vec<&str> = entries
                .iter()
                .filter_map(|entry| entry.get("path").and_then(Value::as_str))
                .take(6)
                .collect();
            format!(
                "{} · {} 项 · {}",
                result
                    .get("workspace")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                entries.len(),
                names.join(", ")
            )
        }
        "read_file" => {
            let path = result.get("path").and_then(Value::as_str).unwrap_or("");
            let content = result.get("content").and_then(Value::as_str).unwrap_or("");
            let first = content.lines().next().unwrap_or("");
            format!("{} · {}", path, clip(first, 80))
        }
        "search_text" => {
            let n = result
                .get("matches")
                .and_then(Value::as_array)
                .map(|matches| matches.len())
                .unwrap_or(0);
            format!("{n} 处匹配")
        }
        _ => "ok".into(),
    }
}

fn traces_summary(traces: &[WorkerToolTrace]) -> String {
    traces
        .iter()
        .map(|trace| {
            let mark = if trace.ok { "ok" } else { "err" };
            format!("{} [{mark}] {}", trace.name, trace.summary)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn clip(value: &str, max: usize) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= max {
        value.to_string()
    } else {
        format!("{}…", chars[..max].iter().collect::<String>())
    }
}

fn compact_worker_payload(payload: &mut Value) {
    let Some(result) = payload.get_mut("result") else {
        return;
    };
    if let Some(Value::String(text)) = result.get_mut("content") {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() > FILE_EXCERPT_CHARS {
            *text = format!(
                "{}…\n[truncated {} chars]",
                chars[..FILE_EXCERPT_CHARS].iter().collect::<String>(),
                chars.len() - FILE_EXCERPT_CHARS
            );
            if let Some(Value::Bool(truncated)) = result.get_mut("truncated") {
                *truncated = true;
            }
        }
    }
}

pub fn reports_value(reports: &[WorkerReport]) -> Value {
    json!({
        "workers": reports,
    })
}

pub fn briefing(reports: &[WorkerReport]) -> String {
    let mut out = String::from("Parallel worker findings:\n");
    for (offset, report) in reports.iter().enumerate() {
        let body = if report.output.trim().is_empty() {
            traces_summary(&report.traces)
        } else {
            clip(&report.output, 1500)
        };
        out.push_str(&format!(
            "\n### {}. {}\n{}\n",
            offset + 1,
            report.goal,
            body
        ));
    }
    out
}

pub fn compose_answer(investigation: &Investigation, synthesis: &str) -> String {
    let mut out = String::new();
    for round in &investigation.rounds {
        if round.body.trim().is_empty() {
            continue;
        }
        out.push_str(&format!("## {}\n\n{}\n\n", round.title, round.body.trim()));
    }
    let title = investigation.synthesis_title();
    let synth = strip_toolish(synthesis);
    let synth = synth
        .trim()
        .trim_start_matches(&title)
        .trim_start_matches("## 综合")
        .trim()
        .to_string();
    out.push_str(&format!("## {title}\n\n"));
    if synth.is_empty() || !protocol::validate_answer(&synth) || protocol::is_thin_analysis(&synth)
    {
        let fallback = analysis_from_briefing(&investigation.briefing());
        let fallback = fallback
            .trim()
            .trim_start_matches(&title)
            .trim_start_matches("## 综合")
            .trim();
        out.push_str(fallback);
    } else {
        out.push_str(&synth);
    }
    out.trim().to_string()
}

pub fn analysis_from_briefing(briefing: &str) -> String {
    let body = briefing
        .strip_prefix("Round briefings:\n")
        .or_else(|| briefing.strip_prefix("Parallel worker findings:\n"))
        .unwrap_or(briefing)
        .trim();
    if body.is_empty() {
        "子任务已结束，但没有可用的文字结论。".into()
    } else {
        format!("## 综合\n\n根据各轮阅读结果：\n{body}")
    }
}

#[cfg(test)]
mod tests {
    use super::strip_toolish;

    #[test]
    fn strips_tool_call_dumps() {
        assert!(strip_toolish(
            "<tool_calls>\n{\"name\":\"read\",\"arguments\":{\"path\":\"x\"}}\n</tool_calls>"
        )
        .is_empty());
        assert!(strip_toolish(
            "```json\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"}}\n```"
        )
        .is_empty());
        assert!(!strip_toolish("- README.md：基准测试入口").is_empty());
    }

    #[test]
    fn rejects_instruction_echo() {
        assert!(super::looks_like_instruction_echo(
            "We need to produce a short Chinese markdown briefing of the round. The round is the current round (round 5)"
        ));
        assert!(!super::looks_like_instruction_echo("- README.md：项目说明"));
    }

    #[test]
    fn strips_complete_and_half_think_markers_without_leaking_delimiters() {
        assert_eq!(strip_toolish("<think></think>正文"), "正文");
        assert_eq!(strip_toolish("<think></think正文"), "正文");
    }
}
