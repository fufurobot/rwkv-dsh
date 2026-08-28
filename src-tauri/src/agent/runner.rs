use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::watch;

use super::config::Settings;
use super::lightning::{self, ContinueRequest};
use super::protocol::{self, ActionKind, FinishReason, GenerationStage, Message, Role, RouteKind};
use super::renderer;
use super::tools::{ToolError, ToolSet};
use super::workers;

const MAX_STEPS: u32 = 8;
const PROTOCOL_RETRIES: u32 = 1;
const DECISION_MAX_OUTPUT_TOKENS: u32 = 256;
const MAX_CONSECUTIVE_TOOL_FAILURES: u32 = 2;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    IntentResolved {
        intent: String,
        source: String,
    },
    Status {
        text: String,
    },
    Token {
        text: String,
    },
    ToolStart {
        step: u32,
        tool: String,
        arguments: Value,
    },
    ToolDone {
        step: u32,
        tool: String,
        ok: bool,
        payload: Value,
    },
    Retry {
        step: u32,
        error: String,
    },
    Step {
        step: u32,
        kind: String,
    },
    Subagent {
        step: u32,
        index: u32,
        goal: String,
        status: String,
        output: Option<String>,
    },
    SubagentTool {
        step: u32,
        index: u32,
        tool: String,
        ok: Option<bool>,
        summary: Option<String>,
        round: Option<u32>,
    },
}

#[derive(Debug, Clone)]
pub struct TurnResult {
    pub output: String,
    pub history: Vec<Message>,
}

struct StreamGate {
    buffer: String,
    mode: StreamMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamMode {
    Unknown,
    Hidden,
    Visible,
}

impl StreamGate {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            mode: StreamMode::Unknown,
        }
    }

    fn hidden() -> Self {
        Self {
            buffer: String::new(),
            mode: StreamMode::Hidden,
        }
    }

    fn push(&mut self, delta: &str) -> Option<String> {
        match self.mode {
            StreamMode::Hidden => None,
            StreamMode::Visible => Some(delta.to_string()),
            StreamMode::Unknown => {
                self.buffer.push_str(delta);
                let trimmed = self.buffer.trim_start();
                if trimmed.starts_with("<tool_call") || trimmed.starts_with("<route") {
                    self.mode = StreamMode::Hidden;
                    return None;
                }
                if let Some(rest) = trimmed.strip_prefix("<think></think>") {
                    self.mode = StreamMode::Visible;
                    if rest.is_empty() {
                        return None;
                    }
                    return Some(rest.to_string());
                }
                if trimmed.starts_with("<think") {
                    self.mode = StreamMode::Hidden;
                    return None;
                }
                if let Some(rest) = trimmed.strip_prefix("<answer>") {
                    self.mode = StreamMode::Visible;
                    if rest.is_empty() {
                        return None;
                    }
                    return Some(rest.to_string());
                }
                if is_possible_tag_prefix(trimmed) {
                    return None;
                }
                self.mode = StreamMode::Visible;
                Some(self.buffer.clone())
            }
        }
    }
}

fn is_possible_tag_prefix(value: &str) -> bool {
    if value.is_empty() || value == "<" {
        return true;
    }
    "<tool_call".starts_with(value)
        || "<think".starts_with(value)
        || "<answer".starts_with(value)
        || "<route".starts_with(value)
}

pub async fn run_turn_xml_legacy(
    client: &reqwest::Client,
    settings: &Settings,
    history: &[Message],
    prompt: &str,
    cancel: watch::Receiver<bool>,
    emit: &(impl Fn(AgentEvent) + Sync),
) -> Result<TurnResult, String> {
    let task = prompt.trim();
    if task.is_empty() {
        return Err("prompt is required".into());
    }
    if *cancel.borrow() {
        return Err("cancelled".into());
    }

    let tools = ToolSet::new(Some(settings.workspace.as_str()).filter(|value| !value.is_empty()))?;
    let mut server = lightning::resolve_route(client, &settings.base_url).await;
    let mut specs = tools.specs();
    specs.push(protocol::run_agents_spec());
    let thinking_fast = settings.thinking_fast();
    let workspace_bound = !settings.workspace.trim().is_empty();
    let resolved = crate::harness::intent::resolve(task, workspace_bound, None);
    let classified = if matches!(
        resolved.intent,
        crate::harness::intent::Intent::Chat | crate::harness::intent::Intent::Capability
    ) {
        RouteKind::Respond
    } else {
        RouteKind::Inspect
    };

    if resolved.intent == crate::harness::intent::Intent::ParallelLookup
        && resolved.tasks.len() >= 2
    {
        return run_split_workers(
            client,
            settings,
            &mut server,
            &tools,
            history,
            task,
            resolved.tasks,
            thinking_fast,
            cancel,
            emit,
        )
        .await;
    }

    let control = if classified == RouteKind::Respond {
        protocol::response_control(thinking_fast, &specs)
    } else {
        protocol::instructions(
            &specs,
            thinking_fast,
            Some(settings.workspace.as_str()).filter(|value| !value.is_empty()),
        )
    };

    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(Message {
        role: Role::System,
        content: control,
    });
    messages.extend(history.iter().cloned());
    messages.push(Message {
        role: Role::User,
        content: task.to_string(),
    });

    let mut turn_messages = vec![Message {
        role: Role::User,
        content: task.to_string(),
    }];
    let mut retries = 0u32;
    let mut route_violations = 0u32;
    let mut seen_tool_calls = std::collections::HashSet::new();
    let mut successful_tool_calls = 0u32;
    let mut tool_attempts = 0u32;
    let mut consecutive_failed_tool = String::new();
    let mut consecutive_tool_failures = 0u32;
    let mut stage = GenerationStage::Decision;
    let mut assistant_prefix = if classified == RouteKind::Inspect {
        protocol::tool_call_prefix().to_string()
    } else {
        String::new()
    };
    let mut force_answer = false;
    let mut collected: Option<workers::Investigation> = None;

    for step in 1..=MAX_STEPS {
        if *cancel.borrow() {
            return Err("cancelled".into());
        }
        if classified == RouteKind::Inspect
            && tool_attempts > 0
            && (step == MAX_STEPS || force_answer)
        {
            let (prepared, prefix) =
                protocol::prepare_answer(&messages, thinking_fast, collected.is_some());
            messages = prepared;
            assistant_prefix = prefix;
            stage = GenerationStage::Answer;
            emit(AgentEvent::Step {
                step,
                kind: "answer".into(),
            });
            emit(AgentEvent::Status {
                text: "answering".into(),
            });
        } else {
            emit(AgentEvent::Step {
                step,
                kind: "generate".into(),
            });
            emit(AgentEvent::Status {
                text: "generating".into(),
            });
        }

        let mut rendered = renderer::render(&messages, thinking_fast)?;
        let (prompt_text, injected) =
            renderer::append_assistant_prefix(&rendered, &assistant_prefix, thinking_fast);
        rendered = prompt_text;

        let mut max_output_tokens = settings.max_tokens;
        if stage == GenerationStage::Decision
            && classified == RouteKind::Inspect
            && successful_tool_calls == 0
        {
            max_output_tokens = DECISION_MAX_OUTPUT_TOKENS.min(settings.max_tokens);
        }

        let request = ContinueRequest {
            prompt: rendered,
            model: settings.model.clone(),
            max_output_tokens,
            stops: protocol::stops(stage),
            temperature: settings.temperature,
            top_k: settings.top_k,
            top_p: settings.top_p,
            alpha_presence: settings.alpha_presence,
            alpha_frequency: settings.alpha_frequency,
            alpha_decay: settings.alpha_decay,
        };

        let mut gate = if stage == GenerationStage::Answer {
            StreamGate::new()
        } else if classified == RouteKind::Inspect {
            StreamGate::hidden()
        } else if injected && assistant_prefix.starts_with("<tool_call") {
            StreamGate::hidden()
        } else {
            StreamGate::new()
        };
        let generated = lightning::continue_prompt_with_phase(
            client,
            &mut server,
            &settings.password,
            request,
            "agent",
            cancel.clone(),
            |delta| {
                if let Some(visible) = gate.push(delta) {
                    emit(AgentEvent::Token { text: visible });
                }
            },
        )
        .await
        .map_err(|err| {
            if err == "cancelled" {
                "cancelled".into()
            } else {
                err
            }
        })?;

        if generated.finish_reason == FinishReason::Cancelled {
            return Err("cancelled".into());
        }

        let mut model_action = renderer::reconstruct_output(&generated.text, thinking_fast);
        if injected && !model_action.trim_start().starts_with(&assistant_prefix) {
            model_action = format!("{assistant_prefix}{model_action}");
        }

        let mut action = match protocol::parse_action(&model_action, generated.finish_reason) {
            Ok(action) => action,
            Err(err) => {
                if retries >= PROTOCOL_RETRIES {
                    if tool_attempts > 0 && step < MAX_STEPS {
                        emit(AgentEvent::Retry { step, error: err });
                        force_answer = true;
                        messages.push(Message {
                            role: Role::User,
                            content: "Your previous response was invalid. Tools are now done. Answer the original task from the Tool results in the user's language. Do not call another tool.".into(),
                        });
                        continue;
                    }
                    return Err(err);
                }
                retries += 1;
                emit(AgentEvent::Retry {
                    step,
                    error: err.clone(),
                });
                let echoed = protocol::retry_echo(&model_action, &err);
                if !echoed.trim().is_empty() {
                    messages.push(Message {
                        role: Role::Assistant,
                        content: echoed,
                    });
                }
                messages.push(Message {
                    role: Role::User,
                    content: protocol::correction(&err),
                });
                continue;
            }
        };
        retries = 0;
        protocol::rewrite_list_file_as_read(&mut action);

        if classified == RouteKind::Respond && action.kind == ActionKind::Tool {
            let err = "tools are unavailable on the respond route".to_string();
            if route_violations >= PROTOCOL_RETRIES {
                return Err(err);
            }
            route_violations += 1;
            emit(AgentEvent::Retry {
                step,
                error: err.clone(),
            });
            messages.push(Message {
                role: Role::Assistant,
                content: model_action,
            });
            messages.push(Message {
                role: Role::User,
                content: "The route for this turn is respond. Answer directly using \
the conversation and do not call workspace tools."
                    .into(),
            });
            continue;
        }

        if action.kind == ActionKind::Final {
            let mut output = action.content.clone();
            let invalid = !protocol::validate_answer(&output);
            if let Some(investigation) = &collected {
                output = workers::compose_answer(investigation, &output);
            } else if invalid {
                output = protocol::answer_contract_fallback(task).to_string();
            }
            turn_messages.push(Message {
                role: Role::Assistant,
                content: output.clone(),
            });
            let mut committed = history.to_vec();
            committed.extend(turn_messages);
            return Ok(TurnResult {
                output,
                history: committed,
            });
        }

        tool_attempts += 1;
        assistant_prefix.clear();
        let call_key = canonical_tool_call(&action.name, &action.arguments);
        let mut executed = false;
        let mut recovery_blocked = false;
        let mut duplicate = false;
        let mut value = Value::Null;
        let mut tool_err: Option<ToolError> = None;

        emit(AgentEvent::Step {
            step,
            kind: "tool".into(),
        });

        if seen_tool_calls.contains(&call_key) {
            duplicate = true;
            tool_err = Some(ToolError::runtime("duplicate tool call rejected"));
        } else {
            seen_tool_calls.insert(call_key);
            if action.name == "run_agents" {
                emit(AgentEvent::ToolStart {
                    step,
                    tool: action.name.clone(),
                    arguments: action.arguments.clone(),
                });
                emit(AgentEvent::Status {
                    text: "workers".into(),
                });
                executed = true;
                match protocol::parse_run_agents_tasks(&action.arguments) {
                    Err(err) => {
                        tool_err = Some(ToolError::invalid(err));
                    }
                    Ok(goals) => {
                        match workers::run_workers(
                            client,
                            settings,
                            &mut server,
                            &tools,
                            goals,
                            task,
                            step,
                            cancel.clone(),
                            emit,
                        )
                        .await
                        {
                            Ok(investigation) => {
                                value = workers::reports_value(&investigation.reports);
                                collected = Some(investigation);
                            }
                            Err(err) if err == "cancelled" => return Err("cancelled".into()),
                            Err(err) => {
                                tool_err = Some(ToolError::runtime(err));
                            }
                        }
                    }
                }
            } else {
                match tools.get(&action.name) {
                    None => {
                        tool_err = Some(ToolError::runtime(format!(
                            "unknown tool \"{}\"",
                            action.name
                        )));
                    }
                    Some(_)
                        if action.name == consecutive_failed_tool
                            && consecutive_tool_failures >= MAX_CONSECUTIVE_TOOL_FAILURES =>
                    {
                        recovery_blocked = true;
                        tool_err = Some(ToolError::runtime(format!(
                            "tool \"{}\" blocked after {MAX_CONSECUTIVE_TOOL_FAILURES} consecutive failures; choose a different tool or answer",
                            action.name
                        )));
                    }
                    Some(tool) => {
                        executed = true;
                        emit(AgentEvent::ToolStart {
                            step,
                            tool: action.name.clone(),
                            arguments: action.arguments.clone(),
                        });
                        emit(AgentEvent::Status {
                            text: format!("calling {}", action.name),
                        });
                        match tool.execute(&super::tools::normalize_arguments(&action.arguments)) {
                            Ok(result) => value = result,
                            Err(err) => tool_err = Some(err),
                        }
                    }
                }
            }
        }

        if executed {
            match &tool_err {
                None => {
                    consecutive_failed_tool.clear();
                    consecutive_tool_failures = 0;
                }
                Some(err) if err.invalid_arguments => {}
                Some(_) if action.name == consecutive_failed_tool => {
                    consecutive_tool_failures += 1;
                }
                Some(_) => {
                    consecutive_failed_tool = action.name.clone();
                    consecutive_tool_failures = 1;
                }
            }
        }

        let payload = if let Some(err) = &tool_err {
            json!({
                "ok": false,
                "tool": action.name,
                "error": err.to_string(),
            })
        } else {
            json!({
                "ok": true,
                "tool": action.name,
                "result": value,
            })
        };
        emit(AgentEvent::ToolDone {
            step,
            tool: action.name.clone(),
            ok: tool_err.is_none(),
            payload: payload.clone(),
        });

        let encoded = serde_json::to_string(&payload).unwrap_or_else(|_| payload.to_string());
        let tool_content = if action.name == "run_agents" && tool_err.is_none() {
            protocol::format_tool_result(
                &collected
                    .as_ref()
                    .map(|investigation| investigation.briefing())
                    .unwrap_or_else(|| encoded.clone()),
            )
        } else {
            compact_tool_result(task, &protocol::format_tool_result(&encoded))
        };
        let recorded = protocol::record_action(&action, &generated.text);
        turn_messages.push(Message {
            role: Role::Assistant,
            content: recorded.clone(),
        });
        turn_messages.push(Message {
            role: Role::Tool,
            content: tool_content.clone(),
        });
        messages.push(Message {
            role: Role::Assistant,
            content: recorded,
        });
        messages.push(Message {
            role: Role::Tool,
            content: tool_content,
        });

        if tool_err.is_none() {
            successful_tool_calls += 1;
            if action.name == "run_agents" {
                force_answer = true;
                messages.push(Message {
                    role: Role::User,
                    content: protocol::synthesis_reminder().into(),
                });
            } else {
                messages.push(Message {
                    role: Role::User,
                    content: protocol::post_tool_reminder().into(),
                });
            }
        } else if duplicate {
            force_answer = true;
            messages.push(Message {
                role: Role::User,
                content: if successful_tool_calls > 0 {
                    "That tool call was rejected because it repeats a successful call.\nAnswer the current task from the evidence already collected.".into()
                } else {
                    "That tool call was rejected because it repeats an earlier failed call.\nTools are now unavailable. Answer from the Tool results, and clearly state anything that could not be verified.".into()
                },
            });
        } else {
            messages.push(Message {
                role: Role::User,
                content: tool_failure_reminder(
                    &action.name,
                    tool_err.as_ref().unwrap(),
                    recovery_blocked,
                    tools.argument_shape(&action.name),
                    tools.get("list_files").is_some(),
                    tools.get("search_text").is_some(),
                ),
            });
        }
    }

    if let Some(investigation) = collected {
        let output = workers::compose_answer(&investigation, "");
        turn_messages.push(Message {
            role: Role::Assistant,
            content: output.clone(),
        });
        let mut committed = history.to_vec();
        committed.extend(turn_messages);
        return Ok(TurnResult {
            output,
            history: committed,
        });
    }

    Err("agent reached the step limit".into())
}

async fn run_split_workers(
    client: &reqwest::Client,
    settings: &Settings,
    server: &mut lightning::LightningRoute,
    tools: &super::tools::ToolSet,
    history: &[Message],
    task: &str,
    goals: Vec<String>,
    thinking_fast: bool,
    cancel: watch::Receiver<bool>,
    emit: &(impl Fn(AgentEvent) + Sync),
) -> Result<TurnResult, String> {
    let arguments = json!({
        "tasks": goals.iter().map(|goal| json!({ "goal": goal })).collect::<Vec<_>>(),
    });
    emit(AgentEvent::ToolStart {
        step: 1,
        tool: "run_agents".into(),
        arguments: arguments.clone(),
    });
    emit(AgentEvent::Status {
        text: format!("workers {}", goals.len()),
    });
    let investigation = workers::run_workers(
        client,
        settings,
        server,
        tools,
        goals,
        task,
        1,
        cancel.clone(),
        emit,
    )
    .await?;
    let briefing = investigation.briefing();
    let payload = json!({
        "ok": true,
        "tool": "run_agents",
        "result": workers::reports_value(&investigation.reports),
    });
    emit(AgentEvent::ToolDone {
        step: 1,
        tool: "run_agents".into(),
        ok: true,
        payload: payload.clone(),
    });

    let recorded = protocol::record_action(
        &protocol::Action {
            kind: ActionKind::Tool,
            name: "run_agents".into(),
            arguments,
            content: String::new(),
        },
        "",
    );
    let mut messages = vec![Message {
        role: Role::System,
        content: protocol::instructions(
            &[],
            thinking_fast,
            Some(settings.workspace.as_str()).filter(|value| !value.is_empty()),
        ),
    }];
    messages.extend(history.iter().cloned());
    messages.push(Message {
        role: Role::User,
        content: task.to_string(),
    });
    messages.push(Message {
        role: Role::Assistant,
        content: recorded.clone(),
    });
    messages.push(Message {
        role: Role::Tool,
        content: protocol::format_tool_result(&briefing),
    });

    let (prepared, prefix) = protocol::prepare_answer(&messages, thinking_fast, true);
    emit(AgentEvent::Step {
        step: 2,
        kind: "answer".into(),
    });
    emit(AgentEvent::Status {
        text: "answering".into(),
    });
    emit(AgentEvent::Token {
        text: format!("\n\n## {}\n\n", investigation.synthesis_title()),
    });
    let rendered = renderer::render(&prepared, thinking_fast)?;
    let (prompt_text, _injected) =
        renderer::append_assistant_prefix(&rendered, &prefix, thinking_fast);
    let request = ContinueRequest {
        prompt: prompt_text,
        model: settings.model.clone(),
        max_output_tokens: settings.max_tokens.max(1),
        stops: protocol::stops(GenerationStage::Answer),
        temperature: settings.temperature,
        top_k: settings.top_k,
        top_p: settings.top_p,
        alpha_presence: settings.alpha_presence,
        alpha_frequency: settings.alpha_frequency,
        alpha_decay: settings.alpha_decay,
    };
    let mut gate = StreamGate::new();
    let generated = lightning::continue_prompt_with_phase(
        client,
        server,
        &settings.password,
        request,
        "synthesis",
        cancel,
        |delta| {
            if let Some(visible) = gate.push(delta) {
                emit(AgentEvent::Token { text: visible });
            }
        },
    )
    .await?;
    if generated.finish_reason == FinishReason::Cancelled {
        return Err("cancelled".into());
    }
    let mut model_action = renderer::reconstruct_output(&generated.text, thinking_fast);
    if !model_action.trim_start().starts_with("<answer>") {
        model_action = format!("<answer>{model_action}");
    }
    let mut output = match protocol::parse_action(&model_action, generated.finish_reason) {
        Ok(action) if action.kind == ActionKind::Final => action.content,
        Ok(action) => action.content,
        Err(_) => String::new(),
    };
    if !protocol::validate_answer(&output) || protocol::is_thin_analysis(&output) {
        output = String::new();
    }
    output = workers::compose_answer(&investigation, &output);
    Ok(TurnResult {
        output: output.clone(),
        history: {
            let mut committed = history.to_vec();
            committed.push(Message {
                role: Role::User,
                content: task.to_string(),
            });
            committed.push(Message {
                role: Role::Assistant,
                content: recorded,
            });
            committed.push(Message {
                role: Role::Tool,
                content: protocol::format_tool_result(&briefing),
            });
            committed.push(Message {
                role: Role::Assistant,
                content: output,
            });
            committed
        },
    })
}

fn canonical_tool_call(name: &str, arguments: &Value) -> String {
    format!("{name}\u{0}{arguments}")
}

fn tool_failure_reminder(
    name: &str,
    err: &ToolError,
    recovery_blocked: bool,
    shape: Option<String>,
    has_list: bool,
    has_search: bool,
) -> String {
    if err.invalid_arguments {
        let mut prompt = format!("The {name} arguments were rejected: {err}. ");
        if let Some(shape) = shape {
            prompt.push_str(&format!(
                "Retry {name} using only the fields in this exact argument shape: {shape}. "
            ));
        }
        prompt.push_str(
            "Do not add optional limit, byte, offset, or pagination fields unless the schema lists them.",
        );
        return prompt;
    }
    let mut prompt = format!("The tool call failed: {err}. ");
    if recovery_blocked {
        prompt.push_str("Do not call the same tool again in this turn. ");
    } else {
        prompt.push_str("Do not repeat the same call. ");
    }
    match name {
        "read_file" if has_list && has_search => {
            prompt.push_str(
                "If the path is uncertain, use list_files or search_text before another read_file call. ",
            );
        }
        "list_files" if has_search => {
            prompt.push_str(
                "Try an existing parent directory or workspace root, or use search_text for a known literal. ",
            );
        }
        _ => {}
    }
    prompt.push_str("Choose a different useful tool or answer with the limitation. Do not guess.");
    prompt
}

fn compact_tool_result(task: &str, wrapped: &str) -> String {
    const OPEN: &str = "<tool_result>";
    const CLOSE: &str = "</tool_result>";
    let Some(payload) = wrapped
        .strip_prefix(OPEN)
        .and_then(|value| value.strip_suffix(CLOSE))
    else {
        return wrapped.to_string();
    };
    let Ok(mut value) = serde_json::from_str::<Value>(payload) else {
        return wrapped.to_string();
    };
    compact_value(task, &mut value);
    match serde_json::to_string(&value) {
        Ok(encoded) => format!("{OPEN}{encoded}{CLOSE}"),
        Err(_) => wrapped.to_string(),
    }
}

fn compact_value(task: &str, value: &mut Value) {
    match value {
        Value::String(text) => {
            *text = relevant_excerpt(task, text);
        }
        Value::Array(items) => {
            if items.len() > 80 {
                items.truncate(80);
            }
            for item in items {
                compact_value(task, item);
            }
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                compact_value(task, item);
            }
        }
        _ => {}
    }
}

fn relevant_excerpt(task: &str, value: &str) -> String {
    const MAX: usize = 2400;
    const PREFIX: usize = 320;
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= MAX {
        return value.to_string();
    }
    let prefix: String = chars[..PREFIX].iter().collect();
    let chunk = MAX - PREFIX - 40;
    let terms: Vec<String> = task
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '.' && ch != '/' && ch != '-')
        .filter(|term| term.chars().count() > 1)
        .map(|term| term.to_lowercase())
        .collect();
    let mut best_start = PREFIX;
    let mut best_score = -1i32;
    let overlap = 160usize;
    let mut start = PREFIX;
    while start < chars.len() {
        let end = (start + chunk).min(chars.len());
        let candidate: String = chars[start..end].iter().collect::<String>().to_lowercase();
        let score = terms
            .iter()
            .map(|term| candidate.matches(term.as_str()).count() as i32)
            .sum();
        if score > best_score {
            best_score = score;
            best_start = start;
        }
        if end == chars.len() {
            break;
        }
        start += chunk.saturating_sub(overlap).max(1);
    }
    let best_end = (best_start + chunk).min(chars.len());
    let window: String = chars[best_start..best_end].iter().collect();
    format!("{prefix}\n...[tool result compacted]...\n{window}")
}

#[cfg(test)]
mod tests {
    use super::StreamGate;

    #[test]
    fn stream_gate_releases_reconstructed_empty_thinking_block() {
        let mut gate = StreamGate::new();
        assert_eq!(gate.push("<think></think>"), None);
        assert_eq!(gate.push("answer"), Some("answer".into()));
    }

    #[test]
    fn stream_gate_releases_text_after_inline_empty_thinking_block() {
        let mut gate = StreamGate::new();
        assert_eq!(gate.push("<think></think>answer"), Some("answer".into()));
    }
}
