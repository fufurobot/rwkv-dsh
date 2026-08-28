mod agent;
pub mod cordis;
pub mod harness;

use agent::tools::{Workspace, WorkspaceEntry};
use agent::{AgentEvent, Message, SessionSnap, SessionStore, Settings, UiMessage};
use cordis::Context;
use harness::profiles::{self, AgentLoopService, Profile, RuntimeHandles};
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;
use tokio::sync::watch;

pub struct AppState {
    settings: Arc<Mutex<Settings>>,
    sessions: Arc<Mutex<SessionStore>>,
    history: Arc<Mutex<Vec<Message>>>,
    cordis: Arc<Mutex<Context>>,
    busy: Mutex<bool>,
    cancel: Mutex<Option<watch::Sender<bool>>>,
}

struct TurnStateGuard<'a> {
    state: &'a AppState,
    armed: bool,
}

impl<'a> TurnStateGuard<'a> {
    fn acquire(state: &'a AppState) -> Result<Self, String> {
        let mut busy = state
            .busy
            .lock()
            .map_err(|_| "busy lock is poisoned".to_string())?;
        if *busy {
            return Err("a turn is already running".into());
        }
        *busy = true;
        Ok(Self { state, armed: true })
    }

    fn install_cancel(&self, sender: watch::Sender<bool>) -> Result<(), String> {
        *self
            .state
            .cancel
            .lock()
            .map_err(|_| "cancel lock is poisoned".to_string())? = Some(sender);
        Ok(())
    }

    fn clear(&self) {
        if let Ok(mut cancel) = self.state.cancel.lock() {
            *cancel = None;
        }
        if let Ok(mut busy) = self.state.busy.lock() {
            *busy = false;
        }
    }

    fn finish(mut self) {
        self.clear();
        self.armed = false;
    }
}

impl Drop for TurnStateGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.clear();
        }
    }
}

fn after_turn_finished<T>(turn_state: TurnStateGuard<'_>, terminal: impl FnOnce() -> T) -> T {
    turn_state.finish();
    terminal()
}

fn settings_path(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    let dir = app.path().app_config_dir().map_err(|err| err.to_string())?;
    Ok(dir.join("settings.json"))
}

fn sessions_path(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    let dir = app.path().app_config_dir().map_err(|err| err.to_string())?;
    Ok(dir.join("sessions.json"))
}

fn persist_sessions(app: &AppHandle, store: &SessionStore) -> Result<(), String> {
    store.save(&sessions_path(app)?)
}

fn apply_workspace(app: &AppHandle, state: &AppState, workspace: &str) -> Result<(), String> {
    if workspace.trim().is_empty() {
        return Ok(());
    }
    let mut settings = state.settings.lock().expect("settings lock");
    if settings.workspace != workspace {
        settings.workspace = workspace.to_string();
        settings.save(&settings_path(app)?)?;
    }
    Ok(())
}

fn activate_session(state: &AppState, store: &SessionStore) {
    *state.history.lock().expect("history lock") = store.active_history();
}

fn reject_if_busy(state: &AppState) -> Result<(), String> {
    if *state.busy.lock().expect("busy lock") {
        Err("a turn is already running".into())
    } else {
        Ok(())
    }
}

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> Settings {
    state.settings.lock().expect("settings lock").clone()
}

#[tauri::command]
fn save_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    mut settings: Settings,
) -> Result<Settings, String> {
    settings.thinking = "off".into();
    settings.save(&settings_path(&app)?)?;
    *state.settings.lock().expect("settings lock") = settings.clone();
    Ok(settings)
}

#[tauri::command]
async fn pick_workspace(app: AppHandle) -> Result<Option<String>, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().pick_folder(move |folder| {
        let path = folder.and_then(|file| file.into_path().ok());
        let _ = tx.send(path.map(|value| value.to_string_lossy().into_owned()));
    });
    Ok(rx.await.unwrap_or(None))
}

#[tauri::command]
fn list_workspace(
    state: State<'_, AppState>,
    path: Option<String>,
) -> Result<Vec<WorkspaceEntry>, String> {
    let workspace = state
        .settings
        .lock()
        .expect("settings lock")
        .workspace
        .clone();
    if workspace.trim().is_empty() {
        return Err("workspace is not set".into());
    }
    let root = Workspace::open(&workspace)?;
    root.list_dir(path.as_deref().unwrap_or(""))
}

#[tauri::command]
fn get_sessions(state: State<'_, AppState>) -> SessionSnap {
    state.sessions.lock().expect("sessions lock").snap()
}

#[tauri::command]
fn save_transcript(
    app: AppHandle,
    state: State<'_, AppState>,
    messages: Vec<UiMessage>,
) -> Result<SessionSnap, String> {
    let mut store = state.sessions.lock().expect("sessions lock");
    store.save_transcript(messages);
    persist_sessions(&app, &store)?;
    Ok(store.snap())
}

#[tauri::command]
fn open_session(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<SessionSnap, String> {
    reject_if_busy(&state)?;
    let mut store = state.sessions.lock().expect("sessions lock");
    store.open_session(&id)?;
    persist_sessions(&app, &store)?;
    apply_workspace(&app, &state, &store.workspace())?;
    activate_session(&state, &store);
    Ok(store.snap())
}

#[tauri::command]
fn open_project(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<SessionSnap, String> {
    reject_if_busy(&state)?;
    let mut store = state.sessions.lock().expect("sessions lock");
    store.open_project(&id)?;
    persist_sessions(&app, &store)?;
    apply_workspace(&app, &state, &store.workspace())?;
    activate_session(&state, &store);
    Ok(store.snap())
}

#[tauri::command]
fn new_session(
    app: AppHandle,
    state: State<'_, AppState>,
    project_id: String,
) -> Result<SessionSnap, String> {
    reject_if_busy(&state)?;
    let mut store = state.sessions.lock().expect("sessions lock");
    store.new_session(&project_id)?;
    persist_sessions(&app, &store)?;
    apply_workspace(&app, &state, &store.workspace())?;
    activate_session(&state, &store);
    Ok(store.snap())
}

#[tauri::command]
async fn add_project(app: AppHandle, state: State<'_, AppState>) -> Result<SessionSnap, String> {
    reject_if_busy(&state)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().pick_folder(move |folder| {
        let path = folder.and_then(|file| file.into_path().ok());
        let _ = tx.send(path.map(|value| value.to_string_lossy().into_owned()));
    });
    let Some(path) = rx.await.unwrap_or(None) else {
        let store = state.sessions.lock().expect("sessions lock");
        return Ok(store.snap());
    };
    let mut store = state.sessions.lock().expect("sessions lock");
    let project_id = store.ensure_project(&path);
    store.open_project(&project_id)?;
    persist_sessions(&app, &store)?;
    apply_workspace(&app, &state, &store.workspace())?;
    activate_session(&state, &store);
    Ok(store.snap())
}

#[tauri::command]
fn pin_session(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<SessionSnap, String> {
    let mut store = state.sessions.lock().expect("sessions lock");
    store.pin_session(&id)?;
    persist_sessions(&app, &store)?;
    Ok(store.snap())
}

#[tauri::command]
fn delete_session(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<SessionSnap, String> {
    reject_if_busy(&state)?;
    let mut store = state.sessions.lock().expect("sessions lock");
    store.delete_session(&id)?;
    persist_sessions(&app, &store)?;
    apply_workspace(&app, &state, &store.workspace())?;
    activate_session(&state, &store);
    Ok(store.snap())
}

#[tauri::command]
fn chat_reset(state: State<'_, AppState>) -> Result<(), String> {
    if *state.busy.lock().expect("busy lock") {
        return Err("a turn is already running".into());
    }
    state.history.lock().expect("history lock").clear();
    Ok(())
}

#[tauri::command]
fn chat_cancel(state: State<'_, AppState>) {
    if let Some(tx) = state.cancel.lock().expect("cancel lock").as_ref() {
        let _ = tx.send(true);
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Payload {
    #[serde(skip_serializing_if = "Option::is_none")]
    turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    step: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    goal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    round: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    intent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
}

#[tauri::command]
async fn chat_send(
    app: AppHandle,
    state: State<'_, AppState>,
    prompt: String,
    turn_id: Option<String>,
) -> Result<(), String> {
    let turn_id = normalize_turn_id(turn_id);
    let turn_state = TurnStateGuard::acquire(&state)?;
    let (tx, rx) = watch::channel(false);
    turn_state.install_cancel(tx)?;

    let workspace = state
        .sessions
        .lock()
        .map_err(|_| "sessions lock is poisoned".to_string())?
        .workspace();
    let history_length = state
        .history
        .lock()
        .map_err(|_| "history lock is poisoned".to_string())?
        .len();
    eprintln!(
        "[chat_send] userChars={} history={} workspace={}",
        prompt.chars().count(),
        history_length,
        workspace
    );
    let cordis = Arc::clone(&state.cordis);
    let agent_loop = cordis
        .lock()
        .map_err(|_| "Cordis context lock is poisoned".to_string())?
        .service::<AgentLoopService>("agentLoop")
        .ok_or_else(|| "agentLoop service is unavailable".to_string())?;
    let emitter = app.clone();
    let event_turn_id = turn_id.clone();
    let emit = move |event: AgentEvent| {
        if let AgentEvent::Token { text } = &event {
            if looks_like_protocol_text(text) {
                return;
            }
        }
        let (name, mut payload) = match event {
            AgentEvent::IntentResolved { intent, source } => (
                "intent_resolved",
                Payload {
                    intent: Some(intent),
                    source: Some(source),
                    ..empty_payload()
                },
            ),
            AgentEvent::Status { text } => (
                "status",
                Payload {
                    text: Some(text),
                    ..empty_payload()
                },
            ),
            AgentEvent::Token { text } => (
                "token_delta",
                Payload {
                    text: Some(text),
                    ..empty_payload()
                },
            ),
            AgentEvent::ToolStart {
                step,
                tool,
                arguments,
            } => (
                "tool_start",
                Payload {
                    step: Some(step),
                    tool: Some(tool),
                    arguments: Some(arguments),
                    ..empty_payload()
                },
            ),
            AgentEvent::ToolDone {
                step,
                tool,
                ok,
                payload,
            } => (
                "tool_done",
                Payload {
                    step: Some(step),
                    tool: Some(tool),
                    ok: Some(ok),
                    payload: Some(payload),
                    ..empty_payload()
                },
            ),
            AgentEvent::Retry { step, error } => (
                "retry",
                Payload {
                    step: Some(step),
                    error: Some(error),
                    ..empty_payload()
                },
            ),
            AgentEvent::Step { step, kind } => (
                "step_mark",
                Payload {
                    step: Some(step),
                    kind: Some(kind),
                    ..empty_payload()
                },
            ),
            AgentEvent::Subagent {
                step,
                index,
                goal,
                status,
                output,
            } => (
                "subagent",
                Payload {
                    step: Some(step),
                    index: Some(index),
                    goal: Some(goal),
                    status: Some(status),
                    output,
                    ..empty_payload()
                },
            ),
            AgentEvent::SubagentTool {
                step,
                index,
                tool,
                ok,
                summary,
                round,
            } => (
                "subagent_tool",
                Payload {
                    step: Some(step),
                    index: Some(index),
                    tool: Some(tool),
                    ok,
                    summary,
                    round,
                    ..empty_payload()
                },
            ),
        };
        payload.turn_id = Some(event_turn_id.clone());
        let _ = emitter.emit(name, payload);
    };

    let trace_emitter = app.clone();
    let trace_turn_id = turn_id.clone();
    let _trace_guard = agent::lightning::install_trace_sink(Arc::new(move |mut payload| {
        if let Some(object) = payload.as_object_mut() {
            object.insert("turnId".into(), serde_json::json!(trace_turn_id.clone()));
        }
        let _ = trace_emitter.emit("model_trace", payload);
    }));
    let result = agent_loop
        .run(&workspace, &prompt, rx, &cordis, &emit)
        .await;

    match result {
        Ok(turn) => {
            {
                let mut store = state.sessions.lock().expect("sessions lock");
                store.save_history(&turn.history);
                let _ = persist_sessions(&app, &store);
            }
            *state.history.lock().expect("history lock") = turn.history;
            let _ = after_turn_finished(turn_state, || {
                app.emit(
                    "turn_done",
                    Payload {
                        turn_id: Some(turn_id.clone()),
                        output: Some(turn.output),
                        ..empty_payload()
                    },
                )
            });
            Ok(())
        }
        Err(err) if err == "cancelled" => {
            let _ = after_turn_finished(turn_state, || {
                app.emit(
                    "turn_error",
                    Payload {
                        turn_id: Some(turn_id.clone()),
                        message: Some("stopped".into()),
                        ..empty_payload()
                    },
                )
            });
            Err("stopped".into())
        }
        Err(err) => {
            let _ = after_turn_finished(turn_state, || {
                app.emit(
                    "turn_error",
                    Payload {
                        turn_id: Some(turn_id.clone()),
                        message: Some(err.clone()),
                        ..empty_payload()
                    },
                )
            });
            Err(err)
        }
    }
}

fn empty_payload() -> Payload {
    Payload {
        turn_id: None,
        text: None,
        step: None,
        kind: None,
        tool: None,
        arguments: None,
        ok: None,
        payload: None,
        error: None,
        message: None,
        output: None,
        index: None,
        goal: None,
        status: None,
        summary: None,
        round: None,
        intent: None,
        source: None,
    }
}

fn looks_like_protocol_text(text: &str) -> bool {
    let lower = text.to_lowercase();
    let wrapped = [
        "<think",
        "</think",
        "<tool_call",
        "```json",
        "function output:",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    wrapped
        && (lower.contains("\"name\"")
            || lower.contains("\"function\"")
            || lower.contains("\"arguments\""))
}

const MAX_TURN_ID_LENGTH: usize = 128;
static NEXT_TURN_ID: AtomicU64 = AtomicU64::new(1);

fn normalize_turn_id(value: Option<String>) -> String {
    value
        .filter(|candidate| valid_turn_id(candidate))
        .unwrap_or_else(new_turn_id)
}

fn valid_turn_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TURN_ID_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn new_turn_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0);
    let sequence = NEXT_TURN_ID.fetch_add(1, Ordering::Relaxed);
    format!("turn-{timestamp:x}-{sequence:x}")
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(180))
        .build()
        .expect("http client");

    let settings = Arc::new(Mutex::new(Settings::default()));
    let sessions = Arc::new(Mutex::new(SessionStore::default()));
    let history = Arc::new(Mutex::new(Vec::new()));
    let cordis = profiles::boot(
        Profile::Desktop,
        RuntimeHandles::new(
            http.clone(),
            Arc::clone(&settings),
            Arc::clone(&sessions),
            Arc::clone(&history),
        ),
    )
    .expect("desktop profile");

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            settings,
            sessions,
            history,
            cordis: Arc::new(Mutex::new(cordis)),
            busy: Mutex::new(false),
            cancel: Mutex::new(None),
        })
        .setup(|app| {
            let path = settings_path(app.handle())?;
            let settings = Settings::load(&path);
            let mut store = SessionStore::load(&sessions_path(app.handle())?);
            store.seed_workspace(&settings.workspace);
            let _ = persist_sessions(app.handle(), &store);
            let history = store.active_history();
            let workspace = store.workspace();
            let app_state = app.state::<AppState>();
            {
                let mut current = app_state.settings.lock().expect("settings lock");
                *current = settings;
                if !workspace.is_empty() && current.workspace != workspace {
                    current.workspace = workspace;
                    let _ = current.save(&path);
                }
            }
            *app_state.sessions.lock().expect("sessions lock") = store;
            *app_state.history.lock().expect("history lock") = history;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_settings,
            save_settings,
            pick_workspace,
            list_workspace,
            get_sessions,
            save_transcript,
            open_session,
            open_project,
            new_session,
            add_project,
            pin_session,
            delete_session,
            chat_send,
            chat_cancel,
            chat_reset
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_without_agent_loop() -> AppState {
        AppState {
            settings: Arc::new(Mutex::new(Settings::default())),
            sessions: Arc::new(Mutex::new(SessionStore::default())),
            history: Arc::new(Mutex::new(Vec::new())),
            cordis: Arc::new(Mutex::new(Context::new())),
            busy: Mutex::new(false),
            cancel: Mutex::new(None),
        }
    }

    #[test]
    fn turn_state_is_cleared_when_agent_loop_setup_fails() {
        let state = state_without_agent_loop();
        let result: Result<(), String> = (|| {
            let turn_state = TurnStateGuard::acquire(&state)?;
            let (sender, _receiver) = watch::channel(false);
            turn_state.install_cancel(sender)?;
            state
                .cordis
                .lock()
                .map_err(|_| "Cordis context lock is poisoned".to_string())?
                .service::<AgentLoopService>("agentLoop")
                .ok_or_else(|| "agentLoop service is unavailable".to_string())?;
            turn_state.finish();
            Ok(())
        })();

        assert_eq!(result, Err("agentLoop service is unavailable".into()));
        assert!(!*state.busy.lock().expect("busy lock"));
        assert!(state.cancel.lock().expect("cancel lock").is_none());
    }

    #[test]
    fn terminal_notification_runs_after_turn_state_is_cleared() {
        let state = state_without_agent_loop();
        let turn_state = TurnStateGuard::acquire(&state).expect("acquire turn state");
        let (sender, _receiver) = watch::channel(false);
        turn_state
            .install_cancel(sender)
            .expect("install cancellation sender");

        after_turn_finished(turn_state, || {
            assert!(state.cancel.lock().expect("cancel lock").is_none());
            let next_turn = TurnStateGuard::acquire(&state).expect("acquire next turn state");
            next_turn.finish();
        });
    }

    #[test]
    fn turn_id_accepts_frontend_safe_values() {
        let value = "mt71x83o-ywh39w".to_string();
        assert!(valid_turn_id(&value));
        assert_eq!(normalize_turn_id(Some(value.clone())), value);
    }

    #[test]
    fn turn_id_replaces_invalid_values_with_unique_server_ids() {
        let invalid = [
            "",
            "has spaces",
            "含有中文",
            "has/slash",
            &"x".repeat(MAX_TURN_ID_LENGTH + 1),
        ];
        for value in invalid {
            let normalized = normalize_turn_id(Some(value.to_string()));
            assert!(valid_turn_id(&normalized));
            assert!(normalized.starts_with("turn-"));
        }
        let first = normalize_turn_id(None);
        let second = normalize_turn_id(None);
        assert_ne!(first, second);
    }

    #[test]
    fn event_payload_serializes_turn_id_in_camel_case() {
        let payload = Payload {
            turn_id: Some("turn-1".into()),
            ..empty_payload()
        };
        let serialized = serde_json::to_value(payload).expect("serialize payload");
        assert_eq!(
            serialized.get("turnId").and_then(|value| value.as_str()),
            Some("turn-1")
        );
        assert!(serialized.get("turn_id").is_none());
    }
}
