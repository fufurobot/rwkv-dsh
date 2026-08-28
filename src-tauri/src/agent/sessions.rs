use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use super::protocol::{Message, Role};

const DEFAULT_TITLE: &str = "新对话";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiTool {
    pub id: String,
    pub step: u32,
    pub name: String,
    pub arguments: Value,
    #[serde(default)]
    pub ok: Option<bool>,
    #[serde(default)]
    pub payload: Option<Value>,
    #[serde(default)]
    pub workers: Option<Value>,
    #[serde(default)]
    pub open: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub tools: Vec<UiTool>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTurn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Session {
    pub id: String,
    pub title: String,
    pub updated_at: u64,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub messages: Vec<UiMessage>,
    #[serde(default)]
    pub history: Vec<StoredTurn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub id: String,
    pub path: String,
    pub sessions: Vec<Session>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStore {
    #[serde(default)]
    pub active_session_id: String,
    #[serde(default)]
    pub active_project_id: String,
    #[serde(default)]
    pub projects: Vec<Project>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMeta {
    pub id: String,
    pub title: String,
    pub updated_at: u64,
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectView {
    pub id: String,
    pub path: String,
    pub name: String,
    pub sessions: Vec<SessionMeta>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sidebar {
    pub active_session_id: String,
    pub active_project_id: String,
    pub projects: Vec<ProjectView>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSnap {
    pub sidebar: Sidebar,
    pub messages: Vec<UiMessage>,
    pub workspace: String,
}

impl SessionStore {
    pub fn load(path: &Path) -> Self {
        let Ok(raw) = fs::read_to_string(path) else {
            return Self::default();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        let encoded = serde_json::to_string_pretty(self).map_err(|err| err.to_string())?;
        fs::write(path, encoded).map_err(|err| err.to_string())
    }

    pub fn seed_workspace(&mut self, workspace: &str) {
        if workspace.trim().is_empty() {
            return;
        }
        let project_id = self.ensure_project(workspace);
        if self.active_project_id.is_empty() {
            self.active_project_id = project_id.clone();
        }
        if self.active_session_id.is_empty() {
            if let Some(session) = self
                .project(&project_id)
                .and_then(|project| project.sessions.first())
            {
                self.active_session_id = session.id.clone();
            }
        }
    }

    pub fn ensure_project(&mut self, path: &str) -> String {
        let normalized = normalize_path(path);
        if let Some(project) = self
            .projects
            .iter()
            .find(|project| paths_equal(&project.path, &normalized))
        {
            return project.id.clone();
        }
        let id = nid();
        self.projects.push(Project {
            id: id.clone(),
            path: normalized,
            sessions: vec![blank_session()],
        });
        id
    }

    pub fn project(&self, id: &str) -> Option<&Project> {
        self.projects.iter().find(|project| project.id == id)
    }

    pub fn project_mut(&mut self, id: &str) -> Option<&mut Project> {
        self.projects.iter_mut().find(|project| project.id == id)
    }

    pub fn session(&self, id: &str) -> Option<&Session> {
        self.projects
            .iter()
            .flat_map(|project| project.sessions.iter())
            .find(|session| session.id == id)
    }

    pub fn session_mut(&mut self, id: &str) -> Option<&mut Session> {
        self.projects
            .iter_mut()
            .flat_map(|project| project.sessions.iter_mut())
            .find(|session| session.id == id)
    }

    pub fn project_for_session(&self, session_id: &str) -> Option<&Project> {
        self.projects.iter().find(|project| {
            project
                .sessions
                .iter()
                .any(|session| session.id == session_id)
        })
    }

    pub fn open_project(&mut self, project_id: &str) -> Result<(), String> {
        let (id, next_session, already_here) = {
            let project = self
                .project(project_id)
                .ok_or_else(|| "project not found".to_string())?;
            let already_here = project
                .sessions
                .iter()
                .any(|session| session.id == self.active_session_id);
            (project.id.clone(), latest_session_id(project), already_here)
        };
        self.active_project_id = id;
        if !already_here {
            self.active_session_id = next_session;
        }
        Ok(())
    }

    pub fn open_session(&mut self, session_id: &str) -> Result<(), String> {
        let project_id = self
            .project_for_session(session_id)
            .ok_or_else(|| "session not found".to_string())?
            .id
            .clone();
        self.active_project_id = project_id;
        self.active_session_id = session_id.to_string();
        Ok(())
    }

    pub fn new_session(&mut self, project_id: &str) -> Result<String, String> {
        let id = {
            let project = self
                .project_mut(project_id)
                .ok_or_else(|| "project not found".to_string())?;
            let session = blank_session();
            let id = session.id.clone();
            project.sessions.insert(0, session);
            id
        };
        self.active_project_id = project_id.to_string();
        self.active_session_id = id.clone();
        Ok(id)
    }

    pub fn pin_session(&mut self, session_id: &str) -> Result<(), String> {
        let session = self
            .session_mut(session_id)
            .ok_or_else(|| "session not found".to_string())?;
        session.pinned = !session.pinned;
        Ok(())
    }

    pub fn delete_session(&mut self, session_id: &str) -> Result<(), String> {
        let project_id = self
            .project_for_session(session_id)
            .ok_or_else(|| "session not found".to_string())?
            .id
            .clone();
        let was_active = self.active_session_id == session_id;
        let next_id = {
            let project = self
                .project_mut(&project_id)
                .ok_or_else(|| "project not found".to_string())?;
            project.sessions.retain(|session| session.id != session_id);
            if project.sessions.is_empty() {
                project.sessions.push(blank_session());
            }
            latest_session_id(project)
        };
        if was_active {
            self.active_session_id = next_id;
            self.active_project_id = project_id;
        }
        Ok(())
    }

    pub fn save_transcript(&mut self, messages: Vec<UiMessage>) {
        let messages = sanitize_ui_messages(messages);
        let title = title_from(&messages);
        let id = self.active_session_id.clone();
        if let Some(session) = self.session_mut(&id) {
            session.messages = messages;
            session.updated_at = now_ms();
            if session.title == DEFAULT_TITLE {
                if let Some(next) = title {
                    session.title = next;
                }
            }
        }
    }

    pub fn save_history(&mut self, history: &[Message]) {
        let stored = history.iter().map(StoredTurn::from).collect();
        let id = self.active_session_id.clone();
        if let Some(session) = self.session_mut(&id) {
            session.history = stored;
            session.updated_at = now_ms();
        }
    }

    pub fn active_history(&self) -> Vec<Message> {
        self.session(&self.active_session_id)
            .map(|session| session.history.iter().map(Message::from).collect())
            .unwrap_or_default()
    }

    pub fn active_messages(&self) -> Vec<UiMessage> {
        self.session(&self.active_session_id)
            .map(|session| sanitize_ui_messages(session.messages.clone()))
            .unwrap_or_default()
    }

    pub fn workspace(&self) -> String {
        self.project(&self.active_project_id)
            .or_else(|| self.project_for_session(&self.active_session_id))
            .map(|project| project.path.clone())
            .unwrap_or_default()
    }

    pub fn snap(&self) -> SessionSnap {
        SessionSnap {
            messages: self.active_messages(),
            workspace: self.workspace(),
            sidebar: self.sidebar(),
        }
    }

    pub fn sidebar(&self) -> Sidebar {
        Sidebar {
            active_session_id: self.active_session_id.clone(),
            active_project_id: self.active_project_id.clone(),
            projects: self
                .projects
                .iter()
                .map(|project| ProjectView {
                    id: project.id.clone(),
                    path: project.path.clone(),
                    name: folder_name(&project.path),
                    sessions: {
                        let mut sessions: Vec<&Session> = project.sessions.iter().collect();
                        sessions.sort_by(|a, b| {
                            b.pinned
                                .cmp(&a.pinned)
                                .then_with(|| b.updated_at.cmp(&a.updated_at))
                        });
                        sessions
                            .into_iter()
                            .map(|session| SessionMeta {
                                id: session.id.clone(),
                                title: session.title.clone(),
                                updated_at: session.updated_at,
                                pinned: session.pinned,
                            })
                            .collect()
                    },
                })
                .collect(),
        }
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self {
            active_session_id: String::new(),
            active_project_id: String::new(),
            projects: Vec::new(),
        }
    }
}

impl From<&Message> for StoredTurn {
    fn from(message: &Message) -> Self {
        Self {
            role: match message.role {
                Role::System => "system".into(),
                Role::User => "user".into(),
                Role::Assistant => "assistant".into(),
                Role::Tool => "tool".into(),
            },
            content: message.content.clone(),
        }
    }
}

impl From<&StoredTurn> for Message {
    fn from(turn: &StoredTurn) -> Self {
        Self {
            role: match turn.role.as_str() {
                "system" => Role::System,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => Role::User,
            },
            content: turn.content.clone(),
        }
    }
}

fn blank_session() -> Session {
    Session {
        id: nid(),
        title: DEFAULT_TITLE.into(),
        updated_at: now_ms(),
        pinned: false,
        messages: Vec::new(),
        history: Vec::new(),
    }
}

fn latest_session_id(project: &Project) -> String {
    project
        .sessions
        .iter()
        .max_by(|a, b| {
            b.pinned
                .cmp(&a.pinned)
                .then_with(|| b.updated_at.cmp(&a.updated_at))
        })
        .map(|session| session.id.clone())
        .unwrap_or_default()
}

fn title_from(messages: &[UiMessage]) -> Option<String> {
    messages
        .iter()
        .find(|message| message.role == "user" && !message.content.trim().is_empty())
        .map(|message| {
            let text = message.content.trim().replace('\n', " ");
            if text.chars().count() > 32 {
                format!("{}…", text.chars().take(32).collect::<String>())
            } else {
                text
            }
        })
}

fn sanitize_ui_messages(messages: Vec<UiMessage>) -> Vec<UiMessage> {
    messages.into_iter().map(sanitize_ui_message).collect()
}

fn sanitize_ui_message(mut message: UiMessage) -> UiMessage {
    if message.role == "assistant" && looks_like_protocol_content(&message.content) {
        message.content.clear();
        if message.error.is_none() {
            message.error =
                Some("历史回答包含未完成的工具协议，原始协议已隐藏；请重新发送原问题。".into());
        }
    }
    message
}

fn looks_like_protocol_content(content: &str) -> bool {
    let trimmed = content.trim_start();
    let lower = content.to_lowercase();
    if [
        "<think",
        "</think",
        "<tool_call",
        "</tool_call",
        "<tool_result",
        "```json",
        "function output:",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
        && (lower.contains("\"name\"")
            || lower.contains("\"function\"")
            || lower.contains("\"arguments\""))
    {
        return true;
    }
    if !trimmed.starts_with('{') {
        return false;
    }
    let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
        return false;
    };
    let Some(object) = value.as_object() else {
        return false;
    };
    [
        "name",
        "function",
        "function_call",
        "tool_call",
        "tool_calls",
    ]
    .iter()
    .any(|key| object.contains_key(*key))
}

fn folder_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| path.to_string())
}

fn normalize_path(path: &str) -> String {
    let value = PathBuf::from(path);
    fs::canonicalize(&value)
        .unwrap_or(value)
        .to_string_lossy()
        .into_owned()
}

fn paths_equal(left: &str, right: &str) -> bool {
    Path::new(left) == Path::new(right) || normalize_path(left) == normalize_path(right)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as u64)
        .unwrap_or(0)
}

static SEQ: AtomicU64 = AtomicU64::new(1);

fn nid() -> String {
    format!("{:x}-{:x}", now_ms(), SEQ.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(content: &str) -> UiMessage {
        UiMessage {
            id: "assistant".into(),
            role: "assistant".into(),
            content: content.into(),
            tools: Vec::new(),
            error: None,
        }
    }

    #[test]
    fn hides_incomplete_tool_protocol_from_transcript() {
        let message = sanitize_ui_message(assistant(
            "<think>\n{\"name\":\"list_files\",\"arguments\":{}}\n</think>\n```json\n{\"name\":\"read_file\"}",
        ));
        assert!(message.content.is_empty());
        assert!(message
            .error
            .as_deref()
            .is_some_and(|error| error.contains("工具协议")));
    }

    #[test]
    fn keeps_normal_json_explanations_visible() {
        let message = sanitize_ui_message(assistant("项目配置示例：{\"name\":\"demo\"}"));
        assert_eq!(message.content, "项目配置示例：{\"name\":\"demo\"}");
        assert!(message.error.is_none());
    }
}
