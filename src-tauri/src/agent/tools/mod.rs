mod calculator;
mod workspace;

use serde_json::{json, Value};

use super::protocol::ToolSpec;
use crate::harness::intent::Intent;

pub use calculator::Calculator;
pub use workspace::{Workspace, WorkspaceEntry};

const LEGACY_DEFAULT_TOOL_NAMES: &[&str] = &["list_files", "read_file", "search", "calculator"];
const WORKSPACE_TOOL_NAMES: &[&str] = &["list_files", "read_file", "search", "write_file"];

pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn execute(&self, arguments: &Value) -> Result<Value, ToolError>;
}

#[derive(Debug)]
pub struct ToolError {
    pub message: String,
    pub invalid_arguments: bool,
}

impl ToolError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            invalid_arguments: true,
        }
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            invalid_arguments: false,
        }
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

pub struct ToolSet {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolSet {
    pub fn new(workspace: Option<&str>) -> Result<Self, String> {
        Self::with_allowlist(workspace, LEGACY_DEFAULT_TOOL_NAMES)
    }

    pub fn for_intent(workspace: Option<&str>, intent: Intent) -> Result<Self, String> {
        let workspace_bound = workspace
            .map(str::trim)
            .is_some_and(|value| !value.is_empty());
        Self::with_allowlist(workspace, intent.allowed_tools(workspace_bound))
    }

    pub fn for_request(
        workspace: Option<&str>,
        intent: Intent,
        allowed: Option<&[&str]>,
    ) -> Result<Self, String> {
        match allowed {
            Some(allowed) => {
                let workspace_bound = workspace
                    .map(str::trim)
                    .is_some_and(|value| !value.is_empty());
                let intent_allowed = intent.allowed_tools(workspace_bound);
                if let Some(name) = allowed
                    .iter()
                    .find(|name| !tool_is_allowed(name, intent_allowed))
                {
                    return Err(format!(
                        "tool {name:?} is outside the allowed scope for intent {}",
                        intent.as_str()
                    ));
                }
                Self::with_allowlist(workspace, allowed)
            }
            None => Self::for_intent(workspace, intent),
        }
    }

    pub fn with_allowlist(workspace: Option<&str>, allowed: &[&str]) -> Result<Self, String> {
        let mut tools: Vec<Box<dyn Tool>> = Vec::new();
        let needs_workspace = WORKSPACE_TOOL_NAMES
            .iter()
            .any(|name| tool_is_allowed(name, allowed));
        if let Some(root) = workspace
            .map(str::trim)
            .filter(|value| !value.is_empty() && needs_workspace)
        {
            let ws = Workspace::open(root)?;
            let include_write = tool_is_allowed("write_file", allowed);
            tools.extend(
                ws.into_tools(include_write)
                    .into_iter()
                    .filter(|tool| tool_is_allowed(&tool.spec().name, allowed)),
            );
        }
        if tool_is_allowed("calculator", allowed) {
            tools.push(Box::new(Calculator));
        }
        Ok(Self { tools })
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|tool| tool.spec()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        let name = canonical_tool_name(name);
        self.tools
            .iter()
            .find(|tool| tool.spec().name == name)
            .map(|tool| tool.as_ref())
    }

    pub fn argument_shape(&self, name: &str) -> Option<String> {
        self.get(name).map(|tool| tool.spec().arguments.to_string())
    }
}

pub fn normalize_arguments(value: &Value) -> Value {
    let decoded = match value {
        Value::String(text) => serde_json::from_str(text).unwrap_or_else(|_| value.clone()),
        other => other.clone(),
    };
    let mut object = match decoded {
        Value::Object(object) => object,
        Value::Null => serde_json::Map::new(),
        other => return other,
    };
    for wrapper in ["args", "params", "parameters", "arguments"] {
        let Some(nested) = object.remove(wrapper) else {
            continue;
        };
        let nested = match nested {
            Value::String(text) => serde_json::from_str::<Value>(&text).ok(),
            other => Some(other),
        };
        if let Some(Value::Object(mut nested)) = nested {
            for (key, value) in std::mem::take(&mut object) {
                nested.insert(key, value);
            }
            object = nested;
            break;
        }
    }
    if !object.contains_key("path") {
        for alias in ["file_path", "filename", "file", "filepath"] {
            if let Some(path) = object.remove(alias) {
                object.insert("path".into(), path);
                break;
            }
        }
    }
    for key in ["max_depth", "max_results", "offset", "max_bytes"] {
        let replacement = match object.get(key) {
            Some(Value::String(text)) => {
                let trimmed = text.trim();
                Some(trimmed.parse::<i64>().ok().map(|number| json!(number)))
            }
            Some(Value::Number(number)) if number.as_i64().is_none() => {
                number.as_f64().map(|float| Some(json!(float as i64)))
            }
            _ => None,
        };
        if let Some(next) = replacement {
            match next {
                Some(number) => {
                    object.insert(key.to_string(), number);
                }
                None => {
                    object.remove(key);
                }
            }
        }
    }
    let flag = object
        .get("case_sensitive")
        .and_then(Value::as_str)
        .map(|text| text.trim().to_ascii_lowercase());
    if let Some(lowered) = flag {
        if lowered == "true" || lowered == "1" {
            object.insert("case_sensitive".into(), json!(true));
        } else if lowered == "false" || lowered == "0" {
            object.insert("case_sensitive".into(), json!(false));
        }
    }
    Value::Object(object)
}

pub fn canonical_tool_name(name: &str) -> &str {
    match name.trim() {
        "search_text" => "search",
        canonical => canonical,
    }
}

fn tool_is_allowed(name: &str, allowed: &[&str]) -> bool {
    let canonical = canonical_tool_name(name);
    allowed
        .iter()
        .any(|candidate| canonical_tool_name(candidate) == canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

    struct TestWorkspace {
        path: PathBuf,
    }

    impl TestWorkspace {
        fn new() -> Self {
            let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "rwkv-agent-tools-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test workspace");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn as_str(&self) -> &str {
            self.path.to_str().expect("UTF-8 test workspace path")
        }
    }

    impl Drop for TestWorkspace {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn spec_names(tools: &ToolSet) -> Vec<String> {
        tools.specs().into_iter().map(|spec| spec.name).collect()
    }

    #[test]
    fn canonical_path_wins_over_legacy_aliases() {
        let value = normalize_arguments(&json!({
            "path": "canonical.txt",
            "file_path": "legacy.txt"
        }));
        assert_eq!(value["path"], "canonical.txt");
    }

    #[test]
    fn hoists_only_one_argument_level() {
        let value = normalize_arguments(&json!({
            "arguments": {
                "args": { "path": "too-deep.txt" },
                "file_path": "one-level.txt"
            }
        }));
        assert_eq!(value["path"], "one-level.txt");
        assert!(value.get("args").is_some());
    }

    #[test]
    fn params_alias_is_hoisted_like_other_argument_wrappers() {
        let value = normalize_arguments(&json!({
            "params": {"path": "note.txt"}
        }));
        assert_eq!(value, json!({"path": "note.txt"}));
    }

    #[test]
    fn legacy_search_name_is_invisible_alias() {
        assert_eq!(canonical_tool_name("search_text"), "search");
    }

    #[test]
    fn default_set_uses_canonical_read_and_calculator_names() {
        let workspace = TestWorkspace::new();
        let tools = ToolSet::new(Some(workspace.as_str())).expect("build default tools");
        assert_eq!(
            spec_names(&tools),
            vec!["list_files", "read_file", "search", "calculator"]
        );
        assert!(tools.get("search").is_some());
        assert!(tools.get("search_text").is_some());
        assert!(tools.get("write_file").is_none());
    }

    #[test]
    fn allowlist_filters_specs_and_runtime_lookup_together() {
        let workspace = TestWorkspace::new();
        let tools = ToolSet::with_allowlist(Some(workspace.as_str()), &["read_file", "write_file"])
            .expect("build filtered tools");
        assert_eq!(spec_names(&tools), vec!["read_file", "write_file"]);
        assert!(tools.get("read_file").is_some());
        assert!(tools.get("write_file").is_some());
        assert!(tools.get("list_files").is_none());
        assert!(tools.get("search").is_none());
        assert!(tools.get("search_text").is_none());
        assert!(tools.get("calculator").is_none());
    }

    #[test]
    fn request_scope_can_narrow_but_never_widen_intent_tools() {
        let workspace = TestWorkspace::new();
        let inventory = ToolSet::for_request(
            Some(workspace.as_str()),
            Intent::InspectProject,
            Some(&["list_files"]),
        )
        .expect("narrow project inspection to inventory");
        assert_eq!(spec_names(&inventory), vec!["list_files"]);

        let error = ToolSet::for_request(
            Some(workspace.as_str()),
            Intent::InspectProject,
            Some(&["write_file"]),
        )
        .err()
        .expect("request scope must not add write access");
        assert!(error.contains("outside the allowed scope"));
    }

    #[test]
    fn empty_allowlist_does_not_touch_workspace() {
        let tools = ToolSet::with_allowlist(Some("/path/that/does/not/exist"), &[])
            .expect("empty tool set");
        assert!(tools.specs().is_empty());
    }

    #[test]
    fn legacy_search_allowlist_alias_stays_canonical() {
        let workspace = TestWorkspace::new();
        let tools = ToolSet::with_allowlist(Some(workspace.as_str()), &["search_text"])
            .expect("build search tool");
        assert_eq!(spec_names(&tools), vec!["search"]);
        assert!(tools.get("search").is_some());
        assert!(tools.get("search_text").is_some());
    }

    #[test]
    fn read_tools_expose_bench_schemas_and_reject_non_relative_paths() {
        let workspace = TestWorkspace::new();
        fs::create_dir(workspace.path().join("nested")).expect("create nested directory");
        fs::write(workspace.path().join("note.txt"), "verified\n").expect("seed file");
        let tools = ToolSet::with_allowlist(
            Some(workspace.as_str()),
            &["list_files", "read_file", "search"],
        )
        .expect("build read tools");

        assert_eq!(
            tools.get("list_files").expect("list tool").spec().arguments,
            json!({
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "max_depth": {"type": "integer"},
                "max_results": {"type": "integer"},
                "files_only": {"type": "boolean"}
            })
        );
        assert_eq!(
            tools.get("search").expect("search tool").spec().arguments,
            json!({"query": {"type": "string"}})
        );
        let reader = tools.get("read_file").expect("read tool");
        let absolute = reader
            .execute(&json!({"path": workspace.path().join("note.txt")}))
            .expect_err("absolute paths remain forbidden even inside the workspace");
        assert!(absolute.message.contains("workspace-relative"));
        let internal_parent = reader
            .execute(&json!({"path": "nested/../note.txt"}))
            .expect_err("parent components are forbidden");
        assert!(internal_parent.message.contains("escapes"));
    }

    #[test]
    fn read_file_chunks_cover_utf8_content_without_loss() {
        let workspace = TestWorkspace::new();
        let content = format!("{}中间标记{}", "a".repeat(700), "尾".repeat(400));
        fs::write(workspace.path().join("long.txt"), &content).expect("seed long file");
        let tools = ToolSet::with_allowlist(Some(workspace.as_str()), &["read_file"])
            .expect("build reader");
        let reader = tools.get("read_file").expect("read tool");
        let mut offset = 0u64;
        let mut rebuilt = String::new();
        loop {
            let chunk = reader
                .execute(&json!({"path": "long.txt", "offset": offset, "max_bytes": 256}))
                .expect("read chunk");
            rebuilt.push_str(chunk["content"].as_str().expect("chunk content"));
            offset = chunk["nextOffset"].as_u64().expect("next offset");
            if chunk["eof"] == true {
                break;
            }
        }
        assert_eq!(rebuilt, content);
        assert_eq!(offset, content.len() as u64);
    }

    #[test]
    fn write_file_overwrites_and_creates_from_line_arrays() {
        let workspace = TestWorkspace::new();
        fs::write(workspace.path().join("app.conf"), "mode = development\n").expect("seed config");
        fs::create_dir(workspace.path().join("nested")).expect("create nested directory");
        let tools = ToolSet::with_allowlist(Some(workspace.as_str()), &["write_file"])
            .expect("build writer");
        let writer = tools.get("write_file").expect("writer registered");

        let overwritten = writer
            .execute(&normalize_arguments(&json!({
                "file_path": "app.conf",
                "content": "mode = production\n"
            })))
            .expect("overwrite config");
        assert_eq!(overwritten["path"], "app.conf");
        assert_eq!(overwritten["created"], false);
        assert_eq!(
            fs::read_to_string(workspace.path().join("app.conf")).expect("read config"),
            "mode = production\n"
        );

        let created = writer
            .execute(&json!({
                "path": "nested/new.txt",
                "content": ["alpha", "beta"]
            }))
            .expect("create file");
        assert_eq!(created["path"], "nested/new.txt");
        assert_eq!(created["created"], true);
        assert_eq!(
            fs::read_to_string(workspace.path().join("nested/new.txt")).expect("read new file"),
            "alpha\nbeta\n"
        );
    }

    #[test]
    fn write_file_rejects_unsafe_or_large_content() {
        let workspace = TestWorkspace::new();
        let tools = ToolSet::with_allowlist(Some(workspace.as_str()), &["write_file"])
            .expect("build writer");
        let writer = tools.get("write_file").expect("writer registered");

        let traversal = writer
            .execute(&json!({"path": "../escape.txt", "content": "no"}))
            .expect_err("reject traversal");
        assert!(traversal.message.contains("escapes"));
        let absolute = writer
            .execute(&json!({"path": "/tmp/escape.txt", "content": "no"}))
            .expect_err("reject absolute path");
        assert!(absolute.message.contains("workspace-relative"));
        let oversized = "x".repeat(64 * 1024 + 1);
        let too_large = writer
            .execute(&json!({"path": "large.txt", "content": oversized}))
            .expect_err("reject large content");
        assert!(too_large.message.contains("write limit"));
        let invalid_lines = writer
            .execute(&json!({"path": "bad.txt", "content": ["ok", 7]}))
            .expect_err("reject non-string line");
        assert!(invalid_lines.invalid_arguments);
    }

    #[cfg(unix)]
    #[test]
    fn write_file_rejects_symlink_targets_and_outside_parents() {
        use std::os::unix::fs::symlink;

        let workspace = TestWorkspace::new();
        let outside = TestWorkspace::new();
        fs::write(outside.path().join("outside.txt"), "safe\n").expect("seed outside file");
        symlink(
            outside.path().join("outside.txt"),
            workspace.path().join("linked.txt"),
        )
        .expect("create file symlink");
        symlink(outside.path(), workspace.path().join("outside-dir"))
            .expect("create directory symlink");
        let tools = ToolSet::with_allowlist(Some(workspace.as_str()), &["write_file"])
            .expect("build writer");
        let writer = tools.get("write_file").expect("writer registered");

        let linked = writer
            .execute(&json!({"path": "linked.txt", "content": "changed\n"}))
            .expect_err("reject file symlink");
        assert!(linked.message.contains("symbolic-link"));
        let outside_parent = writer
            .execute(&json!({"path": "outside-dir/new.txt", "content": "changed\n"}))
            .expect_err("reject outside parent");
        assert!(outside_parent.message.contains("escapes"));
        assert_eq!(
            fs::read_to_string(outside.path().join("outside.txt")).expect("read outside file"),
            "safe\n"
        );
    }
}
