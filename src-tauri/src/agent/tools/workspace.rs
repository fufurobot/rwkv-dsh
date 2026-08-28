use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use super::{Tool, ToolError};
use crate::agent::protocol::ToolSpec;

const MAX_READ_BYTES: usize = 64 * 1024;
const MAX_WRITE_BYTES: usize = 64 * 1024;
const MAX_SEARCH_FILE_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Clone)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn open(root: &str) -> Result<Self, String> {
        let absolute = PathBuf::from(root);
        let absolute = if absolute.is_absolute() {
            absolute
        } else {
            std::env::current_dir()
                .map_err(|err| err.to_string())?
                .join(absolute)
        };
        let resolved = fs::canonicalize(&absolute)
            .map_err(|err| format!("resolve workspace {}: {err}", absolute.display()))?;
        if !resolved.is_dir() {
            return Err("workspace is not a directory".into());
        }
        Ok(Self { root: resolved })
    }

    pub fn root_display(&self) -> String {
        self.root.to_string_lossy().into_owned()
    }

    pub fn into_tools(self, include_write: bool) -> Vec<Box<dyn Tool>> {
        let mut tools: Vec<Box<dyn Tool>> = vec![
            Box::new(ListFiles {
                workspace: self.clone(),
            }),
            Box::new(ReadFile {
                workspace: self.clone(),
            }),
            Box::new(SearchText {
                workspace: self.clone(),
            }),
        ];
        if include_write {
            tools.push(Box::new(WriteFile { workspace: self }));
        }
        tools
    }

    fn resolve(&self, path: &str) -> Result<PathBuf, ToolError> {
        let path = if path.is_empty() { "." } else { path };
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            let relative = candidate
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(".");
            return Err(ToolError::invalid(format!(
                "path must be workspace-relative, such as \"{relative}\"; got absolute path \"{path}\""
            )));
        }
        self.resolve_relative(path)
    }

    fn resolve_relative(&self, path: &str) -> Result<PathBuf, ToolError> {
        let clean = clean_relative_path(path)?;
        let target = self.root.join(&clean);
        let resolved =
            fs::canonicalize(&target).map_err(|err| ToolError::runtime(err.to_string()))?;
        if !resolved.starts_with(&self.root) {
            return Err(ToolError::runtime("path escapes the workspace"));
        }
        Ok(resolved)
    }

    fn resolve_write_target(&self, path: &str) -> Result<PathBuf, ToolError> {
        if path.trim().is_empty() || path == "." {
            return Err(ToolError::invalid("path is required"));
        }
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            return Err(ToolError::invalid("path must be workspace-relative"));
        }
        let clean = clean_relative_path(path)?;
        if clean.as_os_str().is_empty() {
            return Err(ToolError::invalid("path is required"));
        }
        let target = self.root.join(clean);
        let parent = target
            .parent()
            .ok_or_else(|| ToolError::invalid("path needs a parent directory"))?;
        let resolved_parent = fs::canonicalize(parent)
            .map_err(|err| ToolError::runtime(format!("resolve parent directory: {err}")))?;
        if !resolved_parent.starts_with(&self.root) {
            return Err(ToolError::runtime("path escapes the workspace"));
        }
        let file_name = target
            .file_name()
            .ok_or_else(|| ToolError::invalid("path must name a file"))?;
        let resolved_target = resolved_parent.join(file_name);
        match fs::symlink_metadata(&resolved_target) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                Err(ToolError::runtime("symbolic-link writes are not supported"))
            }
            Ok(metadata) if !metadata.is_file() => {
                Err(ToolError::runtime("path is not a regular file"))
            }
            Ok(_) => {
                let canonical = fs::canonicalize(&resolved_target)
                    .map_err(|err| ToolError::runtime(err.to_string()))?;
                if !canonical.starts_with(&self.root) {
                    return Err(ToolError::runtime("path escapes the workspace"));
                }
                Ok(canonical)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(resolved_target),
            Err(error) => Err(ToolError::runtime(error.to_string())),
        }
    }

    fn relative(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .map(|value| value.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| ".".into())
    }

    pub fn list_dir(&self, relative: &str) -> Result<Vec<WorkspaceEntry>, String> {
        let target = if relative.is_empty() || relative == "." {
            self.root.clone()
        } else {
            self.resolve(relative).map_err(|err| err.to_string())?
        };
        if !target.is_dir() {
            return Err("path is not a directory".into());
        }
        let mut entries = Vec::new();
        let read = fs::read_dir(&target).map_err(|err| err.to_string())?;
        for item in read.flatten() {
            let name = item.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || should_skip_directory(&name) {
                continue;
            }
            let is_dir = item.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
            let path = self.relative(&item.path());
            entries.push(WorkspaceEntry {
                name,
                path,
                kind: if is_dir {
                    "directory".into()
                } else {
                    "file".into()
                },
            });
            if entries.len() >= 200 {
                break;
            }
        }
        entries.sort_by(|a, b| match (a.kind.as_str(), b.kind.as_str()) {
            ("directory", "file") => std::cmp::Ordering::Less,
            ("file", "directory") => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        });
        Ok(entries)
    }
}

fn clean_relative_path(path: &str) -> Result<PathBuf, ToolError> {
    let mut clean = PathBuf::new();
    for component in Path::new(path).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => clean.push(part),
            Component::ParentDir => return Err(ToolError::runtime("path escapes the workspace")),
            _ => return Err(ToolError::runtime("path escapes the workspace")),
        }
    }
    Ok(clean)
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceEntry {
    pub name: String,
    pub path: String,
    pub kind: String,
}

struct ListFiles {
    workspace: Workspace,
}

#[derive(Deserialize, Default)]
struct ListFilesArgs {
    #[serde(default)]
    path: String,
    #[serde(default)]
    offset: u64,
    #[serde(default)]
    max_depth: i64,
    #[serde(default)]
    max_results: i64,
    #[serde(default)]
    files_only: bool,
}

impl Tool for ListFiles {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_files".into(),
            description: "List files and directories in the bound workspace with deterministic offset pagination. Use path \".\" or omit path for the project root. Paths are workspace-relative. Generated and VCS directories are skipped.".into(),
            arguments: json!({
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "max_depth": {"type": "integer"},
                "max_results": {"type": "integer"},
                "files_only": {"type": "boolean"}
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        let mut args: ListFilesArgs = serde_json::from_value(arguments.clone())
            .map_err(|err| ToolError::invalid(format!("{err}")))?;
        if args.max_depth == 0 {
            args.max_depth = 2;
        }
        if args.max_results == 0 {
            args.max_results = 50;
        }
        args.max_depth = args.max_depth.clamp(1, 8);
        args.max_results = args.max_results.clamp(1, 80);
        let target = self.workspace.resolve(&args.path)?;
        if !target.is_dir() {
            return Err(ToolError::runtime("path is not a directory"));
        }
        let mut entries = Vec::new();
        let mut state = ListPageState::default();
        walk_list(
            &self.workspace,
            &target,
            1,
            args.max_depth as usize,
            args.offset,
            args.max_results as usize,
            args.files_only,
            &mut entries,
            &mut state,
        )?;
        let next_offset = args.offset.saturating_add(entries.len() as u64);
        Ok(json!({
            "workspace": self.workspace.root_display(),
            "path": self.workspace.relative(&target),
            "entries": entries,
            "truncated": state.has_more || state.depth_limited,
            "hasMore": state.has_more,
            "depthLimited": state.depth_limited,
            "nextOffset": next_offset,
        }))
    }
}

#[derive(Default)]
struct ListPageState {
    seen_paths: HashSet<String>,
    visited_entries: u64,
    has_more: bool,
    depth_limited: bool,
}

fn walk_list(
    workspace: &Workspace,
    dir: &Path,
    depth: usize,
    max_depth: usize,
    offset: u64,
    max_results: usize,
    files_only: bool,
    entries: &mut Vec<Value>,
    state: &mut ListPageState,
) -> Result<(), ToolError> {
    let mut queue = VecDeque::from([(dir.to_path_buf(), depth)]);
    while let Some((current, current_depth)) = queue.pop_front() {
        let read = fs::read_dir(&current).map_err(|err| ToolError::runtime(err.to_string()))?;
        let mut items: Vec<_> = read.filter_map(|entry| entry.ok()).collect();
        items.sort_by_key(|entry| entry.file_name());
        let mut nested = Vec::new();
        for entry in items {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let path = entry.path();
            let is_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
            if should_omit_entry(name, is_dir) {
                continue;
            }
            let Some(relative) = path
                .strip_prefix(&workspace.root)
                .ok()
                .and_then(Path::to_str)
                .map(|value| value.replace('\\', "/"))
            else {
                continue;
            };
            if !is_dir || !files_only {
                if state.seen_paths.insert(relative.clone()) {
                    let index = state.visited_entries;
                    state.visited_entries = state.visited_entries.saturating_add(1);
                    if index >= offset {
                        if entries.len() < max_results {
                            let mut item = json!({
                                "path": relative,
                                "type": if is_dir { "directory" } else { "file" },
                            });
                            if !is_dir {
                                if let Ok(meta) = entry.metadata() {
                                    item["size"] = json!(meta.len());
                                }
                            }
                            entries.push(item);
                        } else {
                            state.has_more = true;
                        }
                    }
                }
            }
            if is_dir && current_depth < max_depth && !should_not_recurse(name) {
                nested.push((path, current_depth + 1));
            } else if is_dir && !should_not_recurse(name) {
                state.depth_limited = true;
            }
        }
        queue.extend(nested);
    }
    Ok(())
}

struct ReadFile {
    workspace: Workspace,
}

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
    #[serde(default)]
    offset: u64,
    #[serde(default)]
    max_bytes: Option<usize>,
}

impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: "Read one UTF-8 text file inside the workspace, up to 64 KiB.".into(),
            arguments: json!({
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "max_bytes": {"type": "integer"}
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        let args: ReadFileArgs = serde_json::from_value(arguments.clone())
            .map_err(|err| ToolError::invalid(format!("{err}")))?;
        if args.path.trim().is_empty() {
            return Err(ToolError::invalid("path is required"));
        }
        let target = self.workspace.resolve(&args.path)?;
        let meta = fs::metadata(&target).map_err(|err| ToolError::runtime(err.to_string()))?;
        if !meta.is_file() {
            return Err(ToolError::runtime("path is not a regular file"));
        }
        let total_bytes = meta.len();
        if args.offset > total_bytes {
            return Err(ToolError::invalid("offset is past the end of the file"));
        }
        let max_bytes = args
            .max_bytes
            .unwrap_or(MAX_READ_BYTES)
            .clamp(256, MAX_READ_BYTES);
        let mut file =
            fs::File::open(&target).map_err(|err| ToolError::runtime(err.to_string()))?;
        file.seek(SeekFrom::Start(args.offset))
            .map_err(|err| ToolError::runtime(err.to_string()))?;
        let mut data = Vec::new();
        file.take(max_bytes as u64 + 4)
            .read_to_end(&mut data)
            .map_err(|err| ToolError::runtime(err.to_string()))?;
        if data.contains(&0) {
            return Err(ToolError::runtime("binary files are not supported"));
        }
        if args.offset < total_bytes
            && data
                .first()
                .is_some_and(|byte| byte & 0b1100_0000 == 0b1000_0000)
        {
            return Err(ToolError::invalid(
                "offset must be on a UTF-8 character boundary",
            ));
        }
        let mut end = data.len().min(max_bytes);
        while end > 0 && end < data.len() && data[end] & 0b1100_0000 == 0b1000_0000 {
            end -= 1;
        }
        let content = std::str::from_utf8(&data[..end])
            .map_err(|_| ToolError::runtime("file is not valid UTF-8 text"))?;
        let next_offset = args.offset + end as u64;
        let eof = next_offset >= total_bytes;
        if !eof && end == 0 {
            return Err(ToolError::runtime(
                "unable to advance the UTF-8 read cursor",
            ));
        }
        Ok(json!({
            "path": self.workspace.relative(&target),
            "workspace": self.workspace.root_display(),
            "content": content,
            "offset": args.offset,
            "nextOffset": next_offset,
            "totalBytes": total_bytes,
            "eof": eof,
            "truncated": !eof,
        }))
    }
}

struct SearchText {
    workspace: Workspace,
}

#[derive(Deserialize)]
struct SearchTextArgs {
    query: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    case_sensitive: bool,
    #[serde(default)]
    max_results: i64,
}

impl Tool for SearchText {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "search".into(),
            description:
                "Search literal text in workspace files. Generated and VCS directories are skipped."
                    .into(),
            arguments: json!({"query": {"type": "string"}}),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        let mut args: SearchTextArgs = serde_json::from_value(arguments.clone())
            .map_err(|err| ToolError::invalid(format!("{err}")))?;
        if args.query.is_empty() {
            return Err(ToolError::invalid("query is required"));
        }
        if args.max_results == 0 {
            args.max_results = 50;
        }
        if !(1..=200).contains(&args.max_results) {
            return Err(ToolError::invalid("max_results must be between 1 and 200"));
        }
        let target = self.workspace.resolve(&args.path)?;
        let mut matches = Vec::new();
        let mut truncated = false;
        search_walk(
            &self.workspace,
            &target,
            &args.query,
            args.case_sensitive,
            args.max_results as usize,
            &mut matches,
            &mut truncated,
        )?;
        Ok(json!({
            "matches": matches,
            "truncated": truncated,
        }))
    }
}

struct WriteFile {
    workspace: Workspace,
}

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: WriteContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WriteContent {
    Text(String),
    Lines(Vec<String>),
}

impl WriteContent {
    fn into_text(self) -> String {
        match self {
            Self::Text(content) => content,
            Self::Lines(lines) => {
                let mut content = lines.join("\n");
                if !content.is_empty() {
                    content.push('\n');
                }
                content
            }
        }
    }
}

impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Overwrite one small UTF-8 text file inside the workspace. content is the complete file as a string or an array of line strings.".into(),
            arguments: json!({
                "path": {"type": "string"},
                "content": {"type": ["string", "array"], "items": {"type": "string"}}
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        let args: WriteFileArgs = serde_json::from_value(arguments.clone())
            .map_err(|err| ToolError::invalid(format!("invalid write_file arguments: {err}")))?;
        let content = args.content.into_text();
        if content.len() > MAX_WRITE_BYTES {
            return Err(ToolError::invalid(format!(
                "content exceeds the {MAX_WRITE_BYTES}-byte write limit"
            )));
        }
        if content.contains('\0') {
            return Err(ToolError::invalid("binary content is not supported"));
        }
        let target = self.workspace.resolve_write_target(&args.path)?;
        let created = !target.exists();
        fs::write(&target, content.as_bytes())
            .map_err(|err| ToolError::runtime(err.to_string()))?;
        Ok(json!({
            "path": self.workspace.relative(&target),
            "workspace": self.workspace.root_display(),
            "bytes": content.len(),
            "created": created,
        }))
    }
}

fn search_walk(
    workspace: &Workspace,
    path: &Path,
    query: &str,
    case_sensitive: bool,
    limit: usize,
    matches: &mut Vec<Value>,
    truncated: &mut bool,
) -> Result<(), ToolError> {
    let meta = fs::metadata(path).map_err(|err| ToolError::runtime(err.to_string()))?;
    if meta.is_file() {
        search_file(workspace, path, query, case_sensitive, limit, matches)?;
        if matches.len() >= limit {
            *truncated = true;
        }
        return Ok(());
    }
    if !meta.is_dir() {
        return Ok(());
    }
    let read = fs::read_dir(path).map_err(|err| ToolError::runtime(err.to_string()))?;
    for entry in read.flatten() {
        if matches.len() >= limit {
            *truncated = true;
            return Ok(());
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let child = entry.path();
        let is_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
        if should_omit_entry(&name, is_dir) {
            continue;
        }
        if is_dir {
            if should_not_recurse(&name) {
                continue;
            }
            search_walk(
                workspace,
                &child,
                query,
                case_sensitive,
                limit,
                matches,
                truncated,
            )?;
            continue;
        }
        if let Ok(meta) = entry.metadata() {
            if !meta.is_file() || meta.len() > MAX_SEARCH_FILE_BYTES {
                continue;
            }
        }
        search_file(workspace, &child, query, case_sensitive, limit, matches)?;
    }
    Ok(())
}

fn search_file(
    workspace: &Workspace,
    path: &Path,
    query: &str,
    case_sensitive: bool,
    limit: usize,
    matches: &mut Vec<Value>,
) -> Result<(), ToolError> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Ok(()),
    };
    let reader = BufReader::new(file);
    let needle = if case_sensitive {
        query.to_string()
    } else {
        query.to_lowercase()
    };
    for (index, line) in reader.lines().enumerate() {
        if matches.len() >= limit {
            break;
        }
        let Ok(line) = line else {
            continue;
        };
        if line.contains('\0') {
            return Ok(());
        }
        let haystack = if case_sensitive {
            line.clone()
        } else {
            line.to_lowercase()
        };
        if haystack.contains(&needle) {
            let text: String = line.trim().chars().take(300).collect();
            matches.push(json!({
                "path": workspace.relative(path),
                "line": index + 1,
                "text": text,
            }));
        }
    }
    Ok(())
}

fn should_omit_entry(name: &str, is_dir: bool) -> bool {
    if name == "." || name == ".." {
        return true;
    }
    if name.starts_with('.') && name != ".gitignore" {
        return true;
    }
    is_dir && should_omit_directory(name)
}

fn should_omit_directory(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".cursor"
            | ".venv"
            | "venv"
            | "__pycache__"
            | ".mypy_cache"
            | ".pytest_cache"
            | "build"
            | "dist"
            | "node_modules"
            | "target"
    )
}

fn should_not_recurse(name: &str) -> bool {
    should_omit_directory(name)
        || name == "runs"
        || name.starts_with("agent_cases")
        || matches!(name, "vendor" | "fixtures" | "testdata")
}

fn should_skip_directory(name: &str) -> bool {
    should_omit_directory(name) || should_not_recurse(name)
}

#[cfg(test)]
mod tests {
    use super::{should_not_recurse, should_omit_entry, ListFiles, Tool, Workspace};
    use serde_json::json;
    use std::collections::HashSet;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_WORKSPACE: AtomicU64 = AtomicU64::new(1);

    #[test]
    fn skips_dotfiles_and_case_dumps() {
        assert!(should_omit_entry(".eslintrc.js", false));
        assert!(should_omit_entry(".claude", true));
        assert!(!should_omit_entry(".gitignore", false));
        assert!(!should_omit_entry("README.md", false));
        assert!(should_not_recurse("agent_cases"));
        assert!(should_not_recurse("agent_cases_hard9"));
        assert!(should_not_recurse("runs"));
        assert!(!should_not_recurse("src"));
    }

    #[test]
    fn files_only_listing_reports_result_and_depth_limits() {
        let sequence = NEXT_WORKSPACE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rwkv-agent-list-files-{}-{sequence}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("nested")).expect("create workspace");
        fs::write(root.join("a.txt"), "a").expect("write a");
        fs::write(root.join("b.txt"), "b").expect("write b");
        fs::write(root.join("nested/c.txt"), "c").expect("write c");
        let tool = ListFiles {
            workspace: Workspace::open(&root.to_string_lossy()).expect("open workspace"),
        };

        let limited = tool
            .execute(&json!({
                "path": ".",
                "max_depth": 8,
                "max_results": 1,
                "files_only": true
            }))
            .expect("limited listing");
        assert_eq!(limited["entries"].as_array().map(Vec::len), Some(1));
        assert_eq!(limited["truncated"], true);
        assert_eq!(limited["hasMore"], true);
        assert_eq!(limited["depthLimited"], false);
        assert_eq!(limited["nextOffset"], 1);
        assert!(limited["entries"]
            .as_array()
            .expect("entries")
            .iter()
            .all(|entry| entry["type"] == "file"));

        let shallow = tool
            .execute(&json!({
                "path": ".",
                "max_depth": 1,
                "max_results": 80,
                "files_only": true
            }))
            .expect("shallow listing");
        assert_eq!(shallow["entries"].as_array().map(Vec::len), Some(2));
        assert_eq!(shallow["truncated"], true);
        assert_eq!(shallow["hasMore"], false);
        assert_eq!(shallow["depthLimited"], true);
        assert_eq!(shallow["nextOffset"], 2);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn paginated_listings_reconstruct_full_order_without_duplicates() {
        let sequence = NEXT_WORKSPACE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rwkv-agent-paginated-list-files-{}-{sequence}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("docs/指南")).expect("create docs");
        fs::create_dir_all(root.join("src/nested")).expect("create src");
        for (path, content) in [
            ("a.txt", "a"),
            ("中.txt", "middle"),
            ("z.txt", "z"),
            ("docs/guide.md", "guide"),
            ("docs/指南/说明.md", "unicode"),
            ("src/main.rs", "fn main() {}"),
            ("src/nested/lib.rs", "pub fn item() {}"),
        ] {
            fs::write(root.join(path), content).expect("write fixture");
        }
        let tool = ListFiles {
            workspace: Workspace::open(&root.to_string_lossy()).expect("open workspace"),
        };

        for files_only in [false, true] {
            let full = tool
                .execute(&json!({
                    "path": ".",
                    "max_depth": 8,
                    "max_results": 80,
                    "files_only": files_only
                }))
                .expect("full listing");
            assert_eq!(full["hasMore"], false);
            assert_eq!(full["depthLimited"], false);
            assert_eq!(full["truncated"], false);
            let expected = full["entries"].as_array().expect("full entries").clone();
            assert!(expected.len() > 4);

            let mut offset = 0u64;
            let mut pages = 0usize;
            let mut rebuilt = Vec::new();
            loop {
                let page = tool
                    .execute(&json!({
                        "path": ".",
                        "offset": offset,
                        "max_depth": 8,
                        "max_results": 2,
                        "files_only": files_only
                    }))
                    .expect("listing page");
                pages += 1;
                assert!(pages < 20, "pagination must terminate");
                assert_eq!(page["depthLimited"], false);
                let page_entries = page["entries"].as_array().expect("page entries");
                rebuilt.extend(page_entries.iter().cloned());
                let next_offset = page["nextOffset"].as_u64().expect("next offset");
                assert_eq!(next_offset, offset + page_entries.len() as u64);
                if page["hasMore"] == false {
                    assert_eq!(page["truncated"], false);
                    break;
                }
                assert_eq!(page["truncated"], true);
                assert!(next_offset > offset);
                offset = next_offset;
            }

            assert!(pages >= 3);
            assert_eq!(rebuilt, expected);
            let paths = rebuilt
                .iter()
                .map(|entry| {
                    entry["path"]
                        .as_str()
                        .expect("UTF-8 workspace-relative path")
                        .to_string()
                })
                .collect::<Vec<_>>();
            assert!(paths.iter().any(|path| path.contains('中')));
            assert!(paths.iter().any(|path| path.contains("指南")));
            assert_eq!(
                paths.iter().cloned().collect::<HashSet<_>>().len(),
                paths.len()
            );
            assert!(rebuilt
                .iter()
                .all(|entry| !files_only || entry["type"] == "file"));
        }

        let _ = fs::remove_dir_all(root);
    }
}
