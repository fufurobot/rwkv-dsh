use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub base_url: String,
    pub password: String,
    pub model: String,
    pub workspace: String,
    /// Kept for settings compatibility; runtime policy always normalizes this to `"off"`.
    pub thinking: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub alpha_presence: f32,
    pub alpha_frequency: f32,
    pub alpha_decay: f32,
}

/// 默认 rwkv_lightning 地址与 13.3B 检查点（`GET /v1/models`）.
const DOCUMENTED_BASE_URL: &str = "http://192.168.0.115:8030/v1";
const DOCUMENTED_MODEL: &str = "rwkv7-g1i-13.3b-20260805-ctx16384";
const LEGACY_BASE_URLS: &[&str] = &[
    "http://127.0.0.1:8000/v1",
    "http://192.168.0.125:8001/v1",
    "http://192.168.0.125:8018/v1",
    "http://192.168.0.115:8000/v1",
    "https://api-7b.rwkvos.com/v1",
    "https://api-13b.rwkvos.com/v1",
];
const LEGACY_MODELS: &[&str] = &[
    "rwkv7-g1i",
    "rwkv-g1i-13b-4922",
    "rwkv7-g1i-7.2b-20260805-ctx16384",
];
const LEGACY_7B_PASSWORD: &str = "rwkv7_7.2b";
const DEFAULT_MAX_TOKENS: u32 = 8192;
const PREVIOUS_DEFAULT_MAX_TOKENS: u32 = 6144;
const MIN_MAX_TOKENS: u32 = 256;
const MAX_MAX_TOKENS: u32 = 16_384;

fn clamp_max_tokens(value: u32) -> u32 {
    value.clamp(MIN_MAX_TOKENS, MAX_MAX_TOKENS)
}

fn default_workspace() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| env!("CARGO_MANIFEST_DIR").into())
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            base_url: DOCUMENTED_BASE_URL.into(),
            password: String::new(),
            model: DOCUMENTED_MODEL.into(),
            workspace: default_workspace(),
            thinking: "off".into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: 0.001,
            top_k: 50,
            top_p: 1.0,
            alpha_presence: 0.0,
            alpha_frequency: 0.0,
            alpha_decay: 0.99,
        }
    }
}

impl Settings {
    pub fn thinking_fast(&self) -> bool {
        self.thinking.eq_ignore_ascii_case("fast")
    }

    pub fn load(path: &Path) -> Self {
        let Ok(raw) = fs::read_to_string(path) else {
            return Self::default();
        };
        let mut settings: Self = serde_json::from_str(&raw).unwrap_or_default();
        let mut dirty = false;
        let base = settings.base_url.trim();
        if LEGACY_BASE_URLS.contains(&base) {
            settings.base_url = DOCUMENTED_BASE_URL.into();
            dirty = true;
        }
        if LEGACY_MODELS.contains(&settings.model.trim()) {
            settings.model = DOCUMENTED_MODEL.into();
            dirty = true;
        }
        if settings.password.trim() == LEGACY_7B_PASSWORD {
            settings.password.clear();
            dirty = true;
        }
        if settings.thinking != "off" {
            settings.thinking = "off".into();
            dirty = true;
        }
        if settings.max_tokens == PREVIOUS_DEFAULT_MAX_TOKENS {
            settings.max_tokens = DEFAULT_MAX_TOKENS;
            dirty = true;
        }
        let max_tokens = clamp_max_tokens(settings.max_tokens);
        if settings.max_tokens != max_tokens {
            settings.max_tokens = max_tokens;
            dirty = true;
        }
        if settings.workspace.trim().is_empty() {
            settings.workspace = default_workspace();
            dirty = true;
        }
        if dirty {
            let _ = settings.save(path);
        }
        settings
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        let mut settings = self.clone();
        settings.thinking = "off".into();
        settings.max_tokens = clamp_max_tokens(settings.max_tokens);
        let encoded = serde_json::to_string_pretty(&settings).map_err(|err| err.to_string())?;
        fs::write(path, encoded).map_err(|err| err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{Settings, DEFAULT_MAX_TOKENS, MAX_MAX_TOKENS, MIN_MAX_TOKENS};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEMP_SETTINGS: AtomicU64 = AtomicU64::new(1);

    fn temporary_settings_path() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let sequence = NEXT_TEMP_SETTINGS.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rwkv-agent-settings-{}-{nonce}-{sequence}.json",
            std::process::id(),
        ))
    }

    fn write_settings(path: &std::path::Path, settings: &Settings) {
        fs::write(
            path,
            serde_json::to_string_pretty(settings).expect("serialize settings"),
        )
        .expect("write settings");
    }

    fn read_settings(path: &std::path::Path) -> Settings {
        serde_json::from_str(&fs::read_to_string(path).expect("read settings"))
            .expect("parse settings")
    }

    #[test]
    fn defaults_disable_thinking_and_use_official_token_default() {
        let settings = Settings::default();
        assert_eq!(settings.thinking, "off");
        assert_eq!(settings.max_tokens, 8192);
        assert_eq!(settings.base_url, "http://192.168.0.115:8030/v1");
        assert_eq!(settings.model, "rwkv7-g1i-13.3b-20260805-ctx16384");
        assert!(settings.password.is_empty());
    }

    #[test]
    fn load_migrates_previous_lan_and_cloud_defaults() {
        let path = temporary_settings_path();
        let mut previous = Settings::default();
        previous.base_url = "http://192.168.0.125:8018/v1".into();
        previous.model = "rwkv7-g1i-7.2b-20260805-ctx16384".into();
        write_settings(&path, &previous);

        let loaded = Settings::load(&path);
        assert_eq!(loaded.base_url, "http://192.168.0.115:8030/v1");
        assert_eq!(loaded.model, "rwkv7-g1i-13.3b-20260805-ctx16384");
        assert_eq!(read_settings(&path).base_url, loaded.base_url);
        assert_eq!(read_settings(&path).model, loaded.model);
        fs::remove_file(path).expect("remove temporary settings");
    }

    #[test]
    fn load_migrates_previous_defaults() {
        let path = temporary_settings_path();
        let mut previous = Settings::default();
        previous.thinking = "fast".into();
        previous.max_tokens = 6144;
        write_settings(&path, &previous);

        let loaded = Settings::load(&path);
        assert_eq!(loaded.thinking, "off");
        assert_eq!(loaded.max_tokens, DEFAULT_MAX_TOKENS);

        let persisted = read_settings(&path);
        assert_eq!(persisted.thinking, "off");
        assert_eq!(persisted.max_tokens, DEFAULT_MAX_TOKENS);
        fs::remove_file(path).expect("remove temporary settings");
    }

    #[test]
    fn load_preserves_valid_custom_token_budgets() {
        for max_tokens in [256, 512, 4096] {
            let path = temporary_settings_path();
            let mut settings = Settings::default();
            settings.max_tokens = max_tokens;
            write_settings(&path, &settings);

            let loaded = Settings::load(&path);
            assert_eq!(loaded.max_tokens, max_tokens);
            assert_eq!(read_settings(&path).max_tokens, max_tokens);
            fs::remove_file(path).expect("remove temporary settings");
        }
    }

    #[test]
    fn load_clamps_token_budget_to_ui_bounds() {
        for (max_tokens, expected) in [
            (0, MIN_MAX_TOKENS),
            (MIN_MAX_TOKENS - 1, MIN_MAX_TOKENS),
            (MAX_MAX_TOKENS + 1, MAX_MAX_TOKENS),
            (u32::MAX, MAX_MAX_TOKENS),
        ] {
            let path = temporary_settings_path();
            let mut settings = Settings::default();
            settings.max_tokens = max_tokens;
            write_settings(&path, &settings);

            let loaded = Settings::load(&path);
            assert_eq!(loaded.max_tokens, expected);
            assert_eq!(read_settings(&path).max_tokens, expected);
            fs::remove_file(path).expect("remove temporary settings");
        }
    }

    #[test]
    fn save_disables_thinking_and_clamps_token_budget() {
        for (max_tokens, expected) in [
            (0, MIN_MAX_TOKENS),
            (4096, 4096),
            (u32::MAX, MAX_MAX_TOKENS),
        ] {
            let path = temporary_settings_path();
            let mut settings = Settings::default();
            settings.thinking = "fast".into();
            settings.max_tokens = max_tokens;

            settings.save(&path).expect("save normalized settings");

            let persisted = read_settings(&path);
            assert_eq!(persisted.thinking, "off");
            assert_eq!(persisted.max_tokens, expected);
            fs::remove_file(path).expect("remove temporary settings");
        }
    }
}
