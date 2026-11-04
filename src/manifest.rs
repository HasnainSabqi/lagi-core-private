/*!
Manifest Schema Validation for .lagi Format

Strict serde-based validation, supporting both the legacy `model` section and
the lightweight LAGI-Edge-1B runtime sections (`model_config`,
`ideation_loop`, `memory_config`).
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Legacy model configuration for GGUF models
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelConfig {
    pub series: String,
    pub gguf_file: String,
    #[serde(default = "default_quantization")]
    pub quantization: String,
    #[serde(default = "default_context_window")]
    pub context_window: u32,
}

fn default_quantization() -> String {
    "Q5_K_M".to_string()
}

fn default_context_window() -> u32 {
    8192
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.series.is_empty() {
            return Err(ManifestError::MissingField("model.series".to_string()));
        }
        if self.gguf_file.is_empty() {
            return Err(ManifestError::MissingField("model.gguf_file".to_string()));
        }
        if !self.gguf_file.ends_with(".gguf") {
            return Err(ManifestError::InvalidField(
                "model.gguf_file must end with .gguf".to_string(),
            ));
        }
        if self.context_window < 1024 || self.context_window > 131072 {
            return Err(ManifestError::InvalidField(
                "context_window must be between 1024 and 131072".to_string(),
            ));
        }
        Ok(())
    }
}

/// Lightweight runtime model configuration (LAGI-Edge-1B, CPU-first)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRuntimeConfig {
    pub gguf_filename: String,
    #[serde(default = "default_context_window")]
    pub context_window: u32,
    #[serde(default = "default_cpu_threads")]
    pub cpu_threads: u32,
    #[serde(default)]
    pub use_gpu: bool,
}

fn default_cpu_threads() -> u32 {
    4
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            gguf_filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
            context_window: 8192,
            cpu_threads: 4,
            use_gpu: false,
        }
    }
}

impl ModelRuntimeConfig {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.gguf_filename.is_empty() {
            return Err(ManifestError::MissingField(
                "model_config.gguf_filename".to_string(),
            ));
        }
        if !self.gguf_filename.ends_with(".gguf") {
            return Err(ManifestError::InvalidField(
                "model_config.gguf_filename must end with .gguf".to_string(),
            ));
        }
        if self.context_window < 1024 || self.context_window > 131072 {
            return Err(ManifestError::InvalidField(
                "model_config.context_window must be between 1024 and 131072".to_string(),
            ));
        }
        if self.cpu_threads == 0 {
            return Err(ManifestError::InvalidField(
                "model_config.cpu_threads must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Legacy ideation loop configuration (seconds-scale interval)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct IdeationLoopConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_interval")]
    pub interval_seconds: u32,
    #[serde(default = "default_spark_temp")]
    pub spark_temperature: f32,
    #[serde(default = "default_critic_temp")]
    pub critic_temperature: f32,
    #[serde(default = "default_min_score")]
    pub min_confidence_score: u8,
}

fn default_true() -> bool {
    true
}
fn default_interval() -> u32 {
    1800
}
fn default_spark_temp() -> f32 {
    0.85
}
fn default_critic_temp() -> f32 {
    0.20
}
fn default_min_score() -> u8 {
    80
}

impl Default for IdeationLoopConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_seconds: 1800,
            spark_temperature: 0.85,
            critic_temperature: 0.20,
            min_confidence_score: 80,
        }
    }
}

impl IdeationLoopConfig {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.interval_seconds < 60 || self.interval_seconds > 86400 {
            return Err(ManifestError::InvalidField(
                "interval_seconds must be between 60 and 86400".to_string(),
            ));
        }
        if self.spark_temperature < 0.0 || self.spark_temperature > 2.0 {
            return Err(ManifestError::InvalidField(
                "spark_temperature must be between 0.0 and 2.0".to_string(),
            ));
        }
        if self.critic_temperature < 0.0 || self.critic_temperature > 1.0 {
            return Err(ManifestError::InvalidField(
                "critic_temperature must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.min_confidence_score > 100 {
            return Err(ManifestError::InvalidField(
                "min_confidence_score must be between 0 and 100".to_string(),
            ));
        }
        Ok(())
    }
}

/// 5ms-ticker ideation loop settings (LAGI-Edge-1B spark/critic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdeationLoopSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_interval_ms")]
    pub interval_ms: u64,
    #[serde(default = "default_idle_trigger_seconds")]
    pub idle_trigger_seconds: u64,
    #[serde(default = "default_spark_temp")]
    pub temperature_spark: f32,
    #[serde(default = "default_critic_temp")]
    pub temperature_critic: f32,
}

fn default_interval_ms() -> u64 {
    5
}
fn default_idle_trigger_seconds() -> u64 {
    10
}

impl Default for IdeationLoopSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_ms: 5,
            idle_trigger_seconds: 10,
            temperature_spark: 0.85,
            temperature_critic: 0.20,
        }
    }
}

impl IdeationLoopSettings {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.interval_ms == 0 || self.interval_ms > 60_000 {
            return Err(ManifestError::InvalidField(
                "ideation_loop.interval_ms must be between 1 and 60000".to_string(),
            ));
        }
        if self.temperature_spark < 0.0 || self.temperature_spark > 2.0 {
            return Err(ManifestError::InvalidField(
                "ideation_loop.temperature_spark must be between 0.0 and 2.0".to_string(),
            ));
        }
        if self.temperature_critic < 0.0 || self.temperature_critic > 1.0 {
            return Err(ManifestError::InvalidField(
                "ideation_loop.temperature_critic must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Local persistent memory settings (SQLite + vectors)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySettings {
    #[serde(default = "default_db_name")]
    pub db_name: String,
    #[serde(default = "default_vector_dimensions")]
    pub vector_dimensions: u32,
}

fn default_db_name() -> String {
    "lagi_memory.db".to_string()
}
fn default_vector_dimensions() -> u32 {
    384
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            db_name: "lagi_memory.db".to_string(),
            vector_dimensions: 384,
        }
    }
}

impl MemorySettings {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.db_name.is_empty() {
            return Err(ManifestError::MissingField(
                "memory_config.db_name".to_string(),
            ));
        }
        if self.vector_dimensions == 0 || self.vector_dimensions > 4096 {
            return Err(ManifestError::InvalidField(
                "memory_config.vector_dimensions must be between 1 and 4096".to_string(),
            ));
        }
        Ok(())
    }
}

/// Tool definition for agent capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolDefinition {
    pub name: String,
    pub module_path: String,
    pub description: String,
}

impl ToolDefinition {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.name.is_empty() {
            return Err(ManifestError::MissingField("tool.name".to_string()));
        }
        if self.module_path.is_empty() {
            return Err(ManifestError::MissingField("tool.module_path".to_string()));
        }
        if !self.name.chars().all(|c| c.is_alphanumeric() || c == '_')
            || self
                .name
                .chars()
                .next()
                .map_or(true, |c| c.is_ascii_digit())
        {
            return Err(ManifestError::InvalidField(
                "tool.name must be a valid identifier".to_string(),
            ));
        }
        Ok(())
    }
}

/// Root manifest for .lagi container
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LagiManifest {
    #[serde(default = "default_version")]
    pub lagi_version: String,
    pub name: String,
    #[serde(default = "default_author")]
    pub author: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_privacy")]
    pub privacy_level: String,
    /// Legacy full model manifest section
    #[serde(default)]
    pub model: Option<ModelConfig>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    /// Legacy ideation configuration
    #[serde(default)]
    pub ideation: Option<IdeationLoopConfig>,
    /// Lightweight runtime model configuration (LAGI-Edge-1B)
    #[serde(default)]
    pub model_config: Option<ModelRuntimeConfig>,
    /// 5ms-ticker ideation loop settings (LAGI-Edge-1B)
    #[serde(default)]
    pub ideation_loop: Option<IdeationLoopSettings>,
    /// Local memory graph settings
    #[serde(default)]
    pub memory_config: Option<MemorySettings>,
    #[serde(default)]
    pub env_vars: HashMap<String, serde_json::Value>,
}

fn default_version() -> String {
    "1.0.0".to_string()
}
fn default_author() -> String {
    "Liumi Corporation".to_string()
}
fn default_privacy() -> String {
    "AIR_GAPPED_LOCAL".to_string()
}

impl LagiManifest {
    /// Load manifest from JSON file
    pub fn from_file(path: &std::path::Path) -> Result<Self, ManifestError> {
        let content = std::fs::read_to_string(path)?;
        let manifest: LagiManifest = serde_json::from_str(&content)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Save manifest to JSON file
    pub fn to_file(&self, path: &std::path::Path) -> Result<(), ManifestError> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Serialize to JSON bytes
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        let json = serde_json::to_vec(self)?;
        Ok(json)
    }

    /// Deserialize from JSON bytes
    pub fn from_json_bytes(data: &[u8]) -> Result<Self, ManifestError> {
        let manifest: LagiManifest = serde_json::from_slice(data)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Resolve the GGUF model filename from either manifest schema
    pub fn model_gguf_filename(&self) -> Option<&str> {
        if let Some(mc) = &self.model_config {
            Some(mc.gguf_filename.as_str())
        } else {
            self.model.as_ref().map(|m| m.gguf_file.as_str())
        }
    }

    /// Resolve effective ideation settings (new section takes precedence)
    pub fn ideation_settings(&self) -> IdeationLoopSettings {
        self.ideation_loop.clone().unwrap_or_default()
    }

    /// Resolve effective memory settings
    pub fn memory_settings(&self) -> MemorySettings {
        self.memory_config.clone().unwrap_or_default()
    }

    /// Resolve effective runtime model settings
    pub fn runtime_model(&self) -> ModelRuntimeConfig {
        self.model_config.clone().unwrap_or_default()
    }

    /// Validate all fields
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.lagi_version.is_empty() {
            return Err(ManifestError::MissingField("lagi_version".to_string()));
        }
        if self.name.is_empty() {
            return Err(ManifestError::MissingField("name".to_string()));
        }
        if self.name.len() > 128 {
            return Err(ManifestError::InvalidField(
                "name must be <= 128 characters".to_string(),
            ));
        }
        if self.author.len() > 128 {
            return Err(ManifestError::InvalidField(
                "author must be <= 128 characters".to_string(),
            ));
        }
        if self.description.len() > 1024 {
            return Err(ManifestError::InvalidField(
                "description must be <= 1024 characters".to_string(),
            ));
        }
        if !["AIR_GAPPED_LOCAL", "LOCAL_ONLY", "PRIVATE_CLOUD"]
            .contains(&self.privacy_level.as_str())
        {
            return Err(ManifestError::InvalidField(
                "privacy_level must be one of: AIR_GAPPED_LOCAL, LOCAL_ONLY, PRIVATE_CLOUD"
                    .to_string(),
            ));
        }

        if self.model.is_none() && self.model_config.is_none() {
            return Err(ManifestError::MissingField(
                "model or model_config".to_string(),
            ));
        }

        if let Some(ref model) = self.model {
            model.validate()?;
        }
        if let Some(ref mc) = self.model_config {
            mc.validate()?;
        }

        for tool in &self.tools {
            tool.validate()?;
        }

        if let Some(ref ideation) = self.ideation {
            ideation.validate()?;
        }
        if let Some(ref il) = self.ideation_loop {
            il.validate()?;
        }
        if let Some(ref mem) = self.memory_config {
            mem.validate()?;
        }

        Ok(())
    }
}

/// Manifest errors
#[derive(Error, Debug)]
pub enum ManifestError {
    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid field: {0}")]
    InvalidField(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_manifest() -> LagiManifest {
        LagiManifest {
            lagi_version: "1.0.0".to_string(),
            name: "TestAgent".to_string(),
            author: "Test".to_string(),
            description: "Test agent".to_string(),
            privacy_level: "AIR_GAPPED_LOCAL".to_string(),
            model: Some(ModelConfig {
                series: "Test-Model".to_string(),
                gguf_file: "model.gguf".to_string(),
                quantization: "Q5_K_M".to_string(),
                context_window: 8192,
            }),
            tools: vec![],
            ideation: None,
            model_config: None,
            ideation_loop: None,
            memory_config: None,
            env_vars: HashMap::new(),
        }
    }

    fn edge_manifest() -> LagiManifest {
        LagiManifest {
            lagi_version: "1.0.0".to_string(),
            name: "LAGI-1B-Edge-Core".to_string(),
            author: "Liumi Corporation".to_string(),
            description: String::new(),
            privacy_level: "AIR_GAPPED_LOCAL".to_string(),
            model: None,
            tools: vec![],
            ideation: None,
            model_config: Some(ModelRuntimeConfig::default()),
            ideation_loop: Some(IdeationLoopSettings::default()),
            memory_config: Some(MemorySettings::default()),
            env_vars: HashMap::new(),
        }
    }

    #[test]
    fn test_model_config_validation() {
        let model = ModelConfig {
            series: "Test-Model".to_string(),
            gguf_file: "model.gguf".to_string(),
            quantization: "Q4_K_M".to_string(),
            context_window: 2048,
        };
        assert!(model.validate().is_ok());

        let bad_model = ModelConfig {
            series: String::new(),
            ..model.clone()
        };
        assert!(bad_model.validate().is_err());
    }

    #[test]
    fn test_runtime_model_config_validation() {
        assert!(ModelRuntimeConfig::default().validate().is_ok());
        let bad = ModelRuntimeConfig {
            gguf_filename: "model.bin".to_string(),
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_tool_definition_validation() {
        let tool = ToolDefinition {
            name: "test_tool".to_string(),
            module_path: "tools.test".to_string(),
            description: "A test tool".to_string(),
        };
        assert!(tool.validate().is_ok());

        let bad_tool = ToolDefinition {
            name: "123invalid".to_string(),
            module_path: "tools.test".to_string(),
            description: String::new(),
        };
        assert!(bad_tool.validate().is_err());
    }

    #[test]
    fn test_legacy_manifest_roundtrip() {
        let manifest = legacy_manifest();
        let parsed: LagiManifest =
            serde_json::from_str(&serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
        assert_eq!(parsed.name, "TestAgent");
        assert_eq!(parsed.model.as_ref().unwrap().series, "Test-Model");
        assert_eq!(parsed.model_gguf_filename(), Some("model.gguf"));
    }

    #[test]
    fn test_edge_manifest_roundtrip() {
        let manifest = edge_manifest();
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let parsed: LagiManifest = serde_json::from_str(&json).unwrap();
        assert!(parsed.validate().is_ok());
        assert_eq!(
            parsed.model_gguf_filename(),
            Some("qwen2.5-1.5b-instruct-q4_k_m.gguf")
        );
        assert_eq!(parsed.ideation_settings().interval_ms, 5);
        assert_eq!(parsed.memory_settings().vector_dimensions, 384);
    }

    #[test]
    fn test_spec_manifest_json_shape() {
        // The exact JSON from the LAGI-Edge-1B specification
        let json = r#"{
          "lagi_version": "1.0.0",
          "name": "LAGI-1B-Edge-Core",
          "author": "Liumi Corporation",
          "model_config": {
            "gguf_filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "context_window": 4096,
            "cpu_threads": 4,
            "use_gpu": false
          },
          "ideation_loop": {
            "enabled": true,
            "interval_ms": 5,
            "idle_trigger_seconds": 10,
            "temperature_spark": 0.85,
            "temperature_critic": 0.2
          },
          "memory_config": {
            "db_name": "lagi_memory.db",
            "vector_dimensions": 384
          }
        }"#;
        let parsed = LagiManifest::from_json_bytes(json.as_bytes()).unwrap();
        assert_eq!(parsed.name, "LAGI-1B-Edge-Core");
        assert!(parsed.model.is_none());
        assert!(parsed.model_config.is_some());
        assert_eq!(parsed.runtime_model().cpu_threads, 4);
        assert!(!parsed.runtime_model().use_gpu);
    }
}
