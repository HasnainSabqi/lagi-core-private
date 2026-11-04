/*!
LAGI Core Engine - Liumi Autonomous General Intelligence

A production-grade binary container, encryption envelope, and runtime engine
for deploying Living, On-Device Autonomous General Intelligence.
*/

// Core modules
pub mod chat;
pub mod codex;
pub mod crypto;
pub mod heartbeat;
pub mod ideation;
pub mod manifest;
pub mod memory;
pub mod migration;
pub mod model;
pub mod notifications;
pub mod packager;
pub mod profiler;
pub mod runtime;
pub mod scheduler;
#[cfg(feature = "serve")]
pub mod serve;
pub mod spec;
pub mod synaptic_recall;
pub mod tools;
pub mod unpackager;

// Re-exports
pub use codex::{
    build_bias_table, build_chatml, guard_output, CodexBias, FORBIDDEN_LOGIT_BIAS,
    FORBIDDEN_STRINGS, IDENTITY_CORRECTION, IM_END, IM_END_TOKEN_ID, IM_START, IM_START_TOKEN_ID,
    LAGI_PERSONA,
};
pub use crypto::{decrypt_from_container, encrypt_for_container, CryptoError, CryptoManager};
pub use crypto::{derive_distribution_key, derive_hardware_key, derive_transit_key};
pub use heartbeat::{
    start_5ms_heartbeat, Heartbeat, HeartbeatConfig, HeartbeatEvent, HeartbeatState,
};
pub use ideation::IdeationEngine;
pub use ideation::{ThoughtFrame, IDEATION_SYSTEM_PROMPT};
pub use manifest::{
    IdeationLoopConfig, IdeationLoopSettings, LagiManifest, MemorySettings,
    ModelConfig as ManifestModelConfig, ModelRuntimeConfig, ToolDefinition,
};
pub use memory::{
    LedgerTask, MemoryEntry, MemoryError, MemoryGraph, MemoryStats, MemoryType, OnboardingProfile,
    TaskLedger, TaskPriority, TaskStatus, ONBOARDING_QUESTIONS,
};
pub use migration::{export_workspace, import_workspace, MigrationError};
pub use model::LagiModelRunner;
pub use notifications::{show_desktop_notification, IDEATION_NOTIFICATION_THRESHOLD};
pub use packager::{
    find_state_block, flush_memory_to_container, load_memory_from_container, pack_agent,
    read_state, write_state, LagiPackager, PackagerError, RawStateBlock, STATE_MAGIC,
};
pub use profiler::{
    auto_detect_ram_tier, auto_detect_smart_threads, hardware_badge, resolve_hardware_settings,
    HardwareProfile, HardwareSettings,
};
pub use scheduler::{ScheduledTask, TaskScheduler, TaskType};
pub use spec::{
    pack_header, unpack_header, verify_magic, EncryptionMode, Flags, FORMAT_VERSION, HEADER_FORMAT,
    HEADER_SIZE, MAGIC_BYTES,
};
pub use synaptic_recall::SynapticMemoryManager;
pub use tools::{ToolDispatcher, ToolOutput};
pub use unpackager::{
    unpack_agent, Asset, ContainerInfo, LagiUnpackager, UnpackagerError, UnpackedContainer,
};

/// LAGI version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
