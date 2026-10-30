/*!
LAGI Core Engine - Liumi Autonomous General Intelligence

A production-grade binary container, encryption envelope, and runtime engine
for deploying Living, On-Device Autonomous General Intelligence.
*/

// Core modules
pub mod crypto;
pub mod codex;
pub mod packager;
pub mod unpackager;
pub mod ideation;
pub mod manifest;
pub mod spec;
pub mod memory;
pub mod heartbeat;
pub mod model;
pub mod chat;
pub mod runtime;
pub mod profiler;
pub mod synaptic_recall;
pub mod tools;
pub mod notifications;
pub mod scheduler;
#[cfg(feature = "serve")]
pub mod serve;

// Re-exports
pub use crypto::{
    CryptoManager, CryptoError, encrypt_for_container, decrypt_from_container,
};
pub use packager::{
    LagiPackager, PackagerError, pack_agent,
    find_state_block, read_state, write_state, flush_memory_to_container,
    load_memory_from_container, RawStateBlock, STATE_MAGIC,
};
pub use codex::{
    build_chatml, build_bias_table, guard_output, CodexBias,
    FORBIDDEN_LOGIT_BIAS, FORBIDDEN_STRINGS, IDENTITY_CORRECTION, LAGI_PERSONA,
    IM_END, IM_END_TOKEN_ID, IM_START, IM_START_TOKEN_ID,
};
pub use unpackager::{LagiUnpackager, UnpackagerError, unpack_agent, UnpackedContainer, Asset, ContainerInfo};
pub use ideation::IdeationEngine;
pub use manifest::{
    ModelConfig as ManifestModelConfig, ToolDefinition, IdeationLoopConfig, IdeationLoopSettings,
    MemorySettings, ModelRuntimeConfig, LagiManifest,
};
pub use spec::{
    MAGIC_BYTES, FORMAT_VERSION, HEADER_FORMAT, HEADER_SIZE,
    EncryptionMode, Flags, pack_header, unpack_header, verify_magic,
};
pub use memory::{MemoryGraph, MemoryEntry, MemoryType, MemoryError, MemoryStats, TaskLedger, LedgerTask, TaskPriority, TaskStatus, OnboardingProfile, ONBOARDING_QUESTIONS};
pub use ideation::{ThoughtFrame, IDEATION_SYSTEM_PROMPT};
pub use heartbeat::{Heartbeat, HeartbeatConfig, HeartbeatState, HeartbeatEvent, start_5ms_heartbeat};
pub use model::LagiModelRunner;
pub use synaptic_recall::SynapticMemoryManager;
pub use tools::{ToolDispatcher, ToolOutput};
pub use profiler::{auto_detect_ram_tier, auto_detect_smart_threads, hardware_badge, resolve_hardware_settings, HardwareProfile, HardwareSettings};
pub use notifications::{show_desktop_notification, IDEATION_NOTIFICATION_THRESHOLD};
pub use scheduler::{ScheduledTask, TaskScheduler, TaskType};

/// LAGI version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");