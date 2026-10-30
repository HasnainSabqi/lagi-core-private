/*!
LAGI Framework Runtime — the integration surface for host software

`LagiRuntime` bundles everything a host app needs in one object:
- decrypts a `.lagi` container into RAM
- starts the 5ms heartbeat + continuous spark/critic ideation
- exposes a live **thought event stream** via Tokio broadcast channels

## Embedding example

```rust,no_run
use lagi_core::runtime::{LagiRuntime, RuntimeEvent};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let runtime = LagiRuntime::load(Path::new("agent.lagi"), None).await?;

    // Subscribe to the imagination stream alongside other handlers
    let mut thoughts = runtime.subscribe_thoughts();
    tokio::spawn(async move {
        while let Ok(thought) = thoughts.recv().await {
            println!("LAGI dreamed: {}", thought.critic);
        }
    });

    runtime.start().await?;

    // Direct Q&A continues alongside the background loop
    let reply = runtime.ask("What was your latest idea?").await?;
    println!("{}", reply);

    runtime.stop().await?;
    Ok(())
}
```
*/

use crate::heartbeat::{Heartbeat, HeartbeatConfig};
use crate::ideation::{IdeationEngine, ThoughtEvent};
use crate::manifest::LagiManifest;
use crate::memory::MemoryGraph;
use crate::model::ThoughtGenerator;
use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::JoinHandle;

use tokio::sync::{broadcast, Mutex};

/// Events a host application can react to
#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    /// 5ms heartbeat tick (with the tick counter)
    Heartbeat(u64),
    /// The ideation engine produced and validated a new thought
    Thought(ThoughtEvent),
    /// Agent started / stopped
    Lifecycle { running: bool, cycles: u64 },
}

/// The full LAGI runtime, ready to embed into host software
pub struct LagiRuntime {
    manifest: LagiManifest,
    memory: Arc<MemoryGraph>,
    engine: Arc<IdeationEngine>,
    /// Model backend driving thought generation (stub or llama.cpp)
    model: Arc<dyn ThoughtGenerator>,
    /// 5ms heartbeat (started/stopped by the runtime)
    heartbeat: Mutex<Heartbeat>,
    /// Combined event channel — heartbeats + thoughts
    events_tx: broadcast::Sender<RuntimeEvent>,
    /// Thought→RuntimeEvent forwarding task handle
    forward_task: Mutex<Option<JoinHandle<()>>>,
    /// dir/file where the SQLite memory db lives (alongside the .lagi)
    memory_path: PathBuf,
    /// temp file holding the llama model (if the real backend is active);
    /// deleted when the runtime is dropped
    _llama_tmp: Option<PathBuf>,
}

impl LagiRuntime {
    /// Decrypt a `.lagi` container into RAM and build the runtime around it
    pub async fn load(path: &std::path::Path, context: Option<&[u8]>) -> Result<Self> {
        let unpackager = crate::unpackager::LagiUnpackager::new(path);
        let container = unpackager
            .unpack(context)
            .map_err(Into::<anyhow::Error>::into)?;

        let manifest = container.manifest.clone();
        let mem_cfg = manifest.memory_settings();

        let memory_path = path
            .parent()
            .map(|p| p.join(&mem_cfg.db_name))
            .unwrap_or_else(|| PathBuf::from(&mem_cfg.db_name));
        let memory = Arc::new(MemoryGraph::new(&memory_path, None)?);

        let model_cfg = manifest.runtime_model();
        let model_bytes = container
            .get_model()
            .ok_or_else(|| anyhow!("container has no model asset"))?
            .to_vec();

        let (model, llama_tmp) = build_model_backend(&model_bytes, &model_cfg)?;

        let loop_cfg = manifest.ideation_settings();
        let engine = Arc::new(
            IdeationEngine::new(loop_cfg.temperature_spark, loop_cfg.temperature_critic)
                .with_schedule(loop_cfg.interval_ms, loop_cfg.idle_trigger_seconds),
        );

        let (events_tx, _) = broadcast::channel(512);
        let heartbeat = Heartbeat::with_config(HeartbeatConfig {
            interval_ms: 5,
            max_jitter_ms: 1,
        });

        Ok(Self {
            manifest,
            memory,
            engine,
            model,
            heartbeat: Mutex::new(heartbeat),
            events_tx,
            forward_task: Mutex::new(None),
            memory_path,
            _llama_tmp: llama_tmp,
        })
    }

    /// Build the runtime from already-decrypted parts (no container load).
    ///
    /// Used by [`LagiRuntime::load`] internally; also usable by tests or host
    /// code that already has the manifest and model bytes in RAM.
    pub async fn from_decrypted(
        manifest: LagiManifest,
        model_bytes: &[u8],
        memory_path: PathBuf,
    ) -> Result<Self> {
        let model_cfg = manifest.runtime_model();
        let (model, llama_tmp) = build_model_backend(model_bytes, &model_cfg)?;

        let memory = Arc::new(MemoryGraph::new(&memory_path, None)?);

        let loop_cfg = manifest.ideation_settings();
        let engine = Arc::new(
            IdeationEngine::new(loop_cfg.temperature_spark, loop_cfg.temperature_critic)
                .with_schedule(loop_cfg.interval_ms, loop_cfg.idle_trigger_seconds),
        );

        let (events_tx, _) = broadcast::channel(512);
        let heartbeat = Heartbeat::with_config(HeartbeatConfig {
            interval_ms: 5,
            max_jitter_ms: 1,
        });

        Ok(Self {
            manifest,
            memory,
            engine,
            model,
            heartbeat: Mutex::new(heartbeat),
            events_tx,
            forward_task: Mutex::new(None),
            memory_path,
            _llama_tmp: llama_tmp,
        })
    }

    /// Subscribe to the combined runtime event stream (heartbeats + thoughts)
    pub fn subscribe(&self) -> broadcast::Receiver<RuntimeEvent> {
        self.events_tx.subscribe()
    }

    /// Narrower view: thoughts only
    pub fn subscribe_thoughts(&self) -> broadcast::Receiver<ThoughtEvent> {
        self.engine.subscribe()
    }

    /// Start the runtime: heartbeat ticker + continuous ideation loop
    pub async fn start(&self) -> Result<()> {
        // 1) ThoughtEvent forwarder — one per `start`, guarded against double-start
        {
            let mut forwarder = self.forward_task.lock().await;
            if forwarder.is_none() {
                let mut engine_rx = self.engine.subscribe();
                let events_tx = self.events_tx.clone();
                *forwarder = Some(tokio::spawn(async move {
                    while let Ok(thought) = engine_rx.recv().await {
                        let _ = events_tx.send(RuntimeEvent::Thought(thought));
                    }
                }));
            }
        }

        // 2) 5ms heartbeat bridged into events
        {
            let mut hb = self.heartbeat.lock().await;
            if !hb.is_running() {
                let tx = self.events_tx.clone();
                hb.start(move |event| {
                    let _ = tx.send(RuntimeEvent::Heartbeat(event.tick));
                });
            }
        }

        // 3) Continuous imagination
        self.engine
            .start_background_loop(self.model.clone(), self.memory.clone())
            .await;

        let _ = self.events_tx.send(RuntimeEvent::Lifecycle {
            running: true,
            cycles: self.engine.cycle_count(),
        });
        Ok(())
    }

    /// Stop ideation, heartbeat, and the event forwarder
    pub async fn stop(&self) -> Result<()> {
        self.engine.stop().await;

        {
            let mut hb = self.heartbeat.lock().await;
            hb.stop().await;
        }

        if let Some(handle) = self.forward_task.lock().await.take() {
            handle.abort();
        }

        let _ = self.events_tx.send(RuntimeEvent::Lifecycle {
            running: false,
            cycles: self.engine.cycle_count(),
        });
        Ok(())
    }


    /// Force the idle gate open — next ideation tick runs immediately.
    ///
    /// Host software usually wants this off (default: think only after
    /// `idle_trigger_seconds` of inactivity). CLI tools may call it once so
    /// the loop starts firing immediately on startup.
    pub fn mark_idle(&self) {
        self.engine.last_activity_override(0);
    }

    /// Direct one-shot Q&A against the same model backing the ideation loop.
    ///
    /// Notes activity on the engine so the background loop backs off while a
    /// direct question is in flight (avoids contending for CPU with the host).
    pub async fn ask(&self, prompt: &str) -> Result<String> {
        self.engine.note_activity();
        let model = self.model.clone();
        let prompt = prompt.to_string();
        tokio::task::spawn_blocking(move || model.generate_thought(&prompt, 0.7))
            .await?
    }

    /// Number of completed spark/critic cycles so far
    pub fn cycle_count(&self) -> u64 {
        self.engine.cycle_count()
    }

    /// Whether the background loop is currently running
    pub fn is_running(&self) -> bool {
        self.engine.is_running()
    }

    /// Agent identity + model settings as declared in the container
    pub fn manifest(&self) -> &LagiManifest {
        &self.manifest
    }

    /// Access the model backend for custom generation (e.g., ideation spark/critic)
    pub fn model(&self) -> Arc<dyn ThoughtGenerator> {
        self.model.clone()
    }

    /// Path of the on-disk SQLite memory graph
    pub fn memory_path(&self) -> &std::path::Path {
        &self.memory_path
    }
} // impl LagiRuntime

fn build_model_backend(
    model_bytes: &[u8],
    model_cfg: &crate::manifest::ModelRuntimeConfig,
) -> Result<(Arc<dyn ThoughtGenerator>, Option<PathBuf>)> {
    #[cfg(feature = "llama")]
    {
        let (engine, path) = crate::chat::llama_engine::engine_from_ram(model_bytes, model_cfg)?;
        return Ok((Arc::new(engine), Some(path)));
    }

    #[cfg(not(feature = "llama"))]
    {
        let runner =
            crate::model::LagiModelRunner::init_from_ram(model_bytes, model_cfg.cpu_threads)?;
        Ok((Arc::new(runner), None))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_runtime_thought_event_stream() {
        let dir = tempfile::tempdir().unwrap();
        let memory_path = dir.path().join("memory.db");

        let manifest = LagiManifest::from_json_bytes(
            br#"{
                "name": "TestAgent",
                "model": {
                    "series": "Test-Model",
                    "gguf_file": "model.gguf",
                    "quantization": "Q4_K_M",
                    "context_window": 2048
                }
            }"#,
        )
        .unwrap();

        // tiny RAM model for tests
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&[0u8; 64]);

        let runtime = LagiRuntime::from_decrypted(manifest.clone(), &bytes, memory_path)
            .await
            .unwrap();

        let mut thoughts = runtime.subscribe_thoughts();
        runtime.start().await.unwrap();
        assert!(runtime.is_running());
        runtime.mark_idle();

        let event = tokio::time::timeout(Duration::from_secs(2), thoughts.recv())
            .await
            .expect("should produce a thought within 2s")
            .expect("channel should not close");

        assert_eq!(event.cycle, 1);
        assert!(!event.critic.is_empty());

        runtime.stop().await.unwrap();
        assert!(!runtime.is_running());
    }

    #[tokio::test]
    async fn test_runtime_combined_event_stream() {
        let dir = tempfile::tempdir().unwrap();
        let memory_path = dir.path().join("memory.db");

        let manifest = LagiManifest::from_json_bytes(
            br#"{
                "name": "TestAgent",
                "model": {
                    "series": "Test-Model",
                    "gguf_file": "model.gguf",
                    "quantization": "Q4_K_M",
                    "context_window": 2048
                }
            }"#,
        )
        .unwrap();

        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&[0u8; 64]);

        let runtime = LagiRuntime::from_decrypted(manifest, &bytes, memory_path)
            .await
            .unwrap();

        let mut events = runtime.subscribe();
        runtime.start().await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        runtime.stop().await.unwrap();

        let mut got = Vec::new();
        while let Ok(e) = events.try_recv() {
            got.push(e);
        }
        assert!(
            got.iter()
                .any(|e| matches!(e, RuntimeEvent::Lifecycle { running: true, .. })),
            "lifecycle start event should be delivered"
        );
    }
}

