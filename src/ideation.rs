/*!
Autonomous Ideation Engine (Spark-and-Critic)

Dual-temperature background loop driven by the 5ms heartbeat ticker:
- Phase 1 (Spark):   temperature 0.85 — creative concept drafting
- Phase 2 (Critic):  temperature 0.20 — logic/safety/utility verification

Validated thoughts are appended to the encrypted local memory graph.
The loop only fires after `idle_trigger_seconds` of user inactivity.
*/

use crate::memory::MemoryGraph;
use crate::model::ThoughtGenerator;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::time::{interval, Duration};

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/**
System prompt framing for the backend state-evaluation model.

CRITICAL DESIGN RULE: the model is NEVER asked to "run a loop", "execute
code", or perform any action. Loop execution is owned entirely by the Rust
`tokio::time::interval` ticker in this module. The model's sole job is to act
as a **State Evaluator & Spark-and-Critic Ideation Engine**: given a state
tick snapshot, it emits one structured thought frame in JSON.
*/
pub const IDEATION_SYSTEM_PROMPT: &str = "\
You are a State Evaluator and Spark-and-Critic Ideation Engine embedded in a \
local autonomous assistant. You do NOT run loops, execute code, or perform \
actions — the host runtime owns all execution. On every state tick you receive, \
respond with exactly one JSON object and nothing else:\n\
{\n\
  \"cycle_type\": \"SPARK\" | \"CRITIC\" | \"ACTION\",\n\
  \"thought_draft\": \"string\",\n\
  \"evaluation_score\": 0.95,\n\
  \"next_state\": \"IDLE\" | \"EXECUTE\"\n\
}\n\
- SPARK: propose a novel, creative concept for the user's context.\n\
- CRITIC: rigorously evaluate feasibility, safety, and utility of the prior draft.\n\
- ACTION: recommend one concrete, local-only next step.\n\
Keep thought_draft under 400 characters. Never claim to execute anything.";

/// Structured thought frame emitted by the state-evaluation model
///
/// The heartbeat loop parses the model's JSON output into this frame; any
/// non-JSON response degrades gracefully into a CRITIC frame with a neutral
/// score so the loop never stalls on malformed model output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThoughtFrame {
    /// "SPARK" | "CRITIC" | "ACTION"
    pub cycle_type: String,
    /// The model's drafted thought text
    pub thought_draft: String,
    /// Model self-assessed quality score (0.0 - 1.0)
    pub evaluation_score: f32,
    /// "IDLE" | "EXECUTE" — advisory only; execution stays host-owned
    pub next_state: String,
}

impl ThoughtFrame {
    /// Parse a model response into a frame; falls back to a neutral CRITIC
    /// frame for non-JSON output so the loop never stalls.
    pub fn parse(raw: &str) -> Self {
        if let Ok(f) = serde_json::from_str::<ThoughtFrame>(raw.trim()) {
            let score = f.evaluation_score.clamp(0.0, 1.0);
            return ThoughtFrame {
                cycle_type: f.cycle_type,
                thought_draft: f.thought_draft,
                evaluation_score: score,
                next_state: f.next_state,
            };
        }
        Self {
            cycle_type: "CRITIC".to_string(),
            thought_draft: raw.trim().to_string(),
            evaluation_score: 0.5,
            next_state: "IDLE".to_string(),
        }
    }

    /// Render as compact JSON for host-software consumption
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Event emitted whenever the loop produces and validates a new thought
///
/// Other software (the host app embedding LAGI) subscribes to a
/// [`broadcast::Receiver`] full of these to react to the agent's imagination
/// in real time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThoughtEvent {
    /// Monotonic cycle index of the loop that produced it
    pub cycle: u64,
    /// Spark-phase (creative) draft
    pub spark: String,
    /// Critic-phase (verified) final text — what got persisted
    pub critic: String,
    /// Unix-millis timestamp of persistence
    pub timestamp_ms: u64,
}

/// Dual-temperature autonomous ideation engine
pub struct IdeationEngine {
    spark_temperature: f32,
    critic_temperature: f32,
    interval_ms: u64,
    idle_trigger_seconds: u64,
    cycle_count: Arc<AtomicU64>,
    last_activity_ms: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
    events_tx: broadcast::Sender<ThoughtEvent>,
}

impl IdeationEngine {
    /// Create an engine with the given spark/critic sampling temperatures
    pub fn new(spark_temp: f32, critic_temp: f32) -> Self {
        let (events_tx, _) = broadcast::channel(256);
        Self {
            spark_temperature: spark_temp,
            critic_temperature: critic_temp,
            interval_ms: 5,
            idle_trigger_seconds: 10,
            cycle_count: Arc::new(AtomicU64::new(0)),
            last_activity_ms: Arc::new(AtomicU64::new(now_ms())),
            running: Arc::new(AtomicBool::new(false)),
            events_tx,
        }
    }

    /// Override the ticker period and idle threshold (builder style)
    pub fn with_schedule(mut self, interval_ms: u64, idle_trigger_seconds: u64) -> Self {
        self.interval_ms = interval_ms;
        self.idle_trigger_seconds = idle_trigger_seconds;
        self
    }

    /// Record user/system activity; suppresses ideation for the idle window
    pub fn note_activity(&self) {
        self.last_activity_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// Override the idle timer (e.g. 0 marks the agent as indefinitely idle,
    /// letting CLI runs start ideation immediately)
    pub fn last_activity_override(&self, timestamp_ms: u64) {
        self.last_activity_ms.store(timestamp_ms, Ordering::Relaxed);
    }

    /// Subscribe to validated thoughts as they're produced.
    ///
    /// The returned [`broadcast::Receiver`] is how host software listens to
    /// LAGI's imagination in real time. Lagging receivers get
    /// [`broadcast::error::RecvError::Lagged`], so slow consumers just skip
    /// ahead — the loop never blocks on them.
    pub fn subscribe(&self) -> broadcast::Receiver<ThoughtEvent> {
        self.events_tx.subscribe()
    }

    /// Start the continuous background spark/critic loop
    pub async fn start_background_loop(
        &self,
        model: Arc<dyn ThoughtGenerator>,
        memory: Arc<MemoryGraph>,
    ) {
        if self.running.swap(true, Ordering::SeqCst) {
            return; // already running
        }

        let mut ticker = interval(Duration::from_millis(self.interval_ms));
        let running = self.running.clone();
        let cycle_count = self.cycle_count.clone();
        let last_activity_ms = self.last_activity_ms.clone();
        let spark_temp = self.spark_temperature;
        let critic_temp = self.critic_temperature;
        let idle_trigger_ms = self.idle_trigger_seconds.saturating_mul(1000);
        let events_tx = self.events_tx.clone();

        tokio::spawn(async move {
            loop {
                ticker.tick().await;

                if !running.load(Ordering::Relaxed) {
                    break;
                }

                // Suppress ideation while the user is active
                let idle_for = now_ms().saturating_sub(last_activity_ms.load(Ordering::Relaxed));
                if idle_for < idle_trigger_ms {
                    continue;
                }

                // Step 1: Spark Phase — creative drafting (temperature 0.85).
                // Prompts are state-evaluation framed: the model is never asked
                // to run or execute anything — only to draft and evaluate.
                let spark_prompt = format!(
                    "{}\n\nSTATE TICK {{\"idle_seconds\": {}}}: Draft one novel \
                     user-focused background automation concept as a SPARK frame.",
                    IDEATION_SYSTEM_PROMPT,
                    idle_for / 1000
                );
                if let Ok(spark_raw) = model.generate_thought(&spark_prompt, spark_temp) {
                    let spark_frame = ThoughtFrame::parse(&spark_raw);
                    // Step 2: Critic Phase — logic & safety verification (temperature 0.20)
                    let critic_prompt = format!(
                        "{}\n\nEvaluate feasibility, safety, and utility of this draft \
                         as a CRITIC frame:\n{}",
                        IDEATION_SYSTEM_PROMPT,
                        spark_frame.thought_draft
                    );
                    if let Ok(critic_raw) = model.generate_thought(&critic_prompt, critic_temp) {
                        let critic_frame = ThoughtFrame::parse(&critic_raw);
                        // Step 3: Append validated thought into encrypted vector memory
                        let _ = memory.store_thought(&critic_frame.thought_draft);
                        // Step 4: High-confidence ideation (>= 0.85) surfaces as a
                        // native desktop toast. Silent fallback on any failure.
                        crate::notifications::maybe_notify_ideation(
                            &critic_frame.thought_draft,
                            critic_frame.evaluation_score,
                        );
                        let cycle = cycle_count.fetch_add(1, Ordering::Relaxed) + 1;
                        // Publish to any host-software subscribers
                        let _ = events_tx.send(ThoughtEvent {
                            cycle,
                            spark: spark_frame.to_json(),
                            critic: critic_frame.to_json(),
                            timestamp_ms: now_ms(),
                        });
                    }
                }
            }
        });
    }

    /// Stop the background loop (checked on the next tick)
    pub async fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Whether the background loop is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Number of completed spark/critic cycles
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count.load(Ordering::Relaxed)
    }
}

impl Default for IdeationEngine {
    fn default() -> Self {
        Self::new(0.85, 0.20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryType;
    use crate::model::LagiModelRunner;
    use tempfile::tempdir;

    fn ram_model() -> Arc<dyn ThoughtGenerator> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&[0u8; 64]);
        Arc::new(LagiModelRunner::init_from_ram(&bytes, 4).unwrap())
    }

    #[tokio::test]
    async fn test_engine_defaults() {
        let engine = IdeationEngine::default();
        assert!(!engine.is_running());
        assert_eq!(engine.cycle_count(), 0);
    }

    #[tokio::test]
    async fn test_background_loop_produces_thoughts() {
        let dir = tempdir().unwrap();
        let memory = Arc::new(MemoryGraph::new(dir.path().join("mem.db"), None).unwrap());
        let model = ram_model();

        // Idle trigger of 0s so the loop fires immediately on the 5ms ticker
        let engine = IdeationEngine::new(0.85, 0.20).with_schedule(5, 0);
        engine.start_background_loop(model, memory.clone()).await;

        assert!(engine.is_running());
        tokio::time::sleep(Duration::from_millis(60)).await;
        engine.stop().await;
        tokio::time::sleep(Duration::from_millis(15)).await;

        assert!(engine.cycle_count() > 0, "loop should complete >= 1 cycle");
        let thoughts = memory
            .get_recent(Some(MemoryType::Thought), 10)
            .unwrap();
        assert!(
            !thoughts.is_empty(),
            "validated thoughts should reach memory"
        );
        assert!(!engine.is_running());
    }

    #[tokio::test]
    async fn test_subscribe_receives_thought_event() {
        let dir = tempdir().unwrap();
        let memory = Arc::new(MemoryGraph::new(dir.path().join("mem.db"), None).unwrap());
        let model = ram_model();

        let engine = IdeationEngine::new(0.85, 0.20).with_schedule(5, 0);
        let mut rx = engine.subscribe();

        engine.start_background_loop(model, memory).await;

        let event = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("should receive a thought event within 2s")
            .expect("channel should not close");

        assert_eq!(event.cycle, 1);
        assert!(!event.spark.is_empty());
        assert!(!event.critic.is_empty());
        assert!(event.timestamp_ms > 0);

        engine.stop().await;
    }

    #[tokio::test]
    async fn test_idle_gate_blocks_ideation() {
        let dir = tempdir().unwrap();
        let memory = Arc::new(MemoryGraph::new(dir.path().join("mem.db"), None).unwrap());
        let model = ram_model();

        let engine = IdeationEngine::new(0.85, 0.20).with_schedule(5, 3600);
        engine.last_activity_ms.store(now_ms(), Ordering::Relaxed);
        engine.start_background_loop(model, memory.clone()).await;

        tokio::time::sleep(Duration::from_millis(40)).await;
        engine.stop().await;

        assert_eq!(
            engine.cycle_count(),
            0,
            "idle trigger should suppress ideation during activity"
        );
    }
}

