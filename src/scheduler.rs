/*!
Proactive Timer & Cron Scheduler

Thread-safe asynchronous task scheduler that lets LAGI perform proactive
time-delayed actions (e.g. "say hello to me after 1 minute") without ever
blocking the chat prompt.

Timers are registered with `TaskScheduler::schedule` (returns instantly) and
driven by the background Tokio heartbeat: a ticker task calls
`check_and_fire()` once per second. When a task expires it fires a native OS
desktop notification and prints the reminder to the console, re-painting the
`lagi>` prompt so the REPL stays usable.
*/

use crate::notifications::show_desktop_notification;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The kind of scheduled task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskType {
    /// A plain user-facing reminder (notification + console message).
    Reminder,
    /// A deferred tool invocation: carries the clean natural-language
    /// command to dispatch through `ToolDispatcher` when the timer expires
    /// (e.g. "create a folder on desktop and name it exp9").
    Tool(String),
}

/// A single scheduled, time-delayed task.
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    /// Monotonic unique id
    pub id: u64,
    /// Absolute moment the task fires
    pub trigger_at: Instant,
    /// Message delivered when the task fires
    pub message: String,
    /// Reminder or Tool
    pub task_type: TaskType,
}

/// Thread-safe scheduler shared between the chat thread and the heartbeat.
#[derive(Clone, Default)]
pub struct TaskScheduler {
    tasks: Arc<Mutex<Vec<ScheduledTask>>>,
    next_id: Arc<AtomicU64>,
    /// Optional memory handle so expired `TaskType::Tool` tasks can build a
    /// `ToolDispatcher` at fire time. Wired once by the host REPL.
    tool_memory: Arc<Mutex<Option<std::sync::Arc<crate::memory::MemoryGraph>>>>,
}

impl TaskScheduler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Give the scheduler access to VectorCore memory so deferred tool
    /// actions can be dispatched when their timer expires.
    pub fn set_tool_memory(&self, memory: std::sync::Arc<crate::memory::MemoryGraph>) {
        if let Ok(mut slot) = self.tool_memory.lock() {
            *slot = Some(memory);
        }
    }

    /// Register a reminder that fires `duration` from now. Returns the task id.
    /// Never blocks for longer than the mutex critical section (~microseconds),
    /// so the chat prompt stays fully responsive.
    pub fn schedule(&self, duration: Duration, message: impl Into<String>) -> u64 {
        self.schedule_task(duration, message.into(), TaskType::Reminder)
    }

    /// Register a task of any type that fires `duration` from now.
    pub fn schedule_task(
        &self,
        duration: Duration,
        message: String,
        task_type: TaskType,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let task = ScheduledTask {
            id,
            trigger_at: Instant::now() + duration,
            message,
            task_type,
        };
        if let Ok(mut tasks) = self.tasks.lock() {
            tasks.push(task);
        }
        id
    }

    /// Fire every expired task: show a native OS desktop notification and
    /// print the reminder to the console. Called from the heartbeat ticker.
    /// Returns the ids of the tasks that fired.
    pub fn check_and_fire(&self) -> Vec<u64> {
        let now = Instant::now();
        let mut fired = Vec::new();
        let expired = match self.tasks.lock() {
            Ok(mut tasks) => {
                let (due, pending): (Vec<ScheduledTask>, Vec<ScheduledTask>) =
                    tasks.drain(..).partition(|t| t.trigger_at <= now);
                *tasks = pending;
                due
            }
            Err(_) => return fired,
        };
        for task in expired {
            fired.push(task.id);
            match &task.task_type {
                TaskType::Reminder => {
                    show_desktop_notification("⏰ LAGI Reminder", &task.message);
                    println!("\n⏰ [LAGI Timer]: {}\nlagi> ", task.message);
                }
                TaskType::Tool(command) => {
                    let result = self.execute_deferred_tool(command);
                    show_desktop_notification("⏰ LAGI Action", &result.text);
                    println!("\n⏰ [LAGI Action]: {}\nlagi> ", result.text);
                }
            }
            use std::io::Write;
            let _ = std::io::stdout().flush();
            tracing::debug!("scheduled task {} fired: {}", task.id, task.message);
        }
        fired
    }

    /// Execute a deferred `TaskType::Tool` action: dispatch the stored clean
    /// command through `ToolDispatcher` so the physical action actually runs
    /// (creates a folder, writes a file, opens the browser, ...). Works both
    /// from the Tokio heartbeat and from the plain-OS-thread fallback.
    fn execute_deferred_tool(
        &self,
        command: &str,
    ) -> crate::tools::ToolOutput {
        use crate::tools::{ToolDispatcher, ToolOutput};

        let memory = self
            .tool_memory
            .lock()
            .ok()
            .and_then(|slot| slot.clone());
        let Some(memory) = memory else {
            return ToolOutput {
                tool: "delayed_tool",
                text: format!("Scheduled action skipped (no memory wired): {}", command),
            };
        };

        let scheduler: Arc<TaskScheduler> = Arc::new(self.clone());
        let dispatcher = ToolDispatcher::with_scheduler(memory, scheduler);
        let command = command.to_string();

        // Run the dispatch on a dedicated OS thread with its own short-lived
        // Tokio runtime. This is safe from every caller: the Tokio heartbeat
        // (block_in_place cannot be used on current-thread runtimes), the
        // plain-OS-thread fallback ticker, and synchronous tests alike.
        let cmd_for_fallback = command.clone();
        let fallback = move || ToolOutput {
            tool: "delayed_tool",
            text: format!("Failed to execute scheduled action: {}", cmd_for_fallback),
        };
        let command_for_match = command.clone();
        let worker = std::thread::Builder::new()
            .name("lagi-deferred-tool".into())
            .spawn(move || match tokio::runtime::Runtime::new() {
                Ok(rt) => rt.block_on(async move { dispatcher.dispatch(&command).await }),
                Err(err) => Some(ToolOutput {
                    tool: "delayed_tool",
                    text: format!("Failed to run scheduled action: {}", err),
                }),
            });
        match worker {
            Ok(handle) => match handle.join() {
                Ok(Some(out)) => out,
                Ok(None) => ToolOutput {
                    tool: "delayed_tool",
                    text: format!(
                        "Scheduled action was not recognized as a tool: {}",
                        command_for_match
                    ),
                },
                Err(_) => fallback(),
            },
            Err(err) => ToolOutput {
                tool: "delayed_tool",
                text: format!("Failed to spawn scheduled action runner: {}", err),
            },
        }
    }

    /// Number of tasks still pending.
    pub fn pending_count(&self) -> usize {
        self.tasks.lock().map(|t| t.len()).unwrap_or(0)
    }

    /// Spawn the background heartbeat ticker that drives the scheduler,
    /// checking for expired tasks once per second. Must be called from within
    /// a Tokio runtime (the chat REPL already runs inside one).
    pub fn spawn_tick(scheduler: Arc<TaskScheduler>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(1));
            loop {
                ticker.tick().await;
                scheduler.check_and_fire();
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_increments_ids() {
        let s = TaskScheduler::new();
        let a = s.schedule(Duration::from_secs(60), "one");
        let b = s.schedule(Duration::from_secs(120), "two");
        assert_ne!(a, b);
        assert_eq!(s.pending_count(), 2);
    }

    #[test]
    fn test_not_fired_early() {
        let s = TaskScheduler::new();
        s.schedule(Duration::from_secs(60), "later");
        assert!(s.check_and_fire().is_empty());
        assert_eq!(s.pending_count(), 1);
    }

    #[test]
    fn test_fires_when_expired() {
        let s = TaskScheduler::new();
        s.schedule_task(Duration::from_millis(0), "now".to_string(), TaskType::Reminder);
        std::thread::sleep(Duration::from_millis(5));
        let fired = s.check_and_fire();
        assert_eq!(fired.len(), 1);
        assert_eq!(s.pending_count(), 0);
        // Second check must not re-fire.
        assert!(s.check_and_fire().is_empty());
    }

    #[test]
    fn test_mixed_task_types_fire_in_order() {
        let s = TaskScheduler::new();
        s.schedule_task(Duration::from_millis(0), "a".to_string(), TaskType::Reminder);
        s.schedule_task(Duration::from_millis(0), "b".to_string(), TaskType::Tool("noop".to_string()));
        std::thread::sleep(Duration::from_millis(5));
        let fired = s.check_and_fire();
        assert_eq!(fired, vec![0, 1]);
    }

    /// End-to-end: an expired `TaskType::Tool` task must physically execute
    /// the action (create the folder on disk) via ToolDispatcher.
    #[test]
    fn test_deferred_tool_actually_executes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let memory = crate::memory::MemoryGraph::new(dir.path().join("mem.lvc"), None)
            .expect("memory");
        let s = TaskScheduler::new();
        s.set_tool_memory(std::sync::Arc::new(memory));

        let folder_name = format!("lagi_delayed_e2e_{}", std::process::id());
        let command = format!("create a folder on desktop and name it {}", folder_name);
        s.schedule_task(
            Duration::from_millis(0),
            command.clone(),
            TaskType::Tool(command),
        );
        std::thread::sleep(Duration::from_millis(5));
        let fired = s.check_and_fire();
        assert_eq!(fired.len(), 1);

        // The physical action must have happened.
        let desktop = dirs_desktop();
        let created = desktop.join(&folder_name);
        assert!(created.exists(), "deferred folder was not created at {:?}", created);
        let _ = std::fs::remove_dir_all(&created);
    }

    fn dirs_desktop() -> std::path::PathBuf {
        // Mirrors PcToolExecutor::resolve_folder("desktop") for the test.
        std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(|home| std::path::PathBuf::from(home).join("Desktop"))
            .unwrap_or_else(std::env::temp_dir)
    }
}
