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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// True while the foreground engine is actively emitting assistant tokens.
///
/// The scheduler always delivers the OS desktop notification immediately. When
/// this flag is set, only the terminal copy is queued so it cannot corrupt a
/// streamed word or answer.
pub static IS_STREAMING: AtomicBool = AtomicBool::new(false);

/// True while the foreground REPL is reading a user input line.
///
/// Typing and streaming are separate states: a scheduler alert must never
/// repaint a partially entered command, but it may be delivered as soon as
/// that line has been submitted.
pub static IS_TYPING: AtomicBool = AtomicBool::new(false);

/// Serializes tests that exercise the process-wide streaming coordinator.
#[cfg(test)]
fn test_output_lock() -> &'static Mutex<()> {
    static TEST_OUTPUT_LOCK: Mutex<()> = Mutex::new(());
    &TEST_OUTPUT_LOCK
}

/// Serializes foreground stream transitions with background terminal writes.
static OUTPUT_LOCK: Mutex<()> = Mutex::new(());

/// Terminal messages deferred because the foreground engine was streaming.
static PENDING_NOTIFICATIONS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn output_lock() -> MutexGuard<'static, ()> {
    OUTPUT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn pending_notifications() -> MutexGuard<'static, Vec<String>> {
    PENDING_NOTIFICATIONS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Mark the foreground engine as streaming.
///
/// This is idempotent and safe to call from nested generation paths.
pub fn begin_streaming() {
    let _guard = output_lock();
    IS_STREAMING.store(true, Ordering::Release);
}

/// Finish foreground generation and return terminal messages that completed
/// while it was active. The atomic flag is cleared before messages are drained.
pub fn finish_streaming() -> Vec<String> {
    let _guard = output_lock();
    IS_STREAMING.store(false, Ordering::Release);
    std::mem::take(&mut *pending_notifications())
}

/// Mark the foreground REPL as reading a user input line.
pub fn begin_typing() {
    let _guard = output_lock();
    IS_TYPING.store(true, Ordering::Release);
}

/// Clear the typing state and return terminal messages queued while the user
/// was entering a command.
pub fn finish_typing() -> Vec<String> {
    let _guard = output_lock();
    IS_TYPING.store(false, Ordering::Release);
    std::mem::take(&mut *pending_notifications())
}

/// Clear the typing state and atomically print any queued completion summary.
pub fn finish_typing_and_print() {
    let _guard = output_lock();
    IS_TYPING.store(false, Ordering::Release);
    let messages = std::mem::take(&mut *pending_notifications());
    if !messages.is_empty() {
        use std::io::Write;
        let mut stdout = std::io::stdout().lock();
        for message in messages {
            let _ = writeln!(
                stdout,
                "📌 [Background Completed While You Were Chatting]: {}",
                message
            );
        }
        let _ = stdout.flush();
    }
}

/// RAII typing marker that resets `IS_TYPING` when input is submitted.
pub struct TypingGuard;

impl TypingGuard {
    pub fn new() -> Self {
        begin_typing();
        Self
    }
}

impl Drop for TypingGuard {
    fn drop(&mut self) {
        let _guard = output_lock();
        IS_TYPING.store(false, Ordering::Release);
    }
}

/// RAII streaming marker. It guarantees that `IS_STREAMING` is reset even if
/// inference returns an error or unwinds.
pub struct StreamingGuard;

impl StreamingGuard {
    pub fn new() -> Self {
        begin_streaming();
        Self
    }
}

impl Drop for StreamingGuard {
    fn drop(&mut self) {
        // Keep the streaming-state transition, queue drain, and summary write
        // under the same lock. A scheduler tick cannot interleave between the
        // flag being cleared and the completion summary being printed.
        let _guard = output_lock();
        IS_STREAMING.store(false, Ordering::Release);
        let messages = std::mem::take(&mut *pending_notifications());
        if !messages.is_empty() {
            use std::io::Write;
            let mut stdout = std::io::stdout().lock();
            for message in messages {
                let _ = writeln!(
                    stdout,
                    "📌 [Background Completed While You Were Chatting]: {}",
                    message
                );
            }
            let _ = stdout.flush();
        }
    }
}

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
    pub fn schedule_task(&self, duration: Duration, message: String, task_type: TaskType) -> u64 {
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
            let (title, message) = match &task.task_type {
                TaskType::Reminder => {
                    show_desktop_notification("⏰ LAGI Reminder", &task.message);
                    ("⏰ [LAGI Timer]", task.message.clone())
                }
                TaskType::Tool(command) => {
                    let result = self.execute_deferred_tool(command);
                    show_desktop_notification("⏰ LAGI Action", &result.text);
                    ("⏰ [LAGI Action]", result.text)
                }
            };

            // Serialize the streaming-state check with the terminal write.
            // The desktop notification and deferred tool already ran; only the
            // terminal copy is delayed when the foreground is streaming.
            {
                let _guard = output_lock();
                if IS_STREAMING.load(Ordering::Acquire) || IS_TYPING.load(Ordering::Acquire) {
                    pending_notifications().push(message.clone());
                } else {
                    use std::io::Write;
                    let mut stdout = std::io::stdout().lock();
                    let _ = write!(stdout, "\r\x1b[2K\n{}: {}\nlagi> ", title, message);
                    let _ = stdout.flush();
                }
            }
            tracing::debug!("scheduled task {} fired: {}", task.id, task.message);
        }
        fired
    }

    /// Execute a deferred `TaskType::Tool` action: dispatch the stored clean
    /// command through `ToolDispatcher` so the physical action actually runs
    /// (creates a folder, writes a file, opens the browser, ...). Works both
    /// from the Tokio heartbeat and from the plain-OS-thread fallback.
    fn execute_deferred_tool(&self, command: &str) -> crate::tools::ToolOutput {
        use crate::tools::{ToolDispatcher, ToolOutput};

        let memory = self.tool_memory.lock().ok().and_then(|slot| slot.clone());
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
    fn test_expired_message_is_queued_while_streaming() {
        let _test_guard = crate::scheduler::test_output_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let s = TaskScheduler::new();
        s.schedule_task(
            Duration::from_millis(0),
            "queued while streaming".to_string(),
            TaskType::Reminder,
        );
        std::thread::sleep(Duration::from_millis(5));
        begin_streaming();
        assert_eq!(s.check_and_fire().len(), 1);
        assert!(IS_STREAMING.load(Ordering::Acquire));
        let messages = finish_streaming();
        assert_eq!(messages, vec!["queued while streaming"]);
        assert!(!IS_STREAMING.load(Ordering::Acquire));
        assert!(finish_streaming().is_empty());
    }

    #[test]
    fn test_expired_message_is_queued_while_typing() {
        let _test_guard = crate::scheduler::test_output_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let s = TaskScheduler::new();
        s.schedule_task(
            Duration::from_millis(0),
            "queued while typing".to_string(),
            TaskType::Reminder,
        );
        std::thread::sleep(Duration::from_millis(5));
        begin_typing();
        assert_eq!(s.check_and_fire().len(), 1);
        assert!(IS_TYPING.load(Ordering::Acquire));
        assert_eq!(finish_typing(), vec!["queued while typing"]);
        assert!(!IS_TYPING.load(Ordering::Acquire));
    }

    #[test]
    fn test_multiple_expired_messages_are_queued_in_order() {
        let _test_guard = crate::scheduler::test_output_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let s = TaskScheduler::new();
        s.schedule_task(
            Duration::from_millis(0),
            "first".to_string(),
            TaskType::Reminder,
        );
        s.schedule_task(
            Duration::from_millis(0),
            "second".to_string(),
            TaskType::Reminder,
        );
        std::thread::sleep(Duration::from_millis(5));
        begin_streaming();
        assert_eq!(s.check_and_fire().len(), 2);
        assert_eq!(finish_streaming(), vec!["first", "second"]);
    }

    #[test]
    fn test_fires_when_expired() {
        let _test_guard = crate::scheduler::test_output_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let s = TaskScheduler::new();
        s.schedule_task(
            Duration::from_millis(0),
            "now".to_string(),
            TaskType::Reminder,
        );
        std::thread::sleep(Duration::from_millis(5));
        let fired = s.check_and_fire();
        assert_eq!(fired.len(), 1);
        assert_eq!(s.pending_count(), 0);
        // Second check must not re-fire.
        assert!(s.check_and_fire().is_empty());
    }

    #[test]
    fn test_mixed_task_types_fire_in_order() {
        let _test_guard = crate::scheduler::test_output_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let s = TaskScheduler::new();
        s.schedule_task(
            Duration::from_millis(0),
            "a".to_string(),
            TaskType::Reminder,
        );
        s.schedule_task(
            Duration::from_millis(0),
            "b".to_string(),
            TaskType::Tool("noop".to_string()),
        );
        std::thread::sleep(Duration::from_millis(5));
        let fired = s.check_and_fire();
        assert_eq!(fired, vec![0, 1]);
    }

    /// End-to-end: an expired `TaskType::Tool` task must physically execute
    /// the action (create the folder on disk) via ToolDispatcher.
    #[test]
    fn test_deferred_tool_actually_executes() {
        let _test_guard = crate::scheduler::test_output_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempfile::tempdir().expect("tempdir");
        let memory =
            crate::memory::MemoryGraph::new(dir.path().join("mem.lvc"), None).expect("memory");
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
        assert!(
            created.exists(),
            "deferred folder was not created at {:?}",
            created
        );
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
