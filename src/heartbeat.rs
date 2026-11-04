/*!
5ms High-Precision Async Heartbeat Ticker

Implements a non-blocking background heartbeat operating at 5 milliseconds
using Tokio for precise timing and state synchronization.
*/

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

/// Heartbeat configuration
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    /// Interval in milliseconds (default: 5ms)
    pub interval_ms: u64,
    /// Maximum allowed jitter in milliseconds
    pub max_jitter_ms: u64,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval_ms: 5,
            max_jitter_ms: 1,
        }
    }
}

/// Heartbeat state shared across the system
#[derive(Debug, Default, Clone)]
pub struct HeartbeatState {
    /// Total ticks since start
    pub tick_count: u64,
    /// Last tick timestamp (milliseconds since epoch)
    pub last_tick_ms: u64,
    /// Average tick duration in microseconds
    pub avg_tick_duration_us: f64,
    /// Maximum observed tick duration in microseconds
    pub max_tick_duration_us: u64,
}

/// Heartbeat event for subscribers
#[derive(Debug, Clone)]
pub struct HeartbeatEvent {
    pub tick: u64,
    pub timestamp_ms: u64,
    pub duration_us: u64,
}

/// High-precision async heartbeat ticker
pub struct Heartbeat {
    config: HeartbeatConfig,
    state: Arc<RwLock<HeartbeatState>>,
    handle: Option<JoinHandle<()>>,
    event_tx: Option<tokio::sync::broadcast::Sender<HeartbeatEvent>>,
}

impl Heartbeat {
    /// Create a new heartbeat with default configuration
    pub fn new() -> Self {
        Self::with_config(HeartbeatConfig::default())
    }

    /// Create a new heartbeat with custom configuration
    pub fn with_config(config: HeartbeatConfig) -> Self {
        let (tx, _rx) = tokio::sync::broadcast::channel(1024);
        Self {
            config,
            state: Arc::new(RwLock::new(HeartbeatState::default())),
            handle: None,
            event_tx: Some(tx),
        }
    }

    /// Start the heartbeat ticker
    /// The callback is executed on each tick with sub-millisecond precision
    pub fn start<F>(&mut self, mut callback: F)
    where
        F: FnMut(HeartbeatEvent) + Send + 'static,
    {
        let interval = Duration::from_millis(self.config.interval_ms);
        let state = self.state.clone();
        let tx = self.event_tx.clone();

        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            let mut last_tick = std::time::Instant::now();

            loop {
                ticker.tick().await;
                let now = std::time::Instant::now();
                let duration = now.duration_since(last_tick);
                last_tick = now;

                let duration_us = duration.as_micros() as u64;
                let timestamp_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;

                {
                    let mut state_guard = state.write().await;
                    state_guard.tick_count += 1;
                    state_guard.last_tick_ms = timestamp_ms;

                    let alpha = 0.1;
                    state_guard.avg_tick_duration_us = (alpha * duration_us as f64)
                        + ((1.0 - alpha) * state_guard.avg_tick_duration_us);

                    if duration_us > state_guard.max_tick_duration_us {
                        state_guard.max_tick_duration_us = duration_us;
                    }
                }

                let event = HeartbeatEvent {
                    tick: state.read().await.tick_count,
                    timestamp_ms,
                    duration_us,
                };

                if let Some(ref tx) = tx {
                    let _ = tx.send(event.clone());
                }

                callback(event);
            }
        });

        self.handle = Some(handle);
    }

    /// Stop the heartbeat
    pub async fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
            let _ = handle.await;
        }
    }

    /// Get current heartbeat state
    pub async fn get_state(&self) -> HeartbeatState {
        self.state.read().await.clone()
    }

    /// Get the tick count
    pub async fn get_tick_count(&self) -> u64 {
        self.state.read().await.tick_count
    }

    /// Subscribe to heartbeat events
    pub fn subscribe(&self) -> Option<tokio::sync::broadcast::Receiver<HeartbeatEvent>> {
        self.event_tx.as_ref().map(|tx| tx.subscribe())
    }

    /// Check if heartbeat is running
    pub fn is_running(&self) -> bool {
        self.handle.is_some()
    }
}

impl Default for Heartbeat {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

/// Convenience function to start a heartbeat with a callback
pub async fn start_5ms_heartbeat<F>(mut callback: F) -> JoinHandle<()>
where
    F: FnMut() + Send + 'static,
{
    let mut ticker = tokio::time::interval(Duration::from_millis(5));
    tokio::spawn(async move {
        loop {
            ticker.tick().await;
            callback();
        }
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_heartbeat_basic() {
        let mut hb = Heartbeat::with_config(HeartbeatConfig {
            interval_ms: 10,
            max_jitter_ms: 2,
        });

        let count = Arc::new(AtomicU64::new(0));
        let count_clone = count.clone();

        hb.start(move |_event| {
            count_clone.fetch_add(1, Ordering::Relaxed);
        });

        sleep(Duration::from_millis(55)).await;
        hb.stop().await;

        let final_count = count.load(Ordering::Relaxed);
        assert!(final_count >= 4 && final_count <= 7);
    }

    #[tokio::test]
    async fn test_heartbeat_state() {
        let mut hb = Heartbeat::with_config(HeartbeatConfig {
            interval_ms: 10,
            max_jitter_ms: 2,
        });

        hb.start(|_| {});
        sleep(Duration::from_millis(35)).await;
        hb.stop().await;

        let state = hb.get_state().await;
        assert!(state.tick_count > 0);
        assert!(state.avg_tick_duration_us > 0.0);
    }

    #[tokio::test]
    async fn test_heartbeat_events() {
        let mut hb = Heartbeat::with_config(HeartbeatConfig {
            interval_ms: 10,
            max_jitter_ms: 2,
        });

        let mut rx = hb.subscribe().unwrap();
        hb.start(|_| {});

        let mut count = 0;
        while count < 3 {
            if rx.recv().await.is_ok() {
                count += 1;
            }
        }

        hb.stop().await;
        assert_eq!(count, 3);
    }
}
