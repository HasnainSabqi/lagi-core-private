/*!
Hardware Profiler

Detects the execution capabilities of the host so the Model Delivery
Network can pick the best `.lagi` payload (AVX2/AVX512/NEON, RAM tier,
optional GPU/VRAM).
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Snapshot of host hardware relevant to payload selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub arch: String,
    pub cpu_brand: String,
    pub cpu_cores: u32,
    pub total_ram_mb: u64,
    pub free_ram_mb: u64,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512_vnni: bool,
    pub neon: bool,
    pub is_apple_silicon: bool,
    pub gpu: Option<GpuInfo>,
    /// Suggested payload tier: "cpu-tiny", "cpu", "cuda", "metal"
    pub recommended_tier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub backend: String, // "cuda" | "metal" | "vulkan" | "none"
    pub vram_mb: Option<u64>,
}

impl HardwareProfile {
    /// Probe the current machine
    pub fn detect() -> Self {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_all();

        let arch = std::env::consts::ARCH.to_string();
        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_default();
        let cpu_cores = sys.cpus().len() as u32;
        let total_ram_mb = sys.total_memory() / (1024 * 1024);
        let free_ram_mb = sys.available_memory() / (1024 * 1024);

        let avx2 = is_x86_feature_detected_safe("avx2");
        let avx512f = is_x86_feature_detected_safe("avx512f");
        let avx512_vnni = is_x86_feature_detected_safe("avx512vnni");
        let neon = detect_neon();

        let is_apple_silicon = cfg!(all(target_os = "macos", target_arch = "aarch64"));

        // GPU probing — llama.cpp ships its own; we only care for tiering
        let gpu = detect_gpu();

        let recommended_tier = recommend_tier(avx2, avx512_vnni, total_ram_mb, is_apple_silicon, &gpu);

        Self {
            arch,
            cpu_brand,
            cpu_cores,
            total_ram_mb,
            free_ram_mb,
            avx2,
            avx512f,
            avx512_vnni,
            neon,
            is_apple_silicon,
            gpu,
            recommended_tier,
        }
    }

    /// Render as a flat string map for Python FFI
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("arch".into(), self.arch.clone());
        m.insert("cpu_brand".into(), self.cpu_brand.clone());
        m.insert("cpu_cores".into(), self.cpu_cores.to_string());
        m.insert("total_ram_mb".into(), self.total_ram_mb.to_string());
        m.insert("free_ram_mb".into(), self.free_ram_mb.to_string());
        m.insert("avx2".into(), self.avx2.to_string());
        m.insert("avx512f".into(), self.avx512f.to_string());
        m.insert("avx512_vnni".into(), self.avx512_vnni.to_string());
        m.insert("neon".into(), self.neon.to_string());
        m.insert("is_apple_silicon".into(), self.is_apple_silicon.to_string());
        m.insert("recommended_tier".into(), self.recommended_tier.clone());
        if let Some(gpu) = &self.gpu {
            m.insert("gpu_backend".into(), gpu.backend.clone());
            if let Some(vram) = gpu.vram_mb {
                m.insert("gpu_vram_mb".into(), vram.to_string());
            }
        }
        m
    }
}

#[inline]
fn is_x86_feature_detected_safe(f: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        match f {
            "avx2" => std::is_x86_feature_detected!("avx2"),
            "avx512f" => std::is_x86_feature_detected!("avx512f"),
            "avx512vnni" => std::is_x86_feature_detected!("avx512vnni"),
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = f;
        false
    }
}

#[inline]
fn detect_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true // NEON is baseline on aarch64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn detect_gpu() -> Option<GpuInfo> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Some(GpuInfo {
            backend: "metal".into(),
            vram_mb: None,
        });
    }
    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
        return Some(GpuInfo {
            backend: "cuda".into(),
            vram_mb: None,
        });
    }
    None
}

fn recommend_tier(
    avx2: bool,
    avx512_vnni: bool,
    total_ram_mb: u64,
    is_apple_silicon: bool,
    gpu: &Option<GpuInfo>,
) -> String {
    if is_apple_silicon || gpu.as_ref().map_or(false, |g| g.backend == "metal") {
        return "metal".into();
    }
    if gpu.as_ref().map_or(false, |g| g.backend == "cuda") && total_ram_mb >= 4096 {
        return "cuda".into();
    }
    if total_ram_mb < 2048 {
        return "cpu-tiny".into();
    }
    if avx2 || avx512_vnni {
        "cpu".into()
    } else {
        "cpu-tiny".into()
    }
}

// ---------------------------------------------------------------------------
// Hardware Auto-Sensing: RAM tiers, smart thread allocation, settings badge
// ---------------------------------------------------------------------------

/// Auto-detected execution settings for the local engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSettings {
    /// RAM tier name ("Compact 2G", "Balanced 4G", "Performance 6G")
    pub tier_name: String,
    /// Context window in tokens (8k / 16k / 32k)
    pub context_window: usize,
    /// Prompt-processing batch size (512 / 1024 / 2048)
    pub n_batch: usize,
    /// Smart thread allocation (physical cores minus OS reserve)
    pub threads: u32,
    /// Total physical RAM in GB
    pub total_ram_gb: f64,
}

/// Classify a host into a RAM tier purely from memory sizes.
///
/// Returns (tier_name, context_window, n_batch):
/// - total < 12 GB  -> Compact 2G     (8192 ctx,  512 batch)
/// - 12..24 GB      -> Balanced 4G    (16384 ctx, 1024 batch)
/// - > 24 GB        -> Performance 6G (32768 ctx, 2048 batch)
///
/// If *available* RAM is under 3 GB the tier is downscaled with a hard
/// 4096/256 clamp so the host OS never gets squeezed into swap.
fn ram_tier_for(total_ram_mb: u64, available_ram_mb: u64) -> (&'static str, usize, usize) {
    let (name, ctx, batch) = if total_ram_mb < 12 * 1024 {
        ("Compact 2G", 8192, 512)
    } else if total_ram_mb <= 24 * 1024 {
        ("Balanced 4G", 16384, 1024)
    } else {
        ("Performance 6G", 32768, 2048)
    };

    if available_ram_mb < 3 * 1024 {
        ("Compact 2G (downscaled)", 4096, 256)
    } else {
        (name, ctx, batch)
    }
}

/// Inspect the host via sysinfo and return the auto-detected
/// (context_window, n_batch) pair. Zero flags required.
pub fn auto_detect_ram_tier() -> (usize, usize) {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let (name, ctx, batch) =
        ram_tier_for(sys.total_memory() / (1024 * 1024), sys.available_memory() / (1024 * 1024));
    tracing::debug!("auto-detected RAM tier: {} ({} ctx, {} batch)", name, ctx, batch);
    (ctx, batch)
}

/// Smart thread allocation: use physical cores (not hyperthreads), leave
/// 1-2 cores free for the host OS, and clamp to [2, 12].
pub fn auto_detect_smart_threads() -> u32 {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu();
    let physical = sys.physical_core_count().map(|c| c as u32);
    let logical = sys.cpus().len() as u32;

    smart_threads_for(physical, logical)
}

/// Pure thread-count math for testability.
/// `physical` may be None on exotic platforms; then we estimate from logical.
fn smart_threads_for(physical: Option<u32>, logical: u32) -> u32 {
    let base = match physical {
        Some(p) if p > 0 => p,
        _ => (logical / 2).max(1), // SMT estimate: half the logical cores
    };
    // Leave 1 core free on small hosts, 2 on bigger ones (OS responsiveness).
    let usable = if base <= 3 {
        base.saturating_sub(1)
    } else {
        base.saturating_sub(2)
    };
    usable.clamp(2, 12)
}

/// Full auto-sense resolution honoring optional CLI overrides.
///
/// `ram_arg`: "auto" | "2g" | "4g" | "6g"
/// `threads_arg`: "auto" | decimal number
pub fn resolve_hardware_settings(ram_arg: &str, threads_arg: &str) -> HardwareSettings {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let available_ram_mb = sys.available_memory() / (1024 * 1024);
    let total_ram_gb = total_ram_mb as f64 / 1024.0;

    let (tier_name, context_window, n_batch) = match ram_arg.to_lowercase().as_str() {
        "2g" => ("Compact 2G (forced)", 8192, 512),
        "4g" => ("Balanced 4G (forced)", 16384, 1024),
        "6g" => ("Performance 6G (forced)", 32768, 2048),
        _ => ram_tier_for(total_ram_mb, available_ram_mb),
    };

    let threads = match threads_arg.trim().to_lowercase().as_str() {
        "auto" | "" => auto_detect_smart_threads(),
        n => n
            .parse::<u32>()
            .map(|t| t.clamp(2, 12))
            .unwrap_or_else(|_| auto_detect_smart_threads()),
    };

    HardwareSettings {
        tier_name: tier_name.to_string(),
        context_window,
        n_batch,
        threads,
        total_ram_gb,
    }
}

/// The startup badge line, e.g.
/// `[Auto-Sense: 16 GB RAM detected -> Balanced 4G (16K ctx, 1024 batch) | Smart Threads: 7 cores]`
pub fn hardware_badge(settings: &HardwareSettings) -> String {
    format!(
        "[Auto-Sense: {:.0} GB RAM detected -> {} ({}K ctx, {} batch) | Smart Threads: {} cores]",
        settings.total_ram_gb,
        settings.tier_name,
        settings.context_window / 1024,
        settings.n_batch,
        settings.threads
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_detects_something() {
        let p = HardwareProfile::detect();
        assert!(!p.arch.is_empty());
        assert!(p.cpu_cores > 0);
        assert!(p.total_ram_mb > 0);
        let m = p.to_map();
        assert!(m.contains_key("recommended_tier"));
    }

    #[test]
    fn test_ram_tier_thresholds() {
        assert_eq!(ram_tier_for(8 * 1024, 6 * 1024), ("Compact 2G", 8192, 512));
        assert_eq!(ram_tier_for(16 * 1024, 10 * 1024), ("Balanced 4G", 16384, 1024));
        assert_eq!(ram_tier_for(24 * 1024, 12 * 1024), ("Balanced 4G", 16384, 1024));
        assert_eq!(ram_tier_for(32 * 1024, 16 * 1024), ("Performance 6G", 32768, 2048));
    }

    #[test]
    fn test_ram_tier_downscale_on_low_available() {
        let (name, ctx, batch) = ram_tier_for(32 * 1024, 2 * 1024);
        assert!(name.contains("downscaled"));
        assert_eq!((ctx, batch), (4096, 256));
        // Boundary: exactly 3 GB available does NOT downscale.
        assert_eq!(ram_tier_for(16 * 1024, 3 * 1024), ("Balanced 4G", 16384, 1024));
    }

    #[test]
    fn test_smart_threads_small_host_keeps_cores_free() {
        assert_eq!(smart_threads_for(Some(2), 4), 2); // 1 core free
        assert_eq!(smart_threads_for(Some(4), 8), 2); // 2 cores free
        assert_eq!(smart_threads_for(Some(8), 16), 6);
        assert_eq!(smart_threads_for(Some(16), 32), 12); // clamped
        assert_eq!(smart_threads_for(Some(1), 2), 2); // clamped up
    }

    #[test]
    fn test_smart_threads_fallback_when_physical_unknown() {
        assert_eq!(smart_threads_for(None, 16), 6); // est. 8 phys -> 6
        assert_eq!(smart_threads_for(None, 4), 2); // est. 2 phys -> clamp 2
    }

    #[test]
    fn test_auto_detect_returns_sane_values() {
        let (ctx, batch) = auto_detect_ram_tier();
        assert!(matches!(
            (ctx, batch),
            (8192, 512) | (16384, 1024) | (32768, 2048) | (4096, 256)
        ));
        let t = auto_detect_smart_threads();
        assert!((2..=12).contains(&t));
    }

    #[test]
    fn test_resolve_hardware_settings_overrides_and_badge() {
        let auto = resolve_hardware_settings("auto", "auto");
        assert!(auto.context_window >= 4096);
        assert!(auto.threads >= 2);

        let forced = resolve_hardware_settings("6g", "5");
        assert_eq!(
            (forced.context_window, forced.n_batch, forced.threads),
            (32768, 2048, 5)
        );

        // Invalid numeric input falls back to auto threads.
        let fallback = resolve_hardware_settings("4g", "not-a-number");
        assert_eq!((fallback.context_window, fallback.n_batch), (16384, 1024));
        assert!((2..=12).contains(&fallback.threads));

        let badge = hardware_badge(&forced);
        assert!(badge.starts_with("[Auto-Sense:"));
        assert!(badge.contains("Smart Threads: 5 cores"));
    }
}
