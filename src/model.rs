/*!
Low-Overhead Model Execution (LAGI-Edge-1B, CPU-first)

CPU-optimized inference wrapper that ingests decrypted GGUF weights directly
from RAM buffers provided by `unpackager.rs` â€” the plaintext model is never
written to disk.

Inference is currently a deterministic placeholder with the exact public API
of the production runner, so the llama.cpp sampling backend can be dropped in
behind `generate_thought` without touching `ideation.rs` or `main.rs`.
*/

use anyhow::{anyhow, Result};

/// Anything that can turn a prompt into a thought at a given temperature.
///
/// Object-safe so the ideation engine can swap the in-RAM stub, a llama.cpp
/// backend, or (later) a fully remote worker without code changes downstream.
pub trait ThoughtGenerator: Send + Sync {
    fn generate_thought(&self, prompt: &str, temperature: f32) -> Result<String>;
}

/// CPU-first runner for GGUF models held entirely in RAM
pub struct LagiModelRunner {
    /// Whether a prompt context window matches the model's configured window
    pub context_window: usize,
    /// Number of CPU threads allocated to generation
    pub cpu_threads: u32,
    /// Decrypted GGUF weights, resident in RAM for the process lifetime
    model_bytes: Vec<u8>,
}

impl LagiModelRunner {
    /// Initializes model runner directly from unpacked RAM byte pointers.
    ///
    /// # Errors
    /// Returns an error if the RAM buffer is empty (decryption failed) or
    /// does not carry the `GGUF` magic bytes.
    pub fn init_from_ram(model_bytes: &[u8], threads: u32) -> Result<Self> {
        if model_bytes.is_empty() {
            return Err(anyhow!("Model buffer in RAM is empty. Decryption failed."));
        }

        if model_bytes.len() >= 4 && &model_bytes[..4] != b"GGUF" {
            return Err(anyhow!("RAM buffer is not a valid GGUF model (bad magic)"));
        }

        Ok(Self {
            context_window: 8192,
            cpu_threads: threads,
            model_bytes: model_bytes.to_vec(),
        })
    }

    /// Size of the resident model weights in bytes
    pub fn resident_bytes(&self) -> usize {
        self.model_bytes.len()
    }

    /// Executes single inference pass for continuous ideation or direct prompts.
    ///
    /// Placeholder implementation: the real backend will sample tokens with
    /// the provided temperature from the in-RAM GGUF weights.
    pub fn generate_thought(&self, prompt: &str, temperature: f32) -> Result<String> {
        <Self as ThoughtGenerator>::generate_thought(self, prompt, temperature)
    }
}

impl ThoughtGenerator for LagiModelRunner {
    fn generate_thought(&self, prompt: &str, temperature: f32) -> Result<String> {
        Ok(format!(
            "[LAGI Thought | Temp: {:.2}] Synthesized response for: {}",
            temperature, prompt
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_from_ram_rejects_empty() {
        assert!(LagiModelRunner::init_from_ram(&[], 4).is_err());
    }

    #[test]
    fn test_init_from_ram_rejects_bad_magic() {
        let bad = b"NOT_A_GGUF_FILE_AT_ALL";
        assert!(LagiModelRunner::init_from_ram(bad, 4).is_err());
    }

    #[test]
    fn test_init_from_ram_accepts_gguf_magic() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&[0u8; 32]);
        let runner = LagiModelRunner::init_from_ram(&bytes, 4).unwrap();
        assert_eq!(runner.cpu_threads, 4);
        assert_eq!(runner.context_window, 8192);
        assert_eq!(runner.resident_bytes(), 36);
    }

    #[test]
    fn test_generate_thought_reports_temperature() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&[0u8; 16]);
        let runner = LagiModelRunner::init_from_ram(&bytes, 4).unwrap();
        let out = runner.generate_thought("hello", 0.85).unwrap();
        assert!(out.contains("0.85"));
        assert!(out.contains("hello"));
    }
}
