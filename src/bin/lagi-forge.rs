/*!
lagi-forge — INTERNAL Liumi Corporation No-Code Agent Forge

Interactive console wizard that seals a raw `.gguf` model into an encrypted,
RAM-resident `.lagi` agent container. The employee never writes code, a
manifest, or a codex by hand:

    1. Drag-and-drop / paste the path to a .gguf model file.
    2. Name the agent and describe it.
    3. Pick a Specialty Persona (baked into the .lbc codex system block).
    4. Pick a RAM tier (2G Eco / 4G Balanced / 6G Pro).

The wizard then:
    - generates a validated `manifest.json`,
    - compiles the Liumi Bytecode Codex (`.lbc`) with the specialty persona,
    - stages the workspace and seals it via AES-256-GCM (`pack_agent`),
    - writes `<AgentName>.lagi` into `dist/`.

DO NOT DISTRIBUTE THIS BINARY. It is internal Liumi Corporation IP.
*/

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use lagi_core::codex::{FORBIDDEN_STRINGS, IM_END, IM_START, LAGI_PERSONA};
use lagi_core::pack_agent;

/// RAM tiers offered by the forge.
struct RamTier {
    label: &'static str,
    context_window: u32,
    cpu_threads: u32,
}

const RAM_TIERS: &[RamTier] = &[
    RamTier {
        label: "2G Eco",
        context_window: 2048,
        cpu_threads: 2,
    },
    RamTier {
        label: "4G Balanced",
        context_window: 4096,
        cpu_threads: 4,
    },
    RamTier {
        label: "6G Pro",
        context_window: 8192,
        cpu_threads: 6,
    },
];

fn print_banner() {
    println!("============================================================");
    println!("   L A G I - F O R G E   ///  Liumi Corporation (internal)");
    println!("   No-Code .lagi Agent Sealer  v{}", lagi_core::VERSION);
    println!("============================================================");
    println!();
}

/// Read a non-empty line from stdin. Accepts drag-and-drop paths wrapped in
/// double quotes (Windows terminal behavior) and strips them.
fn prompt(label: &str) -> Result<String, io::Error> {
    loop {
        print!("{} ", label);
        io::stdout().flush()?;
        let mut buf = String::new();
        io::stdin().read_line(&mut buf)?;
        let trimmed = buf.trim().trim_matches('"').trim_matches('\'').trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
        println!("  (please enter a value)");
    }
}

/// Read a menu choice constrained to `1..=max`, or accept the default on empty.
fn prompt_choice(label: &str, max: usize, default: usize) -> Result<usize, io::Error> {
    loop {
        print!("{} [1-{}, default {}] ", label, max, default);
        io::stdout().flush()?;
        let mut buf = String::new();
        io::stdin().read_line(&mut buf)?;
        let t = buf.trim();
        if t.is_empty() {
            return Ok(default);
        }
        if let Ok(n) = t.parse::<usize>() {
            if n >= 1 && n <= max {
                return Ok(n);
            }
        }
        println!("  (enter a number between 1 and {})", max);
    }
}

/// Guess quantization from the model filename, defaulting to Q4_K_M.
fn guess_quantization(file_name: &str) -> &'static str {
    let lower = file_name.to_lowercase();
    if lower.contains("q4") {
        "Q4_K_M"
    } else if lower.contains("q5") {
        "Q5_K_M"
    } else if lower.contains("q6") {
        "Q6_K"
    } else if lower.contains("q8") {
        "Q8_0"
    } else if lower.contains("f16") || lower.contains("fp16") {
        "F16"
    } else {
        "Q4_K_M"
    }
}

/// Filesystem-safe agent name for the output artifact.
fn sanitize_name(name: &str) -> String {
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe = safe
        .trim_matches(|c: char| c == '.' || c == '_')
        .to_string();
    if safe.is_empty() {
        "Agent".to_string()
    } else {
        safe
    }
}
/// Compile the Liumi Bytecode Codex (.lbc): the immutable persona kernel with
/// the agent's specialty layered as a secondary locked context block.
fn compile_lbc_codex(specialty: &str) -> String {
    let mut lbc = String::with_capacity(2048);
    lbc.push_str("# .lbc — Liumi Bytecode Codex (compiled by lagi-forge)\n");
    lbc.push_str("# Locked ChatML framing + persona enforcement kernel.\n\n");
    lbc.push_str("[chatml]\n");
    lbc.push_str(&format!(
        "im_start_token_id = {}\n",
        lagi_core::IM_START_TOKEN_ID
    ));
    lbc.push_str(&format!(
        "im_end_token_id = {}\n",
        lagi_core::IM_END_TOKEN_ID
    ));
    lbc.push_str(&format!(
        "framing = \"{0}system\\n...{1}\\n{0}role\\n...{1}\"\n",
        IM_START, IM_END
    ));
    lbc.push_str("\n[bias_table]\n");
    lbc.push_str("# Every token below is clamped to -100.0 (impossible to emit).\n");
    for forbidden in FORBIDDEN_STRINGS {
        lbc.push_str(&format!("forbid = \"{}\"\n", forbidden));
    }
    lbc.push_str("\n[system_persona]\n");
    lbc.push_str("locked = true\n");
    lbc.push_str("body = <<<LAGI_PERSONA\n");
    lbc.push_str(LAGI_PERSONA);
    lbc.push_str("LAGI_PERSONA\n\n");
    lbc.push_str("[specialty]\n");
    lbc.push_str("locked = true\n");
    lbc.push_str(&format!(
        "body = \"{}\"\n",
        specialty.replace('\\', "\\\\").replace('"', "\\\"")
    ));
    lbc
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\n[forge error] {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    print_banner();

    // ---- Step 1: model file ------------------------------------------------
    println!("STEP 1/4 — Model weights");
    let gguf_raw = prompt("  Path to .gguf model file (drag & drop here):")?;
    let gguf_path = PathBuf::from(gguf_raw);
    if !gguf_path.is_file() {
        return Err(format!("model file not found: {}", gguf_path.display()).into());
    }
    if gguf_path.extension().and_then(|e| e.to_str()) != Some("gguf") {
        return Err(format!("expected a .gguf file, got: {}", gguf_path.display()).into());
    }
    let model_size_mb = fs::metadata(&gguf_path)?.len() as f64 / (1024.0 * 1024.0);
    println!(
        "  ✓ model: {} ({:.1} MB)\n",
        gguf_path.display(),
        model_size_mb
    );

    // ---- Step 2: identity --------------------------------------------------
    println!("STEP 2/4 — Agent identity");
    let agent_name = prompt("  Agent name:")?;
    let description = prompt("  Description:")?;
    let specialty = prompt("  Specialty persona (e.g. 'plasma physics research analyst'):")?;
    println!("  ✓ identity recorded\n");

    // ---- Step 3: RAM tier --------------------------------------------------
    println!("STEP 3/4 — RAM tier");
    for (i, tier) in RAM_TIERS.iter().enumerate() {
        println!(
            "    [{}] {:<12}  ctx: {:>5} tokens, threads: {}",
            i + 1,
            tier.label,
            tier.context_window,
            tier.cpu_threads
        );
    }
    let choice = prompt_choice("  Select tier", RAM_TIERS.len(), 2)?;
    let tier = &RAM_TIERS[choice - 1];
    println!("  ✓ tier: {}\n", tier.label);

    // ---- Step 4: seal ------------------------------------------------------
    println!("STEP 4/4 — Forging (no code required, forging automatically)...");

    let quantization = guess_quantization(
        gguf_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model.gguf"),
    );
    let model_file_name = gguf_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf")
        .to_string();

    // Stage a temporary agent workspace.
    let stage = std::env::temp_dir().join(format!("lagi-forge-{}", std::process::id()));
    if stage.exists() {
        fs::remove_dir_all(&stage)?;
    }
    fs::create_dir_all(&stage)?;

    // manifest.json — validated by lagi_core on pack.
    let manifest = serde_json::json!({
        "lagi_version": lagi_core::VERSION,
        "name": agent_name,
        "author": "Liumi Corporation",
        "description": description,
        "privacy_level": "AIR_GAPPED_LOCAL",
        "model": {
            "series": agent_name,
            "gguf_file": model_file_name,
            "quantization": quantization,
            "context_window": tier.context_window
        },
        "ideation": {
            "enabled": true,
            "interval_seconds": 1800,
            "spark_temperature": 0.85,
            "critic_temperature": 0.20,
            "min_confidence_score": 80
        }
    });
    fs::write(
        stage.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    // Compile and drop the .lbc codex into the payload.
    fs::write(stage.join("codex.lbc"), compile_lbc_codex(&specialty))?;

    // Copy the model weights into the workspace.
    let staged_model = stage.join(&model_file_name);
    fs::copy(&gguf_path, &staged_model)?;

    // Seal: AES-256-GCM envelope generated by the packager.
    let out_dir = Path::new("dist");
    fs::create_dir_all(out_dir)?;
    let out_file = out_dir.join(format!("{}.lagi", sanitize_name(&agent_name)));
    pack_agent(&stage, &out_file, None)?;

    // Clean up the staging area.
    fs::remove_dir_all(&stage).ok();

    let out_size_mb = fs::metadata(&out_file)?.len() as f64 / (1024.0 * 1024.0);
    println!();
    println!("============================================================");
    println!(
        "  ✔ SEALED: {}  ({:.1} MB)",
        out_file.display(),
        out_size_mb
    );
    println!("    persona:  {}", specialty);
    println!(
        "    tier:     {} (ctx {} / {} threads)",
        tier.label, tier.context_window, tier.cpu_threads
    );
    println!("    crypto:   AES-256-GCM + HKDF hardware key derivation");
    println!("------------------------------------------------------------");
    println!("  Launch it:");
    println!("    lagi chat {}", out_file.display());
    println!("    lagi serve {} --port 8080", out_file.display());
    println!("    lagi watch {}", out_file.display());
    println!("============================================================");
    Ok(())
}
