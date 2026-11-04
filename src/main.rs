/*!
LAGI CLI - Public Command Line Interface for LAGI Agents

Public subcommands (packaging & inspection are internal-only, exposed by the
internal `lagi-forge` binary):
    lagi chat     Interactive chat session with a .lagi agent
    lagi serve    OpenAI-compatible HTTP server
    lagi run      Launch the live RAM-resident agent with background ideation
    lagi watch    Watch the agent's imagination stream
*/

use clap::{Parser, Subcommand};
use lagi_core::runtime::{LagiRuntime, RuntimeEvent};
use lagi_core::MemoryGraph;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "lagi")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "LAGI - Liumi Autonomous General Intelligence Core Engine CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, global = true, help = "Enable verbose logging")]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the live RAM-resident runtime agent with background ideation
    Run {
        #[arg(help = "Path to .lagi container file")]
        file: PathBuf,

        #[arg(long, help = "Additional context for key derivation")]
        context: Option<String>,

        #[arg(
            long,
            help = "Stop after N completed ideation cycles (default: run indefinitely)"
        )]
        cycles: Option<u64>,

        #[arg(long, default_value = "auto", help = "RAM tier: auto | 2g | 4g | 6g")]
        ram: String,

        #[arg(long, default_value = "auto", help = "CPU threads: auto | number")]
        threads: String,
    },

    /// Interactive chat session with the agent (decrypts into RAM, opens a REPL)
    Chat {
        #[arg(help = "Path to .lagi container file")]
        file: PathBuf,

        #[arg(long, help = "Additional context for key derivation")]
        context: Option<String>,

        #[arg(long, default_value = "auto", help = "RAM tier: auto | 2g | 4g | 6g")]
        ram: String,

        #[arg(long, default_value = "auto", help = "CPU threads: auto | number")]
        threads: String,

        #[arg(
            long,
            help = "Scrub stale/mock entries (e.g. old 'LAGi' spellings) from memory before starting"
        )]
        clean_memory: bool,
    },

    /// OpenAI-compatible HTTP server (GET /v1/models, POST /v1/chat/completions)
    Serve {
        #[arg(help = "Path to .lagi container file")]
        file: PathBuf,

        #[arg(long, help = "Additional context for key derivation")]
        context: Option<String>,

        #[arg(long, default_value_t = 8080, help = "TCP port to listen on")]
        port: u16,

        #[arg(long, default_value = "127.0.0.1", help = "Bind address")]
        host: String,

        #[arg(long, default_value = "auto", help = "RAM tier: auto | 2g | 4g | 6g")]
        ram: String,

        #[arg(long, default_value = "auto", help = "CPU threads: auto | number")]
        threads: String,

        #[arg(
            long,
            help = "Scrub stale/mock entries (e.g. old 'LAGi' spellings) from memory before starting"
        )]
        clean_memory: bool,
    },

    /// Watch the agent's imagination stream as it produces new thoughts
    Watch {
        #[arg(help = "Path to .lagi container file")]
        file: PathBuf,

        #[arg(long, help = "Additional context for key derivation")]
        context: Option<String>,
    },

    /// Export hardware-bound workspace memory into a secure .lmb bundle
    Export {
        #[arg(help = "Path to source .lagi container file")]
        file: PathBuf,

        #[arg(long, help = "Destination .lmb migration bundle")]
        out: PathBuf,

        #[arg(long, help = "Migration passphrase (prompted securely when omitted)")]
        passphrase: Option<String>,
    },

    /// Import a secure .lmb bundle into a .lagi container
    Import {
        #[arg(help = "Path to destination .lagi container file")]
        file: PathBuf,

        #[arg(long, help = "Source .lmb migration bundle")]
        from: PathBuf,

        #[arg(long, help = "Migration passphrase (prompted securely when omitted)")]
        passphrase: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
    tracing::subscriber::set_global_default(subscriber)?;

    match cli.command {
        Commands::Run {
            file,
            context,
            cycles,
            ram,
            threads,
        } => {
            run_agent(&file, context.as_deref(), cycles, &ram, &threads).await?;
        }
        Commands::Chat {
            file,
            context,
            ram,
            threads,
            clean_memory,
        } => {
            run_chat(&file, context.as_deref(), &ram, &threads, clean_memory).await?;
        }
        Commands::Serve {
            file,
            context,
            port,
            host,
            ram,
            threads,
            clean_memory,
        } => {
            #[cfg(feature = "serve")]
            {
                lagi_core::serve::run_server(
                    &file,
                    context.as_deref(),
                    &host,
                    port,
                    &ram,
                    &threads,
                    clean_memory,
                )
                .await?;
            }
            #[cfg(not(feature = "serve"))]
            {
                let _ = (&file, &context, &port, &host, &ram, &threads, &clean_memory);
                anyhow::bail!("serve requires building with --features serve");
            }
        }
        Commands::Watch { file, context } => {
            run_watch(&file, context.as_deref()).await?;
        }
        Commands::Export {
            file,
            out,
            passphrase,
        } => {
            run_export(&file, &out, passphrase)?;
        }
        Commands::Import {
            file,
            from,
            passphrase,
        } => {
            run_import(&file, &from, passphrase)?;
        }
    }

    Ok(())
}

fn read_passphrase(confirm: bool) -> Result<String, Box<dyn std::error::Error>> {
    let passphrase = rpassword::prompt_password("Migration passphrase: ")?;
    if confirm {
        let confirmation = rpassword::prompt_password("Confirm passphrase: ")?;
        if passphrase != confirmation {
            return Err("passphrases do not match".into());
        }
    }
    if passphrase.is_empty() {
        return Err("passphrase must not be empty".into());
    }
    Ok(passphrase)
}

fn run_export(
    container: &Path,
    out: &Path,
    passphrase: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let passphrase = match passphrase {
        Some(value) => value,
        None => read_passphrase(true)?,
    };
    let bytes = lagi_core::export_workspace(container, out, &passphrase)?;
    println!("Exported {} bytes to {}", bytes, out.display());
    Ok(())
}

fn run_import(
    container: &Path,
    from: &Path,
    passphrase: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let passphrase = match passphrase {
        Some(value) => value,
        None => read_passphrase(false)?,
    };
    let entries = lagi_core::import_workspace(container, from, &passphrase)?;
    println!(
        "Imported {} memory entries into {}",
        entries,
        container.display()
    );
    Ok(())
}

async fn run_agent(
    file: &Path,
    context: Option<&str>,
    max_cycles: Option<u64>,
    ram_arg: &str,
    threads_arg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Loading container: {}", file.display());

    // Hardware auto-sense badge (zero flags required).
    let hw = lagi_core::resolve_hardware_settings(ram_arg, threads_arg);
    println!("{}", lagi_core::hardware_badge(&hw));

    // The CLI is just another host application — it embeds LagiRuntime
    let runtime = LagiRuntime::load(file, context.map(|s| s.as_bytes())).await?;

    let manifest = runtime.manifest();
    info!("Agent: {}", manifest.name);
    info!("Model: LAGI-Edge-1B builtin engine (4 CPU threads)");
    info!("Memory: {}", runtime.memory_path().display());

    let loop_cfg = manifest.ideation_settings();
    info!(
        "Ideation: spark={} critic={}, ticking every {}ms, idle trigger {}s",
        loop_cfg.temperature_spark,
        loop_cfg.temperature_critic,
        loop_cfg.interval_ms,
        loop_cfg.idle_trigger_seconds,
    );

    // Bypass the idle gate for CLI runs so ideation starts immediately
    runtime.mark_idle();

    runtime.start().await?;

    // Live stream of what the agent is imagining
    let mut thoughts = runtime.subscribe_thoughts();
    let thought_forwarder = tokio::spawn(async move {
        while let Ok(t) = thoughts.recv().await {
            println!(
                "  [cycle {}] {}",
                t.cycle,
                t.critic.chars().take(120).collect::<String>()
            );
        }
    });

    info!("LAGI agent live. 5ms heartbeat driving autonomous ideation.");

    if let Some(max) = max_cycles {
        info!("Running until {} ideation cycles complete...", max);
        loop {
            if runtime.cycle_count() >= max {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        runtime.stop().await?;
        info!("Agent stopped. Total cycles: {}", runtime.cycle_count());
    } else {
        tokio::signal::ctrl_c().await?;
        runtime.stop().await?;
        info!("Agent stopped. Total cycles: {}", runtime.cycle_count());
    }

    thought_forwarder.abort();

    // Single-file container persistence: flush the ideation memory graph
    // into the .lagi trailing state block on exit (payload untouched).
    if let Ok(mem) = MemoryGraph::new(runtime.memory_path(), None) {
        match lagi_core::flush_memory_to_container(&mem, file) {
            Ok(bytes) => info!(".lvc state flushed into container ({} bytes)", bytes),
            Err(e) => info!("state flush skipped: {}", e),
        }
    }

    Ok(())
}

async fn run_chat(
    file: &Path,
    context: Option<&str>,
    ram_arg: &str,
    threads_arg: &str,
    clean_memory: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Loading container: {}", file.display());

    // Hardware auto-sense: RAM tier -> (context_window, n_batch), smart
    // threads. Default is 100% automatic; --ram/--threads can override.
    let hw = lagi_core::resolve_hardware_settings(ram_arg, threads_arg);
    println!("{}", lagi_core::hardware_badge(&hw));
    let ctx_override = Some(hw.context_window as u32);
    let n_batch = hw.n_batch as u32;

    let context_bytes = context.map(|s| s.as_bytes());
    let unpackager = lagi_core::LagiUnpackager::new(file);
    let container = unpackager.unpack(context_bytes)?;

    let manifest = container.manifest.clone();
    info!("Agent: {}", manifest.name);

    let model_bytes = container
        .get_model()
        .ok_or("Model asset missing from container")?;
    let model_cfg = lagi_core::ModelRuntimeConfig {
        cpu_threads: hw.threads,
        context_window: hw.context_window as u32,
        ..manifest.runtime_model()
    };

    let memory_cfg = manifest.memory_settings();
    let memory_path = file
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&memory_cfg.db_name);
    let memory = Arc::new(MemoryGraph::new(&memory_path, None)?);

    // Single-file container persistence: hydrate the .lvc graph from the
    // trailing state block of the host .lagi (if present).
    match lagi_core::load_memory_from_container(file, &memory) {
        Ok(n) if n > 0 => info!("Restored {} .lvc memories from container state block", n),
        Ok(_) => {}
        Err(e) => info!("No container state block loaded: {}", e),
    }

    // --clean-memory: scrub stale/mock entries (e.g. old 'LAGi' spellings)
    // recalled from earlier sessions — runs AFTER hydration so entries
    // hiding inside the container state block are scrubbed too.
    if clean_memory {
        match memory.delete_matching("LAGi") {
            Ok(n) if n > 0 => println!(
                "  Cleaned memory: removed {} stale 'LAGi' entr{}",
                n,
                if n == 1 { "y" } else { "ies" }
            ),
            Ok(_) => println!("  Cleaned memory: no stale entries found"),
            Err(e) => println!("  Memory cleanup failed: {}", e),
        }
    }

    // Exit hook: on /exit, EOF or Ctrl+C, flush dirty .lvc memory back into
    // the .lagi trailing state block without touching the model payload.
    let container_path = file.to_path_buf();
    let flush_memory = memory.clone();
    let on_exit: Box<dyn FnOnce() + Send> = Box::new(move || {
        match lagi_core::flush_memory_to_container(&flush_memory, &container_path) {
            Ok(bytes) => println!("  .lvc state flushed into container ({} bytes)", bytes),
            Err(e) => println!("  state flush failed: {}", e),
        }
    });

    #[cfg(feature = "llama")]
    {
        info!("Real inference backend enabled (llama.cpp) + Synaptic Recall + Codex");
        let staged = lagi_core::chat::stage_model_to_temp(model_bytes)?;
        let engine = lagi_core::chat::llama_engine::ChatEngine::load_with_overrides(
            &staged,
            &model_cfg,
            ctx_override,
            n_batch,
        )?;
        let synaptic = lagi_core::SynapticMemoryManager::new(memory.clone());
        let result = lagi_core::chat::run_repl(
            memory,
            model_cfg,
            move |history| {
                let last_user = history
                    .iter()
                    .rev()
                    .find(|(r, _)| r == "user")
                    .map(|(_, c)| c.clone());
                // Synaptic Recall: inject remembered facts before generation.
                let mut full = history.to_vec();
                if let Some(q) = &last_user {
                    if let Some(block) = synaptic.recall_context(q) {
                        full.insert(1, ("system".to_string(), block));
                    } else if lagi_core::SynapticMemoryManager::is_factual_question(q) {
                        // Auto-Teach Loop: unknown factual query -> dual-engine
                        // web search, distillation, and background .lvc commit.
                        let teach = synaptic.clone();
                        let q = q.clone();
                        if let Ok(handle) = tokio::runtime::Handle::try_current() {
                            handle.spawn(async move {
                                if let Ok(n) = teach.auto_teach(&q).await {
                                    if n > 0 {
                                        println!("\n[auto-taught {} new fact(s) into .lvc]", n);
                                        use std::io::Write;
                                        let _ = std::io::stdout().flush();
                                    }
                                }
                            });
                        }
                    }
                }
                let out = engine.respond(&full);
                // Synaptic Commit: persist any newly stated preferences.
                if let Some(q) = &last_user {
                    synaptic.extract_and_commit(q);
                }
                out
            },
            true,
            Some(on_exit),
        );
        let _ = std::fs::remove_file(&staged);
        return result.map_err(Into::into);
    }

    #[cfg(not(feature = "llama"))]
    {
        let _ = (model_bytes, ctx_override, n_batch); // unused without the llama backend
        info!("Placeholder inference (build with --features llama for real decoding)");
        return Ok(lagi_core::chat::run_placeholder_repl(
            memory,
            model_cfg,
            Some(on_exit),
        )?);
    }
}

async fn run_watch(file: &Path, context: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    info!("Loading container: {}", file.display());

    let runtime = LagiRuntime::load(file, context.map(|s| s.as_bytes())).await?;
    runtime.mark_idle();
    runtime.start().await?;

    println!("Watching LAGI think... (Ctrl+C to stop)\n");

    let mut events = runtime.subscribe();
    let mut last_cycles = 0u64;
    loop {
        tokio::select! {
            evt = events.recv() => {
                match evt {
                    Ok(RuntimeEvent::Heartbeat(n)) if n % 200 == 0 => {
                        println!(
                            "  [heartbeat {:>5}]  cycles so far: {}",
                            n,
                            runtime.cycle_count()
                        );
                    }
                    Ok(RuntimeEvent::Heartbeat(_)) => {}
                    Ok(RuntimeEvent::Thought(t)) => {
                        println!(
                            "\n💡 thought #{} at {}\n   spark:  {}\n   critic: {}\n",
                            t.cycle,
                            t.timestamp_ms,
                            t.spark.chars().take(100).collect::<String>(),
                            t.critic.chars().take(200).collect::<String>()
                        );
                        last_cycles = t.cycle;
                    }
                    Ok(RuntimeEvent::Lifecycle { running, cycles }) => {
                        println!("  [lifecycle] running={} cycles={}", running, cycles);
                        last_cycles = cycles;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        println!("  (skipping {} buffered events)", n);
                    }
                    Err(_) => break,
                }
                let _ = last_cycles;
            }
            _ = tokio::signal::ctrl_c() => break,
        }
    }

    runtime.stop().await?;
    println!(
        "\nStopped. Total ideation cycles: {}",
        runtime.cycle_count()
    );
    Ok(())
}
