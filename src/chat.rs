/*!
Interactive Chat REPL

Conversational front-end for a .lagi container. Decrypts the container into
RAM, then runs a read-eval-print loop against the embedded GGUF model.

Build with `--features llama` for real local inference via llama.cpp.
Without that feature flag a deterministic placeholder engine is used so the
REPL UX (history, memory persistence, commands) remains testable.
*/

use crate::manifest::ModelRuntimeConfig;
use crate::memory::MemoryGraph;
use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;

/// Commands understood at the REPL besides raw text
const HELP_TEXT: &str = "Commands:
  /exit | /quit   - close the chat
  /clear          - clear conversation history
  /memory         - show recent thoughts stored in vector memory
  /help           - show this help";

struct ChatState {
    history: Vec<(String, String)>, // (role, content)
    memory: Arc<MemoryGraph>,
    max_history: usize,
}

impl ChatState {
    fn push(&mut self, role: &str, content: &str) {
        if self.history.len() >= self.max_history * 2 {
            let over = self.history.len() - self.max_history * 2 + 2;
            self.history.drain(..over);
        }
        self.history.push((role.to_string(), content.to_string()));
    }

    fn persist_turn(&self, user: &str, assistant: &str) {
        let _ = self
            .memory
            .store_thought(&format!("User: {}\nAssistant: {}", user, assistant));
    }

    fn print_memory(&self, limit: usize) {
        match self.memory.get_recent(None, limit) {
            Ok(entries) => {
                if entries.is_empty() {
                    println!("  (memory is empty)");
                }
                for e in entries {
                    let preview: String = e.content.chars().take(120).collect();
                    println!("  [{:?}] {}", e.memory_type, preview);
                }
            }
            Err(err) => println!("  memory error: {}", err),
        }
    }
}

fn repl_prefix() -> &'static str {
    "\nlagi> "
}

/// Stream-level identity scrub for the live token loop.
///
/// `guard_output` only runs on the final accumulated string, so a streamed
/// "LAGi" would already be on screen by then. Because the pattern can span
/// token boundaries ("LAG" | "Gi"), a 4-character sliding window is held
/// back and normalized before printing.
///
/// Returns `(text_to_print, new_held_buffer)`.
pub fn stream_scrub(held: &str, piece: &str) -> (String, String) {
    let mut buf = String::with_capacity(held.len() + piece.len());
    buf.push_str(held);
    buf.push_str(piece);
    let buf = buf.replace("LAGi", "LAGI");
    let chars: Vec<char> = buf.chars().collect();
    if chars.len() > 4 {
        let split = chars.len() - 4;
        (
            chars[..split].iter().collect(),
            chars[split..].iter().collect(),
        )
    } else {
        (String::new(), buf)
    }
}

/// Shared REPL driver. `respond` maps history to an assistant reply.
///
/// `streamed=true` means `respond` already wrote each token to stdout during
/// generation (print+flush); the driver then must NOT re-print the final
/// accumulated buffer — it only emits the terminating newline.
///
/// `on_exit` runs exactly once when the REPL terminates (EOF, `/exit` or
/// `/quit`) — used to flush dirty `.lvc` state into the host `.lagi` file.
pub fn run_repl(
    memory: Arc<MemoryGraph>,
    model_cfg: ModelRuntimeConfig,
    respond: impl Fn(&[(String, String)]) -> Result<String>,
    streamed: bool,
    on_exit: Option<Box<dyn FnOnce() + Send>>,
) -> Result<()> {
    let mut state = ChatState {
        history: Vec::new(),
        memory,
        max_history: 8,
    };

    // Seed the assistant persona through the .lbc codex (immutable block).
    state.push(
        "system",
        crate::codex::LAGI_PERSONA,
    );

    println!("\n=== LAGI Interactive Chat ===");
    println!("Model: LAGI-Edge-1B builtin engine ({} ctx, {} threads)", model_cfg.context_window, model_cfg.cpu_threads);
    println!("Type /help for commands, /exit to quit.");

    // --- Proactive timer scheduler wired to the background heartbeat ---
    // A ticker task checks for expired reminders once per second, firing a
    // native OS desktop notification + console message without ever blocking
    // the REPL prompt.
    let scheduler = Arc::new(crate::scheduler::TaskScheduler::new());
    // Wire VectorCore memory so expired deferred-tool tasks can be
    // dispatched through ToolDispatcher when their timer fires.
    scheduler.set_tool_memory(state.memory.clone());
    let tick_scheduler = scheduler.clone();
    let ticker_started = tokio::runtime::Handle::try_current().map(|handle| {
        handle.spawn(async move {
            let _ = crate::scheduler::TaskScheduler::spawn_tick(tick_scheduler).await;
        })
    });
    if ticker_started.is_err() {
        // No async runtime (e.g. embedded/sync caller): fall back to a plain
        // OS thread so timers still fire.
        let thread_scheduler = scheduler.clone();
        std::thread::Builder::new()
            .name("lagi-timer-tick".into())
            .spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                thread_scheduler.check_and_fire();
            })
            .ok();
    }

    loop {
        use std::io::Write;
        print!("{}", repl_prefix());
        std::io::stdout().flush()?;

        let mut line = String::new();
        let n = std::io::stdin().read_line(&mut line)?;
        if n == 0 {
            break; // EOF / piped stdin
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/exit" | "/quit" => break,
            "/clear" => {
                state.history.drain(1..);
                println!("  history cleared");
                continue;
            }
            "/memory" => {
                state.print_memory(5);
                continue;
            }
            "/help" => {
                println!("{}", HELP_TEXT);
                continue;
            }
            _ => {}
        }

        // --- Native tool dispatch (web search, PC control, browser, clipboard) ---
        // LAGI-Edge-1B cannot emit reliable JSON tool calls, so the host
        // runtime detects tool intents directly and executes them, committing
        // an episodic record to Liumi VectorCore. Non-tool input falls through
        // to the LLM unchanged.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let tool_memory = state.memory.clone();
            let tool_input = input.to_string();
            let dispatch_scheduler = scheduler.clone();
            if let Some(out) = tokio::task::block_in_place(|| {
                handle.block_on(async move {
                    let dispatcher = crate::tools::ToolDispatcher::with_scheduler(
                        tool_memory,
                        dispatch_scheduler,
                    );
                    dispatcher.dispatch(&tool_input).await
                })
            }) {
                // Print the raw observation, then — crucially — push it into
                // conversation history as a Tool Observation system block so
                // Qwen reads it, synthesizes an answer for the user, and
                // retains it for multi-turn follow-up questions.
                println!("\n[{}]", out.tool);
                println!("{}\n", out.text);
                state.push("user", input);
                state.push(
                    "system",
                    &format!(
                        "[Tool Observation: {}]\n{}\n\n[Instruction: Answer the user's question directly using the specific numbers, facts, and details from the observation above. Do not tell the user to visit websites or click links. State the factual answer concisely.]",
                        out.tool, out.text
                    ),
                );
                let prompt_view = state.history.clone();
                match respond(&prompt_view) {
                    Ok(reply) => {
                        if !streamed {
                            println!("\n{}", reply.trim());
                        } else {
                            println!();
                        }
                        state.push("assistant", reply.trim());
                        auto_save_and_run(input, reply.trim());
                        state.persist_turn(input, reply.trim());
                    }
                    Err(e) => {
                        println!("\n[inference error: {}]", e);
                    }
                }
                continue;
            }
        }

        state.push("user", input);
        let prompt_view = state.history.clone();
        match respond(&prompt_view) {
            Ok(reply) => {
                if !streamed {
                    // Reply was NOT emitted token-by-token; print it once.
                    println!("\n{}", reply.trim());
                } else {
                    // Tokens already streamed to stdout — just terminate the line.
                    println!();
                }
                state.push("assistant", reply.trim());
                auto_save_and_run(input, reply.trim());
                state.persist_turn(input, reply.trim());
            }
            Err(e) => {
                println!("\n[inference error: {}]", e);
            }
        }
    }

    println!("\nSession closed.");

    // Flush dirty .lvc state (RAII drop follows inside the caller).
    if let Some(flush) = on_exit {
        flush();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Auto-save & auto-run pipeline for generated code
// ---------------------------------------------------------------------------

/// A fenced code block extracted from an assistant reply.
struct ExtractedCodeBlock {
    lang: String,
    body: String,
}

/// Extract all fenced markdown code blocks (```lang ... ```) from a reply.
fn extract_code_blocks(reply: &str) -> std::vec::Vec<ExtractedCodeBlock> {
    let mut blocks = std::vec::Vec::new();
    let mut rest = reply;
    while let Some(start) = rest.find("```") {
        let after_fence = &rest[start + 3..];
        // Language tag is the first token on the opening fence line.
        let line_end = after_fence.find('\n').unwrap_or(after_fence.len());
        let lang = after_fence[..line_end]
            .trim()
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();
        let body_start = (start + 3 + line_end).min(rest.len());
        let body = &rest[body_start..];
        let Some(end_rel) = body.find("```") else {
            break;
        };
        if !lang.is_empty() && lang != "text" && lang != "txt" {
            blocks.push(ExtractedCodeBlock {
                lang,
                body: body[..end_rel].trim().to_string(),
            });
        }
        rest = &body[end_rel + 3..];
    }
    blocks
}

/// True when the user's prompt asks LAGI to produce/save runnable code
/// (e.g. 'create a demo gym app in html and save the code files in lixo').
fn wants_auto_save(prompt: &str) -> bool {
    let lower = prompt.to_lowercase();
    let has_generative = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|w| {
            matches!(
                w,
                "app" | "html" | "website" | "script" | "code" | "demo" | "program"
                    | "python" | "webpage"
            )
        });
    if !has_generative {
        return false;
    }
    let has_save_hint = ["save", "in lixo", "on desktop", "in desktop", "folder"]
        .iter()
        .any(|h| lower.contains(h));
    let has_creation_verb = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|w| matches!(w, "create" | "make" | "build" | "write" | "generate" | "develop"));
    has_save_hint || has_creation_verb
}

/// Default file extension for a fenced code-block language tag.
fn default_ext(lang: &str) -> String {
    match lang {
        "python" => "py".to_string(),
        "javascript" => "js".to_string(),
        "rust" => "rs".to_string(),
        "text" | "txt" | "plain" => "txt".to_string(),
        "c++" | "cpp" => "cpp".to_string(),
        other => other.to_string(),
    }
}

/// Infer an output file name for a code block: an explicit 'named X' /
/// 'called X' / 'filename X' marker in the prompt wins, otherwise a
/// per-language default (html gives index.html, python gives app.py, ...).
fn infer_file_name(prompt: &str, lang: &str) -> String {
    let lower = prompt.to_lowercase();
    const MARKERS: &[&str] = &[
        "named as ", "named ", "name it ", "called ", "file name ", "filename ",
    ];
    for marker in MARKERS {
        if let Some(pos) = lower.find(marker) {
            let mut rest = &lower[pos + marker.len()..];
            for cut in [" on ", " in ", " under ", " inside ", " and ", ", ", " with "] {
                if let Some(cpos) = rest.find(cut) {
                    rest = &rest[..cpos];
                }
            }
            let cleaned: String = rest
                .trim()
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
                .collect();
            if !cleaned.is_empty() {
                return if cleaned.contains('.') {
                    cleaned
                } else {
                    format!("{}.{}", cleaned, default_ext(lang))
                };
            }
        }
    }
    match lang {
        "html" | "htm" => "index.html".to_string(),
        "python" | "py" => "app.py".to_string(),
        "javascript" | "js" => "app.js".to_string(),
        "css" => "style.css".to_string(),
        "rust" | "rs" => "main.rs".to_string(),
        _ => format!("app.{}", default_ext(lang)),
    }
}

/// Resolve the save target folder from the prompt:
/// 'in (the) lixo folder' maps to `desktop/lixo`, otherwise `desktop`.
fn infer_target_folder(prompt: &str) -> String {
    let lower = prompt.to_lowercase();
    if let Some(pos) = lower.find(" folder") {
        let word = lower[..pos]
            .trim_end()
            .split_whitespace()
            .last()
            .unwrap_or("");
        if !matches!(
            word,
            "" | "the" | "a" | "this" | "that" | "my" | "in" | "on" | "under" | "inside" | "new"
        ) {
            return format!("desktop/{}", word);
        }
    }
    "desktop".to_string()
}

/// Post-generation handler: if the user asked for code/app generation with a
/// save location, extract fenced code blocks from the reply, write each one
/// to disk via `PcToolExecutor::write_file`, and automatically launch the
/// first runnable artifact (.html in the default browser, .py via `python`).
fn auto_save_and_run(prompt: &str, reply: &str) {
    let blocks = extract_code_blocks(reply);
    if blocks.is_empty() || !wants_auto_save(prompt) {
        return;
    }
    let folder = infer_target_folder(prompt);
    let exec = crate::tools::pc_control::PcToolExecutor::new();
    let mut used_names: std::vec::Vec<String> = std::vec::Vec::new();
    let mut runnable: Option<std::path::PathBuf> = None;

    for (idx, block) in blocks.iter().enumerate() {
        let mut name = infer_file_name(prompt, &block.lang);
        if used_names.iter().any(|n| n == &name) {
            let (stem, ext) = match name.rsplit_once('.') {
                Some((s, e)) => (s.to_string(), e.to_string()),
                None => (name.clone(), "txt".to_string()),
            };
            name = format!("{}_{}.{}", stem, idx + 1, ext);
        }
        used_names.push(name.clone());

        match exec.write_file(&folder, &name, &block.body) {
            Ok(msg) => {
                // `write_file` reports the full absolute path in its message.
                const WRITTEN_AT: &str = "written at ";
                let path = msg
                    .find(WRITTEN_AT)
                    .map(|p| std::path::PathBuf::from(msg[p + WRITTEN_AT.len()..].trim()));
                println!("\u{1F4C1} Saved: {}", msg);
                let ext_ok = matches!(
                    name.rsplit('.').next(),
                    Some("html") | Some("htm") | Some("py")
                );
                if ext_ok && runnable.is_none() {
                    runnable = path;
                }
            }
            Err(err) => println!("\u{26A0} Could not save {}: {}", name, err),
        }
    }

    if let Some(path) = runnable {
        match exec.launch_or_run_file(&path) {
            Ok(out) => println!("\u{1F680} Running App: {}", out),
            Err(err) => println!("\u{26A0} Could not run app: {}", err),
        }
    }
}

/// Placeholder responder used in the default (no llama feature) build
pub fn run_placeholder_repl(
    memory: Arc<MemoryGraph>,
    model_cfg: ModelRuntimeConfig,
    on_exit: Option<Box<dyn FnOnce() + Send>>,
) -> Result<()> {
    run_repl(memory, model_cfg, |history| {
        let last = history.last().map(|(_, c)| c.as_str()).unwrap_or("");
        Ok(format!(
            "[placeholder model] I received: \"{}\". Rebuild with `--features llama` for real inference.",
            last.chars().take(80).collect::<String>()
        ))
    }, false, on_exit)
}

/// Materialize the decrypted model to a secure temp file and hand back the path
/// (llama.cpp needs a file path; it is deleted when the session ends).
pub fn stage_model_to_temp(model_bytes: &[u8]) -> Result<PathBuf> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("lagi-model-{}.gguf", std::process::id()));
    std::fs::write(&path, model_bytes)?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// Real inference engine (requires `--features llama`)
// ---------------------------------------------------------------------------

#[cfg(feature = "llama")]
pub mod llama_engine {
    use super::*;
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel};
    use llama_cpp_2::sampling::LlamaSampler;
    use std::num::NonZeroU32;

    /// Real GGUF-backed chat engine driving the REPL
    pub struct ChatEngine {
        backend: LlamaBackend,
        model: LlamaModel,
        n_ctx: u32,
        /// Prompt-processing batch size (from the auto-sensed RAM tier)
        n_batch: u32,
        cpu_threads: i32,
        max_gen_tokens: usize,
        /// .lbc codex bias table: forbidden identity/refusal tokens at -100.0
        codex_biases: Vec<crate::codex::CodexBias>,
        n_vocab: i32,
    }

    impl ChatEngine {
        /// Load a GGUF model from a staged temp file on CPU
        pub fn load(model_path: &std::path::Path, cfg: &ModelRuntimeConfig) -> Result<Self> {
            Self::load_with_overrides(model_path, cfg, None, 2048)
        }

        /// Load with hardware auto-sense overrides (context window / batch).
        pub fn load_with_overrides(
            model_path: &std::path::Path,
            cfg: &ModelRuntimeConfig,
            ctx_override: Option<u32>,
            n_batch: u32,
        ) -> Result<Self> {
            // Silence llama.cpp native logging (llama.log, ggml diagnostics)
            static INIT_ONCE: std::sync::Once = std::sync::Once::new();
            INIT_ONCE.call_once(|| {
                llama_cpp_2::send_logs_to_tracing(
                    llama_cpp_2::LogOptions::default().with_logs_enabled(false),
                );
            });

            let backend = LlamaBackend::init()?;

            let model = LlamaModel::load_from_file(
                &backend,
                model_path,
                &LlamaModelParams::default(),
            )?;

            // .lbc codex enforcement: tokenize forbidden strings and clamp
            // every constituent token to -100.0 during sampling.
            let codex_biases = crate::codex::llama_support::codex_biases_for_model(&model);
            let n_vocab = model.n_vocab();

            Ok(Self {
                backend,
                model,
                n_ctx: ctx_override
                    .unwrap_or(cfg.context_window)
                    .max(1024),
                n_batch: n_batch.max(32),
                cpu_threads: cfg.cpu_threads as i32,
                max_gen_tokens: 512,
                codex_biases,
                n_vocab,
            })
        }

        /// Generate one assistant reply for the given conversation history
        pub fn respond(&self, history: &[(String, String)]) -> Result<String> {
            self.run_inference(history, false)
        }

        /// Raw completion used by the ideation loop (quiet, single-shot)
        pub fn think(&self, prompt: &str, temperature: f32) -> Result<String> {
            let history = vec![
                (
                    "system".to_string(),
                    "You are LAGI, a Liumi Autonomous General Intelligence agent.".to_string(),
                ),
                ("user".to_string(), prompt.to_string()),
            ];
            self.run_inference_temp(&history, temperature, true)
        }

        fn run_inference(&self, history: &[(String, String)], quiet: bool) -> Result<String> {
            self.run_inference_temp(history, 0.7, quiet)
        }

        fn run_inference_temp(
            &self,
            history: &[(String, String)],
            temperature: f32,
            quiet: bool,
        ) -> Result<String> {
            // .lbc codex: bypass apply_chat_template entirely and inject the
            // hardcoded ChatML token stream with the immutable persona first.
            let prompt = crate::codex::build_chatml(history);
            let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;

            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(self.n_ctx))
                .with_n_threads(self.cpu_threads)
                .with_n_batch(self.n_batch);
            let mut ctx = self.model.new_context(&self.backend, ctx_params)?;

            // Chunked prefill: decode the prompt in slices of n_batch so long
            // histories with Tool Observations never trip
            // GGML_ASSERT(n_tokens_all <= cparams.n_batch). Logits are only
            // requested on the final token of the final chunk.
            let batch_size = self.n_batch as usize;
            let chunk_count = tokens.chunks(batch_size).count();
            let mut n_cur: i32 = 0;
            for (ci, chunk) in tokens.chunks(batch_size).enumerate() {
                let last_chunk = ci == chunk_count - 1;
                let mut batch = LlamaBatch::new(chunk.len().max(1), 1);
                for (i, &tok) in chunk.iter().enumerate() {
                    let want_logits = last_chunk && i == chunk.len() - 1;
                    batch.add(tok, n_cur + i as i32, &[0], want_logits)?;
                }
                ctx.decode(&mut batch)?;
                n_cur += chunk.len() as i32;
            }
            // .lbc codex sampling chain: forbidden-token logit bias (-100.0)
            // is applied FIRST, before the normal top-k/top-p/temp/dist stack.
            let codex_sampler = LlamaSampler::logit_bias(
                self.n_vocab,
                &crate::codex::llama_support::to_llama_logit_biases(&self.codex_biases),
            );
            let mut sampler = LlamaSampler::chain(
                [
                    codex_sampler,
                    LlamaSampler::top_k(40),
                    LlamaSampler::top_p(0.95, 1),
                    LlamaSampler::temp(temperature),
                    LlamaSampler::dist(1_234_567_890),
                ],
                false,
            );

            let mut out = String::new();
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            // Unprinted tail (<= 4 chars) held back for the stream scrub.
            let mut held = String::new();

            for _ in 0..self.max_gen_tokens {
                let token = sampler.sample(&ctx, -1);
                sampler.accept(token);

                if self.model.is_eog_token(token) {
                    break;
                }

                let piece = self.model.token_to_piece(token, &mut decoder, true, None)?;
                if !quiet {
                    // Stream-level .lbc scrub: normalize 'LAGi' (possibly
                    // spanning token boundaries) before it hits the screen.
                    let (head, new_held) = super::stream_scrub(&held, &piece);
                    held = new_held;
                    if !head.is_empty() {
                        print!("{}", head);
                        use std::io::Write;
                        std::io::stdout().flush().ok();
                    }
                }
                out.push_str(&piece);

                let mut next = LlamaBatch::new(1, 1);
                next.add(token, n_cur, &[0], true)?;
                n_cur += 1;
                ctx.decode(&mut next)?;
            }
            if !quiet {
                // Flush the held-back tail (scrubbed) to close the stream.
                print!("{}", held.replace("LAGi", "LAGI"));
                println!();
            }

            // .lbc codex output guard: scrub any multi-token identity leakage.
            Ok(crate::codex::guard_output(out.trim()))
        }
    }

    // The same real engine backs the continuous ideation loop
    impl crate::model::ThoughtGenerator for ChatEngine {
        fn generate_thought(&self, prompt: &str, temperature: f32) -> Result<String> {
            self.think(prompt, temperature)
        }
    }

    /// Build an engine from in-RAM model bytes (stages a temp file internally
    /// because llama.cpp loads from paths; the file is owned by the returned
    /// engine and the caller is responsible for deleting it after drop).
    pub fn engine_from_ram(
        model_bytes: &[u8],
        model_cfg: &ModelRuntimeConfig,
    ) -> Result<(ChatEngine, std::path::PathBuf)> {
        engine_from_ram_with_overrides(model_bytes, model_cfg, None, 2048)
    }

    /// Same as [`engine_from_ram`] with hardware auto-sense overrides.
    pub fn engine_from_ram_with_overrides(
        model_bytes: &[u8],
        model_cfg: &ModelRuntimeConfig,
        ctx_override: Option<u32>,
        n_batch: u32,
    ) -> Result<(ChatEngine, std::path::PathBuf)> {
        let staged = super::stage_model_to_temp(model_bytes)?;
        let engine =
            ChatEngine::load_with_overrides(&staged, model_cfg, ctx_override, n_batch)?;
        Ok((engine, staged))
    }

    /// Full REPL: stage model to temp file, load engine, drive conversation
    pub fn run_real_repl(
        model_bytes: &[u8],
        model_cfg: ModelRuntimeConfig,
        memory: Arc<MemoryGraph>,
        on_exit: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<()> {
        let staged = super::stage_model_to_temp(model_bytes)?;
        let engine = ChatEngine::load(&staged, &model_cfg)?;
        let result = super::run_repl(memory, model_cfg, |history| engine.respond(history), true, on_exit);
        let _ = std::fs::remove_file(&staged);
        result
    }
}

#[cfg(test)]
mod auto_save_tests {
    use super::*;

    #[test]
    fn test_extract_code_blocks() {
        let reply = "Here you go:\n```html\n<h1>Hi</h1>\n```\nenjoy\n```python\nprint(1)\n```";
        let blocks = extract_code_blocks(reply);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].lang, "html");
        assert_eq!(blocks[0].body, "<h1>Hi</h1>");
        assert_eq!(blocks[1].lang, "python");
        assert_eq!(blocks[1].body, "print(1)");
    }

    #[test]
    fn test_extract_code_blocks_ignores_unlabeled_and_unclosed() {
        assert!(extract_code_blocks("```\nplain\n```").is_empty());
        assert!(extract_code_blocks("```html\n<h1>unclosed").is_empty());
        assert!(extract_code_blocks("no code here").is_empty());
    }

    #[test]
    fn test_wants_auto_save() {
        assert!(wants_auto_save(
            "create a demo gym membership app in html and save the code files in lixo folder on desktop"
        ));
        assert!(wants_auto_save("write a python script"));
        assert!(!wants_auto_save("what is html"));
        assert!(!wants_auto_save("make a folder"));
        assert!(!wants_auto_save("tell me a joke"));
    }

    #[test]
    fn test_infer_file_name() {
        assert_eq!(infer_file_name("make an app", "html"), "index.html");
        assert_eq!(infer_file_name("make an app", "python"), "app.py");
        assert_eq!(
            infer_file_name("save it as named gym tracker on desktop", "html"),
            "gymtracker.html"
        );
        assert_eq!(
            infer_file_name("called report on desktop", "text"),
            "report.txt"
        );
    }

    #[test]
    fn test_infer_target_folder() {
        assert_eq!(
            infer_target_folder("save the files in lixo folder on desktop"),
            "desktop/lixo"
        );
        assert_eq!(infer_target_folder("create an app"), "desktop");
        assert_eq!(
            infer_target_folder("save it in the new folder"),
            "desktop"
        );
    }
}

#[cfg(test)]
mod stream_scrub_tests {
    use super::stream_scrub;

    /// Drive the scrubber piece-by-piece and reconstruct the full screen output.
    fn scrub_all(pieces: &[&str]) -> String {
        let mut held = String::new();
        let mut screen = String::new();
        for piece in pieces {
            let (head, new_held) = stream_scrub(&held, piece);
            held = new_held;
            screen.push_str(&head);
        }
        // Final flush mirrors the engine's end-of-stream behavior.
        screen.push_str(&held.replace("LAGi", "LAGI"));
        screen
    }

    #[test]
    fn test_stream_scrub_replaces_within_single_token() {
        assert_eq!(scrub_all(&["I am ", "LAGi", ", ready."]), "I am LAGI, ready.");
    }

    #[test]
    fn test_stream_scrub_replaces_across_token_boundary() {
        assert_eq!(scrub_all(&["I am LA", "Gi, ready."]), "I am LAGI, ready.");
        assert_eq!(scrub_all(&["LA", "G", "i is great"]), "LAGI is great");
    }

    #[test]
    fn test_stream_scrub_passes_clean_and_lowercase() {
        // Lowercase 'lagi' (container extension / CLI name) stays untouched.
        assert_eq!(scrub_all(&["open agent", ".lagi now"]), "open agent.lagi now");
        assert_eq!(scrub_all(&["LAGI is ready"]), "LAGI is ready");
    }
}

