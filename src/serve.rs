/*!
OpenAI-Compatible HTTP Server for LAGI

`lagi serve <container.lagi>` exposes a strict OpenAI-schema REST API so
standard clients (Cursor, LangChain, openai-python) work out of the box:

- `GET  /v1/models`                — list the resident model
- `POST /v1/chat/completions`      — chat, with full SSE streaming support

Synaptic Recall is wired into every request: before generation the user's
message is recalled against Liumi VectorCore (< 0.2 ms) and injected into the
system context; after generation new preferences are extracted and committed
asynchronously.

CORS is fully open (`Any`) so local web apps and browser tools can hit
localhost:8080 without preflight blocks. Ctrl+C triggers graceful shutdown:
in-flight requests drain, and model weights + `.lvc` state deallocate cleanly.
*/

use crate::manifest::ModelRuntimeConfig;
use crate::memory::MemoryGraph;
use crate::synaptic_recall::SynapticMemoryManager;
use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::StreamExt;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

pub const MODEL_ID: &str = "LAGI-Edge-1B";

/// Model backend abstraction over the llama.cpp engine / placeholder runner.
enum Backend {
    #[cfg(feature = "llama")]
    Llama(Arc<crate::chat::llama_engine::ChatEngine>),
    #[cfg_attr(feature = "llama", allow(dead_code))]
    Placeholder,
}

impl Backend {
    fn respond(&self, history: &[(String, String)], _temperature: f32) -> Result<String> {
        // .lbc codex output guard on every served completion.
        let out = match self {
            #[cfg(feature = "llama")]
            Backend::Llama(engine) => engine.respond(history)?,
            Backend::Placeholder => {
                let last = history
                    .iter()
                    .rev()
                    .find(|(role, _)| role == "user")
                    .map(|(_, c)| c.clone())
                    .unwrap_or_default();
                format!(
                    "[placeholder model] I received: \"{}\". Rebuild with `--features llama` for real inference.",
                    last.chars().take(80).collect::<String>()
                )
            }
        };
        Ok(crate::codex::guard_output(&out))
    }
}

/// Shared server state (Axum handlers run on a multi-threaded runtime).
pub struct ServerState {
    backend: Backend,
    synaptic: SynapticMemoryManager,
    tools: crate::tools::ToolDispatcher,
    model_cfg: ModelRuntimeConfig,
    #[allow(dead_code)]
    agent_name: String,
}

/// OpenAI chat-completions request (strict schema subset we consume).
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[allow(dead_code)]
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

fn default_temperature() -> f32 {
    0.7
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Build the Axum router with CORS fully open.
pub fn build_router(state: Arc<ServerState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(cors)
        .with_state(state)
}

/// GET /v1/models — OpenAI model listing.
async fn models_handler(State(state): State<Arc<ServerState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "liumi",
            "context_window": state.model_cfg.context_window,
        }]
    }))
}

/// POST /v1/chat/completions — OpenAI-compatible chat endpoint with SSE.
async fn chat_completions_handler(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // --- Synaptic Recall: inject recalled facts into the system context ---
    let recall_start = std::time::Instant::now();
    let last_user = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let recalled = state.synaptic.recall_context(&last_user);
    let recall_us = recall_start.elapsed().as_micros();
    if recall_us > 200 {
        info!("synaptic recall took {} us (over 0.2ms soft budget)", recall_us);
    }

    // Rebuild the message history with the recalled block merged into the
    // leading system message (or prepended as one if none exists).
    let mut history: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    if let Some(recall_block) = recalled {
        match history.first_mut() {
            Some((role, content)) if role == "system" => {
                content.push_str("\n\n");
                content.push_str(&recall_block);
            }
            _ => {
                let mut with_system =
                    vec![("system".to_string(), recall_block)];
                with_system.extend(history);
                history = with_system;
            }
        }
    } else if SynapticMemoryManager::is_factual_question(&last_user) {
        // Auto-Teach Loop: unknown factual query -> background web search,
        // distillation, and .lvc commit so future requests recall locally.
        let teach = state.synaptic.clone();
        let q = last_user.clone();
        tokio::spawn(async move {
            if let Ok(n) = teach.auto_teach(&q).await {
                if n > 0 {
                    info!("auto-taught {} new fact(s) into .lvc", n);
                }
            }
        });
    }

    // --- Native tool dispatch: web search / PC control / browser / clipboard.
    // Tool results are served directly without spending LLM tokens.
    if let Some(out) = state.tools.dispatch(&last_user).await {
        info!("tool executed: {}", out.tool);
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
        let created = chrono::Utc::now().timestamp();
        let model_id = req.model.clone().unwrap_or_else(|| MODEL_ID.to_string());
        return build_response(req.stream, id, created, model_id, &last_user, &out.text).await;
    }

    // --- Generation (blocking CPU work off the async runtime) ---
    let backend_temperature = req.temperature;
    let backend_state = state.clone();
    let gen_result = tokio::task::spawn_blocking(move || {
        backend_state
            .backend
            .respond(&history, backend_temperature)
    })
    .await;

    let reply = match gen_result {
        Ok(Ok(text)) => text.trim().to_string(),
        Ok(Err(e)) => {
            error!("inference error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": { "message": format!("inference error: {}", e), "type": "server_error" } })),
            )
                .into_response();
        }
        Err(e) => {
            error!("inference task panicked: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": { "message": "inference task failed", "type": "server_error" } })),
            )
                .into_response();
        }
    };

    // --- Asynchronous synaptic commit: extract new user preferences ---
    let synaptic = state.synaptic.clone();
    let user_msg = last_user.clone();
    tokio::spawn(async move {
        let committed = synaptic.extract_and_commit(&user_msg);
        if committed > 0 {
            tracing::debug!("synaptic commit: {} new preference(s)", committed);
        }
    });

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();
    let model_id = req.model.unwrap_or_else(|| MODEL_ID.to_string());
    build_response(req.stream, id, created, model_id, &last_user, &reply).await
}

/// Emit either a single JSON completion or an SSE token stream.
async fn build_response(
    stream: bool,
    id: String,
    created: i64,
    model_id: String,
    prompt: &str,
    reply: &str,
) -> Response {
    if !stream {
        // Non-streaming: single OpenAI chat.completion object.
        return Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": reply },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": estimate_tokens(prompt),
                "completion_tokens": estimate_tokens(reply),
                "total_tokens": estimate_tokens(prompt) + estimate_tokens(reply)
            }
        }))
        .into_response();
    }

    // Streaming: SSE `data: {...delta...}\n\n` chunks terminated by [DONE].
    let chunks = chunk_text(reply, 48);
    let id2 = id.clone();
    let model2 = model_id.clone();

    let stream = futures::stream::iter(chunks.into_iter().map(move |piece| {
        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
            "data: {}\n\n",
            json!({
                "id": id2,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model2,
                "choices": [{
                    "index": 0,
                    "delta": { "content": piece },
                    "finish_reason": null
                }]
            })
        )))
    }))
    .chain(futures::stream::once(async move {
        // Final chunk signals the stop reason, then [DONE] terminates the stream.
        let stop = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        });
        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
            "data: {}\n\ndata: [DONE]\n\n",
            stop
        )))
    }));

    axum::http::Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(axum::body::Body::from_stream(stream))
        .expect("static SSE response parts are always valid")
        .into_response()
}

/// Split text into stream-sized chunks on word boundaries where possible.
fn chunk_text(text: &str, size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for word in text.split_inclusive(' ') {
        current.push_str(word);
        if current.len() >= size {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

fn estimate_tokens(text: &str) -> u64 {
    (text.len() as f64 / 4.0).ceil() as u64
}

/// Launch the OpenAI-compatible server for a `.lagi` container.
///
/// Returns after graceful shutdown (Ctrl+C); the staged model temp file is
/// deleted before returning so weights deallocate from RAM and disk cleanly.
pub async fn run_server(
    file: &Path,
    context: Option<&str>,
    host: &str,
    port: u16,
    ram_arg: &str,
    threads_arg: &str,
    clean_memory: bool,
) -> Result<()> {
    info!("Loading container: {}", file.display());

    // Hardware auto-sense badge (zero flags required).
    let hw = crate::resolve_hardware_settings(ram_arg, threads_arg);
    println!("{}", crate::hardware_badge(&hw));
    let ctx_override = Some(hw.context_window as u32);
    let n_batch = hw.n_batch as u32;
    let unpackager = crate::LagiUnpackager::new(file);
    let container = unpackager.unpack(context.map(|s| s.as_bytes()))?;
    let manifest = container.manifest.clone();
    let model_cfg = crate::manifest::ModelRuntimeConfig {
        cpu_threads: hw.threads,
        ..manifest.runtime_model()
    };

    let memory_cfg = manifest.memory_settings();
    let memory_path = file
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&memory_cfg.db_name);
    let memory = Arc::new(MemoryGraph::new(&memory_path, None)?);

    // Single-file container persistence: hydrate .lvc from the trailing
    // state block of the host .lagi (if present).
    match crate::load_memory_from_container(file, &memory) {
        Ok(n) if n > 0 => info!("Restored {} .lvc memories from container state block", n),
        Ok(_) => {}
        Err(e) => info!("No container state block loaded: {}", e),
    }

    // --clean-memory: scrub stale/mock entries (e.g. old 'LAGi' spellings)
    // AFTER hydration so entries inside the container state block are
    // scrubbed too.
    if clean_memory {
        match memory.delete_matching("LAGi") {
            Ok(n) if n > 0 => info!("Cleaned memory: removed {} stale 'LAGi' entries", n),
            Ok(_) => info!("Cleaned memory: no stale entries found"),
            Err(e) => error!("Memory cleanup failed: {}", e),
        }
    }

    let synaptic = SynapticMemoryManager::new(memory.clone());
    let tools = crate::tools::ToolDispatcher::new(memory.clone());

    // Build the inference backend and keep the staged model path for cleanup.
    let (backend, staged_model): (Backend, Option<PathBuf>) = {
        #[cfg(feature = "llama")]
        {
            let model_bytes = container
                .get_model()
                .ok_or_else(|| anyhow::anyhow!("Model asset missing from container"))?;
            let (engine, staged) = crate::chat::llama_engine::engine_from_ram_with_overrides(
                model_bytes,
                &model_cfg,
                ctx_override,
                n_batch,
            )?;
            (Backend::Llama(Arc::new(engine)), Some(staged))
        }
        #[cfg(not(feature = "llama"))]
        {
            let _ = (&container, &ctx_override, &n_batch); // model asset unused without the llama backend
            info!("Placeholder inference (build with --features llama for real decoding)");
            (Backend::Placeholder, None)
        }
    };

    let state = Arc::new(ServerState {
        backend,
        synaptic,
        tools,
        model_cfg,
        agent_name: manifest.name.clone(),
    });

    let app = build_router(state);
    let addr: std::net::SocketAddr = format!("{}:{}", host, port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!(
        "Serving agent '{}' (model {}) on http://{}:{}",
        manifest.name,
        MODEL_ID,
        host,
        port
    );
    info!("  GET  /v1/models");
    info!("  POST /v1/chat/completions  (stream=true for SSE)");
    info!("Press Ctrl+C to stop.");

    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        info!("SIGINT received - draining requests and deallocating model state...");
    };

    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await;

    // Graceful memory cleanup: remove the staged GGUF so weights deallocate
    // from both RAM and disk when the process exits, and flush the dirty
    // .lvc state into the .lagi trailing state block (payload untouched).
    if let Some(path) = staged_model {
        let _ = std::fs::remove_file(&path);
    }
    match crate::flush_memory_to_container(&memory, file) {
        Ok(bytes) => info!(".lvc state flushed into container ({} bytes)", bytes),
        Err(e) => error!("state flush failed: {}", e),
    }
    result?;
    info!("Server stopped cleanly. VectorCore state flushed.");
    Ok(())
}


