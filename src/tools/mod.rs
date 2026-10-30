pub mod pc_control;
pub mod web_search;

use crate::memory::{MemoryGraph, MemoryType};
use crate::tools::pc_control::PcToolExecutor;
use crate::tools::web_search::{read_webpage, search_web};
use std::sync::Arc;

/**
Natural-language tool dispatcher.

LAGI-Edge-1B is a tiny model — it cannot reliably emit strict JSON tool
calls. Instead the host runtime performs lightweight intent detection on the
raw user utterance and executes matching native tools directly, feeding the
observation back as the assistant reply and committing an episodic record to
Liumi VectorCore. Non-tool utterances return `None` and fall through to the
LLM unchanged.
*/
pub struct ToolDispatcher {
    pc: PcToolExecutor,
    memory: Arc<MemoryGraph>,
    /// Optional proactive timer scheduler; when present, natural timer
    /// intents ("say hello to me after 1 minute") are registered here.
    scheduler: Option<Arc<crate::scheduler::TaskScheduler>>,
}

/// Result of a dispatched tool execution
pub struct ToolOutput {
    pub tool: &'static str,
    pub text: String,
}

impl ToolDispatcher {
    pub fn new(memory: Arc<MemoryGraph>) -> Self {
        Self {
            pc: PcToolExecutor::new(),
            memory,
            scheduler: None,
        }
    }

    /// Create a dispatcher wired to the proactive timer scheduler so
    /// time-delayed intents ("remind me to X in 5 minutes") can be handled.
    pub fn with_scheduler(
        memory: Arc<MemoryGraph>,
        scheduler: Arc<crate::scheduler::TaskScheduler>,
    ) -> Self {
        Self {
            pc: PcToolExecutor::new(),
            memory,
            scheduler: Some(scheduler),
        }
    }

    // -----------------------------------------------------------------------
    // Proactive timer scheduling
    // -----------------------------------------------------------------------

    /// Detect a natural timer intent and register it with the scheduler.
    /// Returns the immediate confirmation output, or `None` when the input
    /// is not a timer request (or no scheduler is wired).
    fn try_schedule_timer(&self, lower: &str) -> Option<ToolOutput> {
        let scheduler = self.scheduler.as_ref()?;
        let (message, duration_secs) = parse_timer_intent(lower)?;
        let duration = std::time::Duration::from_secs(duration_secs);
        let duration_display = format_duration(duration_secs);

        scheduler.schedule_task(
            duration,
            message.clone(),
            crate::scheduler::TaskType::Reminder,
        );
        self.commit(
            "timer",
            &format!("scheduled reminder in {}: {}", duration_display, message),
        );
        Some(ToolOutput {
            tool: "timer",
            text: format!(
                "⏰ Scheduled: I will remind you: \"{}\" in {}.",
                message, duration_display
            ),
        })
    }

    /// Commit an episodic record of this tool action to VectorCore.
    fn commit(&self, tool: &str, summary: &str) {
        let entry = crate::memory::MemoryEntry::new(
            MemoryType::Activity,
            format!("[tool:{}] {}", tool, summary),
            None,
            90,
            vec!["tool".to_string(), tool.to_string()],
        );
        if let Err(err) = self.memory.store(&entry) {
            tracing::debug!("tool memory commit failed (non-fatal): {}", err);
        }
    }

    /// Detect and execute a tool for a user utterance. `None` = not a tool.
    pub async fn dispatch(&self, input: &str) -> Option<ToolOutput> {
        let lower = input.to_lowercase();
        let trimmed = input.trim();

        // --- Universal delayed tool execution (MUST run first) ---
        // "create a folder on desktop after 10 seconds and name it exp9",
        // "open youtube in 5 seconds": extract the time clause, strip it to
        // get the clean action, and if that action is a tool, schedule it
        // for deferred execution instead of running it now.
        if let Some((duration_secs, clean_action)) = extract_delayed_action(input) {
            let clean_lower = clean_action.to_lowercase();
            if let Some(scheduler) = self.scheduler.as_ref() {
                if is_tool_intent(&clean_lower) {
                    scheduler.schedule_task(
                        std::time::Duration::from_secs(duration_secs),
                        clean_action.clone(),
                        crate::scheduler::TaskType::Tool(clean_action.clone()),
                    );
                    let duration_display = format_duration(duration_secs);
                    self.commit(
                        "delayed_tool",
                        &format!(
                            "scheduled tool in {}: {}",
                            duration_display, clean_action
                        ),
                    );
                    return Some(ToolOutput {
                        tool: "delayed_tool",
                        text: format!(
                            "⏰ Scheduled Action: Will execute \"{}\" in {}.",
                            clean_action, duration_display
                        ),
                    });
                }
            }
        }

        // --- Folder creation (natural intent, before other folder intents) ---
        if let Some((location, name)) = parse_folder_creation(&lower) {
            return Some(self.run_create_folder(&location, &name));
        }

        // --- File creation / writing (natural intent, before other file
        //     intents; "create a txt file in lixo folder and write hello
        //     world", "write hello world into file", "make a file ...") ---
        if let Some((folder, file_name, content)) = parse_file_creation(&lower) {
            return Some(self.run_write_file(&folder, &file_name, &content));
        }

        // --- Proactive timers ("say hello to me after 1 minute",
        //     "remind me to stretch in 30 seconds", "in 2 hours tell me
        //     to check the oven") — instant confirmation, non-blocking ---
        if let Some(out) = self.try_schedule_timer(&lower) {
            return Some(out);
        }

        // --- Web search: explicit prefixes ("search the web for X", "google X") ---
        for prefix in [
            "search the web for ",
            "search web for ",
            "web search ",
            "search for ",
            "google ",
        ] {
            if let Some(query) = lower.strip_prefix(prefix) {
                let query = trimmed[trimmed.len() - query.len()..].trim();
                return Some(self.run_web_search(query).await);
            }
        }

        // --- Web search: natural intent anywhere ("can you get information
        //     about X from websearch", "look up X online", ...) ---
        if let Some(query) = extract_web_search_query(&lower) {
            return Some(self.run_web_search(&query).await);
        }

        // --- Open URL in default browser ---
        if let Some(url) = extract_url(trimmed) {
            if ["open ", "launch ", "browse to ", "go to "]
                .iter()
                .any(|p| lower.starts_with(p))
            {
                return Some(self.run_open_browser(url));
            }
        }

        // --- Smart browser opener: expand naked sites into full https URLs ---
        if let Some(url) = extract_open_site_target(&lower) {
            return Some(self.run_open_browser(&url));
        }

        // --- Organize a folder ---
        if lower.contains("organize") || lower.contains("organise") || lower.contains("tidy") {
            let folder = detect_folder(&lower);
            return Some(self.run_organize(folder).await);
        }

        // --- Running apps / processes ---
        if lower.contains("running") && (lower.contains("app") || lower.contains("process") || lower.contains("what"))
            || lower.contains("task manager")
            || lower.contains("processes")
        {
            return Some(self.run_running_apps().await);
        }

        // --- List files ---
        if (lower.starts_with("list ") || lower.starts_with("show "))
            && (lower.contains("file") || lower.contains("folder") || lower.contains("directory"))
        {
            let folder = detect_folder(&lower);
            return Some(self.run_list_files(folder).await);
        }

        // --- Read a local file ---
        if lower.starts_with("read file ") || lower.starts_with("read the file ") || lower.starts_with("open file ")
        {
            let path = trimmed.splitn(3, ' ').last().unwrap_or_default().trim().to_string();
            if !path.is_empty() && !path.starts_with("http") {
                return Some(self.run_read_file(&path));
            }
        }

        // --- Clipboard ---
        if lower.contains("clipboard") {
            let wants_write = lower.contains("write") || lower.contains("copy") || lower.contains("set");
            let rest = trimmed.split_once("clipboard");
            if wants_write {
                if let Some((_, payload_raw)) = rest {
                    let payload =
                        payload_raw.trim_start_matches([':', ' ', '"']).trim_matches('"').trim();
                    if !payload.is_empty() {
                        return Some(self.run_clipboard_write(payload));
                    }
                }
            }
            return Some(self.run_clipboard_read().await);
        }

        // --- Read a webpage by URL ---
        if let Some(url) = extract_url(trimmed) {
            if ["read ", "fetch ", "summarize "].iter().any(|p| lower.starts_with(p)) {
                return Some(self.run_read_webpage(url).await);
            }
        }

        None
    }

    async fn run_web_search(&self, query: &str) -> ToolOutput {
        match search_web(query).await {
            Ok(snippets) => {
                let mut text = format!("Web results for \"{}\":\n", query);
                for (i, s) in snippets.iter().enumerate() {
                    text.push_str(&format!("\n{}. {}\n   {}\n   {}\n", i + 1, s.title, s.url, s.snippet));
                }
                self.commit("web_search", &format!("searched: {}", query));
                ToolOutput { tool: "web_search", text }
            }
            Err(err) => ToolOutput { tool: "web_search", text: format!("Web search failed: {}", err) },
        }
    }

    async fn run_read_webpage(&self, url: &str) -> ToolOutput {
        match read_webpage(url).await {
            Ok(markdown) => {
                let preview: String = markdown.chars().take(4000).collect();
                self.commit("read_webpage", &format!("read: {}", url));
                ToolOutput { tool: "read_webpage", text: preview }
            }
            Err(err) => ToolOutput {
                tool: "read_webpage",
                text: format!("Could not read {}: {}", url, err),
            },
        }
    }

    fn run_open_browser(&self, url: &str) -> ToolOutput {
        match self.pc.open_browser_url(url) {
            Ok(()) => {
                self.commit("open_browser", &format!("opened: {}", url));
                ToolOutput { tool: "open_browser", text: format!("Opened {} in your default browser.", url) }
            }
            Err(err) => ToolOutput { tool: "open_browser", text: format!("Failed to open browser: {}", err) },
        }
    }

    fn run_create_folder(&self, location: &str, name: &str) -> ToolOutput {
        match self.pc.create_folder(location, name) {
            Ok(msg) => {
                self.commit("create_folder", &format!("created {} in {}", name, location));
                ToolOutput { tool: "create_folder", text: msg }
            }
            Err(err) => ToolOutput {
                tool: "create_folder",
                text: format!("Could not create folder: {}", err),
            },
        }
    }

    /// Write a text file under a safe root (optionally with subfolders).
    fn run_write_file(&self, target_folder: &str, file_name: &str, content: &str) -> ToolOutput {
        match self.pc.write_file(target_folder, file_name, content) {
            Ok(msg) => {
                self.commit("write_file", &msg);
                ToolOutput { tool: "write_file", text: msg }
            }
            Err(err) => ToolOutput {
                tool: "write_file",
                text: format!("Could not write file: {}", err),
            },
        }
    }

    async fn run_organize(&self, folder: &str) -> ToolOutput {
        match self.pc.organize_folder(folder).await {
            Ok(report) => {
                self.commit("organize_folder", &report.summary);
                ToolOutput { tool: "organize_folder", text: report.summary }
            }
            Err(err) => ToolOutput {
                tool: "organize_folder",
                text: format!("Could not organize '{}': {}", folder, err),
            },
        }
    }

    async fn run_list_files(&self, folder: &str) -> ToolOutput {
        match self.pc.list_files(folder).await {
            Ok(entries) => {
                let listing = if entries.is_empty() {
                    "(empty folder)".to_string()
                } else {
                    entries
                        .iter()
                        .take(40)
                        .map(|e| format!("  {} {}", if e.is_dir { "[dir] " } else { "      " }, e.name))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                self.commit("list_files", &format!("listed: {}", folder));
                ToolOutput {
                    tool: "list_files",
                    text: format!("Contents of {}:\n{}", folder, listing),
                }
            }
            Err(err) => ToolOutput {
                tool: "list_files",
                text: format!("Could not list '{}': {}", folder, err),
            },
        }
    }

    async fn run_running_apps(&self) -> ToolOutput {
        match self.pc.get_running_apps() {
            Ok(apps) => {
                self.commit("get_running_apps", &format!("{} processes", apps.len()));
                let list: String = apps
                    .iter()
                    .take(30)
                    .map(|p| format!("  {:<28} pid {:>7}  {:>7.1} MB", p.name, p.pid, p.memory_mb))
                    .collect::<Vec<_>>()
                    .join("\n");
                ToolOutput {
                    tool: "get_running_apps",
                    text: format!("Running processes ({} total):\n{}", apps.len(), list),
                }
            }
            Err(err) => ToolOutput {
                tool: "get_running_apps",
                text: format!("Could not inspect processes: {}", err),
            },
        }
    }

    fn run_read_file(&self, path: &str) -> ToolOutput {
        match self.pc.read_local_file(path) {
            Ok(content) => {
                let preview: String = content.chars().take(4000).collect();
                self.commit("read_local_file", &format!("read: {}", path));
                ToolOutput { tool: "read_local_file", text: preview }
            }
            Err(err) => ToolOutput {
                tool: "read_local_file",
                text: format!("Could not read '{}': {}", path, err),
            },
        }
    }

    async fn run_clipboard_read(&self) -> ToolOutput {
        match self.pc.read_clipboard().await {
            Ok(content) => {
                self.commit("read_clipboard", "read clipboard");
                ToolOutput {
                    tool: "read_clipboard",
                    text: format!("Clipboard:\n{}", content),
                }
            }
            Err(err) => ToolOutput {
                tool: "read_clipboard",
                text: format!("Could not read clipboard: {}", err),
            },
        }
    }

    fn run_clipboard_write(&self, payload: &str) -> ToolOutput {
        match self.pc.write_clipboard(payload) {
            Ok(()) => {
                self.commit("write_clipboard", "wrote clipboard");
                ToolOutput { tool: "write_clipboard", text: "Clipboard updated.".to_string() }
            }
            Err(err) => ToolOutput {
                tool: "write_clipboard",
                text: format!("Could not write clipboard: {}", err),
            },
        }
    }
}

/// Extract the first http(s) URL from an utterance
// ---------------------------------------------------------------------------
// Natural-language timer intent parsing
// ---------------------------------------------------------------------------

/// Regex for "say hello to me after 1 minute" / "remind me to stretch in 30 seconds".
static TIMER_MSG_FIRST: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
/// Regex for "in 5 minutes tell me to drink water" / "after 2 hours remind me to X".
static TIMER_MSG_LAST: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();

fn timer_regex(kind: u8) -> &'static regex::Regex {
    let (cell, pattern): (&std::sync::OnceLock<regex::Regex>, &str) = match kind {
        0 => (
            &TIMER_MSG_FIRST,
            r"(?:please\s+)?(?:say|tell me|remind me to)\s+(?P<msg>.+?)\s+(?:after|in)\s+(?P<num>\d+)\s*(?P<unit>second|sec|minute|min|hour|hr)s?\b",
        ),
        _ => (
            &TIMER_MSG_LAST,
            r"(?:please\s+)?(?:in|after)\s+(?P<num>\d+)\s*(?P<unit>second|sec|minute|min|hour|hr)s?\b\s+(?:say|tell me|remind me to)\s+(?P<msg>.+)",
        ),
    };
    cell.get_or_init(|| regex::RegexBuilder::new(pattern).case_insensitive(true).build().expect("valid timer regex"))
}

/// Parse a proactive timer request into `(message, duration_secs)`.
/// Supports both phrasings:
///   - "(say|tell me|remind me to) <msg> (after|in) <n> <unit>s?"
///   - "(in|after) <n> <unit>s? (say|tell me|remind me to) <msg>"
pub fn parse_timer_intent(lower: &str) -> Option<(String, u64)> {
    let caps = timer_regex(0)
        .captures(lower)
        .or_else(|| timer_regex(1).captures(lower))?;

    let n: u64 = caps.name("num")?.as_str().parse().ok()?;
    if n == 0 {
        return None;
    }
    let secs = match caps.name("unit")?.as_str() {
        "second" | "sec" => n,
        "minute" | "min" => n * 60,
        "hour" | "hr" => n * 3600,
        _ => return None,
    };
    let message = caps
        .name("msg")?
        .as_str()
        .trim()
        // "tell me to X" may capture the "to" into the message; strip it.
        .trim_start_matches("to ")
        .trim()
        .trim_end_matches(['.', '!', '?'])
        .to_string();
    if message.is_empty() {
        return None;
    }
    Some((message, secs))
}

/// Human-friendly duration string: `1 minute`, `90 seconds`, `2 hours`.
pub fn format_duration(total_secs: u64) -> String {
    let (value, unit) = if total_secs % 3600 == 0 && total_secs >= 3600 {
        (total_secs / 3600, "hour")
    } else if total_secs % 60 == 0 && total_secs >= 60 {
        (total_secs / 60, "minute")
    } else {
        (total_secs, "second")
    };
    if value == 1 {
        format!("1 {unit}")
    } else {
        format!("{value} {unit}s")
    }
}

// ---------------------------------------------------------------------------
// Universal delayed tool execution parsing
// ---------------------------------------------------------------------------

/// Regex matching a delay clause anywhere in an utterance:
/// "after 10 seconds", "in 5 minutes", "in 2 hrs", ...
static DELAY_CLAUSE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();

fn delay_clause_regex() -> &'static regex::Regex {
    DELAY_CLAUSE.get_or_init(|| {
        regex::RegexBuilder::new(
            r"(?P<after>after|in)\s+(?P<num>\d+)\s*(?P<unit>second|sec|minute|min|hour|hr)s?\b",
        )
        .case_insensitive(true)
        .build()
        .expect("valid delay clause regex")
    })
}

/// Extract a deferred-action delay clause from a raw utterance.
///
/// Returns `(duration_secs, clean_action)` where `clean_action` is the
/// original utterance with the time clause removed (whitespace collapsed,
/// case preserved). Returns `None` when no delay clause is present.
///
/// ```text
/// "create a folder on desktop after 10 seconds and name it exp9"
///     -> (10, "create a folder on desktop and name it exp9")
/// "in 5 seconds open youtube"
///     -> (5, "open youtube")
/// ```
pub fn extract_delayed_action(input: &str) -> Option<(u64, String)> {
    let caps = delay_clause_regex().captures(input)?;
    let n: u64 = caps.name("num")?.as_str().parse().ok()?;
    if n == 0 {
        return None;
    }
    let secs = match caps.name("unit")?.as_str().to_lowercase().as_str() {
        "second" | "sec" => n,
        "minute" | "min" => n * 60,
        "hour" | "hr" => n * 3600,
        _ => return None,
    };
    let whole = caps.get(0)?;
    let mut clean = format!("{} {}", &input[..whole.start()], &input[whole.end()..]);
    // Collapse runs of whitespace introduced by the strip.
    clean = clean.split_whitespace().collect::<Vec<_>>().join(" ");
    let clean = clean.trim().trim_end_matches(['.', '!', '?']).trim().to_string();
    if clean.is_empty() {
        return None;
    }
    Some((secs, clean))
}

/// Lightweight intent check mirroring the main `dispatch()` arms: returns
/// `true` when the (delay-stripped) utterance would be handled by a native
/// tool rather than the LLM. Used to decide whether a delayed action is a
/// schedulable tool invocation or a plain reminder.
fn is_tool_intent(lower: &str) -> bool {
    // Folder / file creation.
    if parse_folder_creation(lower).is_some() || parse_file_creation(lower).is_some() {
        return true;
    }
    // Browser: naked sites, TLDs, domains, and explicit URLs.
    if extract_open_site_target(lower).is_some() {
        return true;
    }
    if extract_url(lower).is_some()
        && ["open ", "launch ", "browse to ", "go to ", "browse ", "read ", "fetch ", "summarize "]
            .iter()
            .any(|p| lower.starts_with(p))
    {
        return true;
    }
    // Web search: explicit prefixes only (natural-language search phrasing
    // is too ambiguous to schedule reliably).
    for prefix in [
        "search the web for ",
        "search web for ",
        "web search ",
        "search for ",
        "google ",
    ] {
        if lower.starts_with(prefix) && lower.len() > prefix.len() {
            return true;
        }
    }
    // Folder organization.
    if lower.contains("organize") || lower.contains("organise") || lower.contains("tidy") {
        return true;
    }
    // Local file reading.
    if lower.starts_with("read file ")
        || lower.starts_with("read the file ")
        || lower.starts_with("open file ")
    {
        return true;
    }
    // List files.
    if (lower.starts_with("list ") || lower.starts_with("show "))
        && (lower.contains("file") || lower.contains("folder") || lower.contains("directory"))
    {
        return true;
    }
    // Clipboard write.
    if lower.contains("clipboard")
        && (lower.contains("write") || lower.contains("copy") || lower.contains("set"))
    {
        return true;
    }
    false
}

/// Extract the first http(s) URL from an utterance
fn extract_url(input: &str) -> Option<&str> {
    input
        .split_whitespace()
        .find(|tok| tok.starts_with("http://") || tok.starts_with("https://"))
}

/// Well-known naked site names -> canonical https entry points.
const SITE_MAP: &[(&str, &str)] = &[
    ("youtube", "https://www.youtube.com"),
    ("google", "https://www.google.com"),
    ("github", "https://github.com"),
    ("chrome", "https://www.google.com"),
    ("wikipedia", "https://www.wikipedia.org"),
    ("reddit", "https://www.reddit.com"),
    ("stackoverflow", "https://stackoverflow.com"),
    ("gmail", "https://mail.google.com"),
    ("twitter", "https://x.com"),
    ("x", "https://x.com"),
];

/// Smart browser opener: for "open|launch|browse|go to <target>" where
/// <target> is a well-known naked site ("open youtube", "open github"),
/// a site TLD ("open youtube.com"), or a bare domain ("open liumi.cloud"),
/// return the expanded https URL. Returns None for "open file ..." and for
/// targets that are neither mapped nor domain-like.
fn extract_open_site_target(lower: &str) -> Option<String> {
    const VERBS: &[&str] = &["open ", "launch ", "browse to ", "go to ", "browse "];
    let verb = VERBS.iter().find(|p| lower.starts_with(**p))?;
    let rest = &lower[verb.len()..];

    // "open file X" / "open the file X" is the local-file reader, not a site.
    let first_word = rest.split_whitespace().next().unwrap_or_default();
    if first_word == "file"
        || (first_word == "the" && rest.trim_start_matches("the ").starts_with("file"))
    {
        return None;
    }

    for tok in rest.split_whitespace() {
        let tok = tok.trim_matches(['.', ',', '!', '?', '"', '\'']);
        if tok.is_empty() {
            continue;
        }
        // Exact naked-site match ("open youtube", "open chrome").
        if let Some((_, url)) = SITE_MAP.iter().find(|(name, _)| *name == tok) {
            return Some((*url).to_string());
        }
        // Site with TLD ("open youtube.com") or bare domain ("open liumi.cloud").
        if tok.contains('.')
            && !tok.starts_with('.')
            && !tok.ends_with('.')
            && tok.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
            && tok.contains(|c: char| c.is_ascii_alphabetic())
        {
            return Some(format!("https://{}", tok));
        }
        // Only consider the first meaningful token.
        break;
    }
    None
}

/// Map natural-language folder hints to concrete directories
fn detect_folder(lower: &str) -> &'static str {
    if lower.contains("screenshot") {
        "screenshots"
    } else if lower.contains("download") {
        "downloads"
    } else if lower.contains("desktop") {
        "desktop"
    } else if lower.contains("document") {
        "documents"
    } else {
        "home"
    }
}

/// Filler phrases stripped (case-insensitively, iteratively) from a
/// natural-language web-search target. Multi-word phrases first.
const SEARCH_FILLERS: &[&str] = &[
    "from websearch",
    "from web search",
    "from the web",
    "on the web",
    "on the internet",
    "in the web",
    "get information about",
    "get information on",
    "find information about",
    "find information on",
    "information about",
    "information on",
    "can you",
    "could you",
    "would you",
    "please",
    "for me",
    "i want",
    "i need",
    "look up",
    "search online",
    "search the web for",
    "search the web",
    "search web for",
    "websearch",
    "web search",
    "online",
    "about",
    "regarding",
];

/// Clean a raw natural-language query target: iteratively strip filler
/// phrases, drop dangling connector tokens ("from", "for", "on", "up",
/// "the"), collapse whitespace, and trim quotes + punctuation.
fn sanitize_query(raw: &str) -> Option<String> {
    let mut text = raw.to_string();
    // Iteratively remove filler phrases until stable.
    loop {
        let mut changed = false;
        let low = text.to_lowercase();
        for filler in SEARCH_FILLERS {
            if let Some(pos) = low.find(filler) {
                let end = (pos + filler.len()).min(text.len());
                text = format!("{} {}", &text[..pos], &text[end..]);
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }

    // Trim quotes, punctuation, and dangling connector words.
    let mut tokens: Vec<&str> = text.split_whitespace().collect();
    while let Some(first) = tokens.first() {
        if matches!(
            first.trim_end_matches(['.', ',', '!', '?']),
            "from" | "for" | "on" | "up" | "the" | "of" | "about"
        ) {
            tokens.remove(0);
        } else {
            break;
        }
    }
    while let Some(last) = tokens.last() {
        if matches!(
            last.trim_start_matches(['.', ',', '!', '?']),
            "from" | "for" | "on" | "up" | "the" | "please" | "now"
        ) {
            tokens.pop();
        } else {
            break;
        }
    }

    // Repeatedly trim quotes, punctuation, and whitespace from both ends.
    let mut query = tokens.join(" ");
    loop {
        let before = query.clone();
        query = query
            .trim_start_matches(['"', '\'', '`', ' ', '\t', '?', '.', '!', ','])
            .trim_end_matches(['"', '\'', '`', ' ', '\t', '?', '.', '!', ','])
            .to_string();
        if query == before {
            break;
        }
    }
    let query = query.trim().to_string();

    // A bare connector or empty extraction is not a real query.
    if query.is_empty() || matches!(query.as_str(), "from" | "for" | "about" | "the") {
        None
    } else {
        Some(query)
    }
}

/// Smart web-search intent detection ANYWHERE in the utterance:
/// "can you get information about X from websearch", "look up X online",
/// "search the web for X", "search X online", ...
///
/// Informational questions ABOUT search itself ("how does web search work?")
/// are deliberately excluded so they fall through to the LLM.
fn extract_web_search_query(lower: &str) -> Option<String> {
    const INTENTS: &[&str] = &[
        "websearch",
        "web search",
        "search the web",
        "search online",
        "search about",
        "look up",
        "look this up",
        "get information",
        "find information",
    ];

    // Guard: questions asking about search itself are not search commands.
    const QUESTION_PREFIXES: &[&str] = &["how ", "what ", "why ", "who ", "when ", "where "];
    if QUESTION_PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return None;
    }

    // Broad fallback: a standalone "search"/"find" verb combined with a
    // web marker ("online", "web", "internet") anywhere in the utterance —
    // covers "search about X online", "can you search about X",
    // "find X online". Word-boundary checked so "research" doesn't fire.
    fn has_verb_token(text: &str, verb: &str) -> bool {
        text.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|w| w == verb)
    }
    let broad_intent: Option<&str> = if INTENTS.iter().any(|p| lower.contains(p)) {
        None // a specific intent matched below anyway
    } else {
        let has_web_marker =
            lower.contains("online") || lower.contains("web") || lower.contains("internet");
        if has_web_marker && has_verb_token(lower, "search") {
            Some("search")
        } else if has_web_marker && has_verb_token(lower, "find") {
            Some("find")
        } else {
            None
        }
    };

    // Word-boundary-aware intent lookup: "research online" must NOT count
    // as "search online". Returns (byte_pos, matched_intent).
    fn find_word_boundary<'a>(lower: &'a str, intent: &'a str) -> Option<(usize, &'a str)> {
        let mut from = 0usize;
        while let Some(rel) = lower[from..].find(intent) {
            let pos = from + rel;
            let before_ok = pos == 0 || !lower.as_bytes()[pos - 1].is_ascii_alphanumeric();
            let end = pos + intent.len();
            let after_ok = end >= lower.len() || !lower.as_bytes()[end].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return Some((pos, intent));
            }
            from = pos + 1;
        }
        None
    }

    let intent: &str = INTENTS
        .iter()
        .find_map(|p| find_word_boundary(lower, p))
        .map(|(_pos, matched)| matched)
        .or(broad_intent)?;
    let (idx, _) = find_word_boundary(lower, intent)?;
    let before = &lower[..idx];
    let after = &lower[idx + intent.len()..];

    // Prefer the text after the intent; fall back to the text before it
    // ("... about X from websearch" puts the target before the keyword).
    let after_trim = after.trim_start();
    let candidate = if after_trim.is_empty()
        || after_trim.starts_with("from")
        || after_trim.starts_with("online")
        || after_trim.starts_with("on the")
    {
        before
    } else {
        after
    };

    sanitize_query(candidate)
}

/// Folder-creation intent: "create (a) folder ...", "make (a) folder ...",
/// "new folder ..." with location hints ("on desktop", "in documents") and
/// name hints ("named lixo", "name it lixo", "called lixo").
/// Returns (location, name).
fn parse_folder_creation(lower: &str) -> Option<(String, String)> {
    const TRIGGERS: &[&str] = &[
        "create a folder",
        "create an folder",
        "create folder",
        "make a folder",
        "make folder",
        "new folder",
        "mkdir",
    ];
    if !TRIGGERS.iter().any(|t| lower.contains(t)) {
        return None;
    }

    // Location: reuse the natural-language folder hint mapping, restricted
    // to the safe create_folder whitelist (screenshots is read-only here).
    let location = match detect_folder(lower) {
        "desktop" | "downloads" | "documents" => detect_folder(lower).to_string(),
        _ => "desktop".to_string(), // safe default target
    };

    // Name: "named X", "name it X", "called X", "named as X".
    const NAME_MARKERS: &[&str] = &["named as ", "named ", "name it ", "name is ", "called "];
    for marker in NAME_MARKERS {
        if let Some(pos) = lower.find(marker) {
            let mut rest = &lower[pos + marker.len()..];
            // Cut the name at the next location/filler connector.
            for cut in [" on ", " in ", " under ", " inside ", " and ", ", "] {
                if let Some(cpos) = rest.find(cut) {
                    rest = &rest[..cpos];
                }
            }
            if let Ok(name) = pc_control::sanitize_folder_name(rest) {
                return Some((location, name));
            }
        }
    }
    None
}

/// Word-boundary token check.
fn has_token(lower: &str, tok: &str) -> bool {
    lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|w| w == tok)
}

/// File-creation/writing intent: "create (a) (txt |text )?file ...",
/// "make (a) file ...", "write X into (the )?file", "new file ...",
/// with folder hints ("in lixo folder", "on desktop") and content hints
/// ("write hello world", "with content X").
/// Returns (target_folder, file_name, content).
fn parse_file_creation(lower: &str) -> Option<(String, String, String)> {
    // Questions about files are not write commands.
    if ["how ", "what ", "why ", "who ", "when ", "where "]
        .iter()
        .any(|p| lower.starts_with(p))
    {
        return None;
    }
    if !has_token(lower, "file") && !has_token(lower, "files") {
        return None;
    }
    let _verb = ["create", "make", "write", "new"]
        .iter()
        .find(|v| has_token(lower, v))?;

    // Generative prompts ("create a demo app in html ...") must fall through
    // to Qwen so the model generates the complete implementation, instead of
    // the dispatcher writing a placeholder note file.
    const GENERATIVE_TOKENS: &[&str] = &[
        "app", "html", "website", "script", "code", "demo", "program",
    ];
    if GENERATIVE_TOKENS.iter().any(|t| has_token(lower, t)) {
        return None;
    }

    // --- Content extraction ---
    let mut content = String::new();
    // "write X into (the )?file" / "to file": content precedes the marker.
    const INTO_MARKERS: &[&str] = &[
        "into the file",
        "into file",
        "in the file",
        "in file",
        "to the file",
        "to file",
    ];
    for marker in INTO_MARKERS {
        if let Some(pos) = lower.find(marker) {
            content = lower[..pos].trim().to_string();
            break;
        }
    }
    if content.is_empty() {
        // "... and write hello world" / "write a file about rust"
        if let Some(pos) = lower.find("write ") {
            let mut rest = lower[pos + "write ".len()..].trim();
            // "write a txt file about rust": drop the file phrase itself.
            if let Some(fpos) = rest.find(" file") {
                let prefix_words: Vec<&str> = rest[..fpos].split_whitespace().collect();
                if prefix_words
                    .iter()
                    .all(|w| matches!(*w, "a" | "the" | "txt" | "text" | "new"))
                {
                    rest = rest[fpos + " file".len()..].trim();
                }
            }
            if let Some(stripped) = rest.strip_prefix("and ") {
                rest = stripped.trim();
            }
            content = rest.to_string();
        }
    }
    // Clean up surrounding noise from the extracted content.
    loop {
        let before = content.clone();
        content = content
            .trim()
            .trim_start_matches(['"', '\'', '`'])
            .trim_end_matches(['"', '\'', '`', '.', ',', '!', '?'])
            .trim()
            .to_string();
        for lead in ["and ", "then ", "please ", "write ", "create ", "make "] {
            if let Some(stripped) = content.strip_prefix(lead) {
                content = stripped.trim().to_string();
            }
        }
        if content == before {
            break;
        }
    }

    // --- File name extraction ---
    let mut file_name = String::new();
    const NAME_MARKERS: &[&str] = &["named as ", "named ", "name it ", "name is ", "called "];
    for marker in NAME_MARKERS {
        if let Some(pos) = lower.find(marker) {
            let mut rest = &lower[pos + marker.len()..];
            for cut in [
                " on ", " in ", " under ", " inside ", " and ", ", ", " with ", " containing ",
                " write ",
            ] {
                if let Some(cpos) = rest.find(cut) {
                    rest = &rest[..cpos];
                }
            }
            if let Ok(name) = pc_control::sanitize_folder_name(rest) {
                file_name = if name.contains('.') {
                    name
                } else {
                    format!("{}.txt", name)
                };
                break;
            }
        }
    }
    if file_name.is_empty() {
        // Default: derive the stem from the first content word, else "note".
        let first = content.split_whitespace().next().unwrap_or("note");
        let stem: String = first.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        let stem = if stem.is_empty() { "note".to_string() } else { stem };
        file_name = format!("{}.txt", stem);
    }

    // --- Folder extraction ---
    let root = match detect_folder(lower) {
        "downloads" => "downloads",
        "documents" => "documents",
        _ => "desktop", // safe default target
    };
    let mut target = root.to_string();
    if let Some(pos) = lower.find(" folder") {
        let before = lower[..pos].trim_end();
        let word = before.split_whitespace().last().unwrap_or("");
        let skip = matches!(
            word,
            "the" | "a" | "this" | "that" | "my" | "in" | "on" | "under" | "inside" | "new"
        ) || matches!(detect_folder(word), "desktop" | "downloads" | "documents" | "screenshots");
        if !word.is_empty() && !skip {
            if let Ok(sub) = pc_control::sanitize_folder_name(word) {
                target = format!("{}/{}", root, sub);
            }
        }
    }

    Some((target, file_name, content))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_url() {
        assert_eq!(extract_url("open https://liumi.cloud now"), Some("https://liumi.cloud"));
        assert_eq!(extract_url("hello"), None);
    }

    #[test]
    fn test_detect_folder() {
        assert_eq!(detect_folder("organize my desktop screenshots"), "screenshots");
        assert_eq!(detect_folder("tidy downloads"), "downloads");
    }

    #[test]
    fn test_smart_web_search_extraction() {
        assert_eq!(
            extract_web_search_query("can you get information about m.hasnain sabqi from websearch").as_deref(),
            Some("m.hasnain sabqi")
        );
        assert_eq!(
            extract_web_search_query("can you search about airbus a380 price online").as_deref(),
            Some("airbus a380 price")
        );
        assert_eq!(
            extract_web_search_query("can you search about quantum tunneling").as_deref(),
            Some("quantum tunneling")
        );
        assert_eq!(
            extract_web_search_query("find rust crates online").as_deref(),
            Some("rust crates")
        );
        // 'research' must not fire the word-boundary verb check.
        assert!(extract_web_search_query("i research online biology topics").is_none());

        assert_eq!(
            extract_web_search_query("look up rust async runtime online").as_deref(),
            Some("rust async runtime")
        );
        assert_eq!(
            extract_web_search_query("search the web for tokamak plasma stability").as_deref(),
            Some("tokamak plasma stability")
        );
        // Questions about search itself must NOT trigger a search.
        assert_eq!(extract_web_search_query("how does web search work?"), None);
        assert_eq!(extract_web_search_query("what is a web search engine"), None);
        // No search intent at all.
        assert_eq!(extract_web_search_query("tell me a joke"), None);
    }

    #[test]
    fn test_sanitize_query_strips_quotes_and_punctuation() {
        assert_eq!(sanitize_query("\"quantum computing\"?").as_deref(), Some("quantum computing"));
        assert_eq!(sanitize_query("  please  rust  ").as_deref(), Some("rust"));
    }

    #[test]
    fn test_smart_browser_site_expansion() {
        assert_eq!(
            extract_open_site_target("open youtube in default browser").as_deref(),
            Some("https://www.youtube.com")
        );
        assert_eq!(
            extract_open_site_target("open youtube.com").as_deref(),
            Some("https://youtube.com")
        );
        assert_eq!(
            extract_open_site_target("open chrome").as_deref(),
            Some("https://www.google.com")
        );
        assert_eq!(
            extract_open_site_target("open github").as_deref(),
            Some("https://github.com")
        );
        assert_eq!(
            extract_open_site_target("open liumi.cloud").as_deref(),
            Some("https://liumi.cloud")
        );
        // Local file reads must not be hijacked by the browser opener.
        assert_eq!(extract_open_site_target("open file notes.txt"), None);
        // Unmapped, non-domain words must not open anything.
        assert_eq!(extract_open_site_target("open the pod bay doors"), None);
    }

    #[test]
    fn test_file_creation_intent_parsing() {
        // The acceptance utterance: subfolder + default file name + content.
        assert_eq!(
            parse_file_creation("create a txt file in lixo folder and write hello world"),
            Some((
                "desktop/lixo".to_string(),
                "hello.txt".to_string(),
                "hello world".to_string()
            ))
        );
        // Explicit "into file" phrasing.
        assert_eq!(
            parse_file_creation("write hello world into file"),
            Some((
                "desktop".to_string(),
                "hello.txt".to_string(),
                "hello world".to_string()
            ))
        );
        // Explicit file name marker.
        assert_eq!(
            parse_file_creation("create a file in downloads folder named report"),
            Some((
                "downloads".to_string(),
                "report.txt".to_string(),
                "".to_string()
            ))
        );
        // Non-creation mentions must not fire.
        assert_eq!(parse_file_creation("read file notes.txt"), None);
        assert_eq!(parse_file_creation("what is a file"), None);
        assert_eq!(parse_file_creation("list files in documents"), None);
        assert_eq!(parse_file_creation("tell me a joke"), None);
    }

    #[test]
    fn test_folder_creation_intent_parsing() {
        assert_eq!(
            parse_folder_creation("create a folder on desktop and name it lixo"),
            Some(("desktop".to_string(), "lixo".to_string()))
        );
        assert_eq!(
            parse_folder_creation("make a folder in documents named invoices 2026"),
            Some(("documents".to_string(), "invoices 2026".to_string()))
        );
        assert_eq!(
            parse_folder_creation("create folder on downloads called dumps"),
            Some(("downloads".to_string(), "dumps".to_string()))
        );
        // Non-creation mentions must not fire.
        assert_eq!(parse_folder_creation("list files in my new folder"), None);
        assert_eq!(parse_folder_creation("what is a folder"), None);
    }

    #[test]
    fn test_timer_intent_parsing() {
        // The acceptance utterance.
        assert_eq!(
            parse_timer_intent("say hello to me after 1 minute"),
            Some(("hello to me".to_string(), 60))
        );
        // "remind me to" phrasing with seconds.
        assert_eq!(
            parse_timer_intent("remind me to stretch in 30 seconds"),
            Some(("stretch".to_string(), 30))
        );
        // Message-first phrasing with abbreviations.
        assert_eq!(
            parse_timer_intent("tell me to drink water in 5 mins"),
            Some(("drink water".to_string(), 300))
        );
        // Number-first phrasing.
        assert_eq!(
            parse_timer_intent("in 2 hours remind me to check the oven"),
            Some(("check the oven".to_string(), 7200))
        );
        assert_eq!(
            parse_timer_intent("after 1 hr say hello"),
            Some(("hello".to_string(), 3600))
        );
        // Trailing punctuation is stripped.
        assert_eq!(
            parse_timer_intent("remind me to breathe in 10 secs."),
            Some(("breathe".to_string(), 10))
        );
        // Non-timer utterances must not fire.
        assert_eq!(parse_timer_intent("what is a minute"), None);
        assert_eq!(parse_timer_intent("tell me a joke"), None);
        assert_eq!(parse_timer_intent("search the web for rust"), None);
        assert_eq!(parse_timer_intent("say hello to me after 0 minutes"), None);
    }

    #[test]
    fn test_extract_delayed_action() {
        // Acceptance utterance: clause stripped, case preserved.
        assert_eq!(
            extract_delayed_action("create a folder on desktop after 10 seconds and name it exp9"),
            Some((
                10,
                "create a folder on desktop and name it exp9".to_string()
            ))
        );
        // Delay-first phrasing.
        assert_eq!(
            extract_delayed_action("in 5 seconds open youtube"),
            Some((5, "open youtube".to_string()))
        );
        // Abbreviations and plural handling.
        assert_eq!(
            extract_delayed_action("write notes in 3 mins then stop"),
            Some((180, "write notes then stop".to_string()))
        );
        assert_eq!(
            extract_delayed_action("open github after 2 hrs"),
            Some((7200, "open github".to_string()))
        );
        // No delay clause -> None.
        assert_eq!(extract_delayed_action("create a folder on desktop"), None);
        // Zero delay is rejected.
        assert_eq!(extract_delayed_action("open youtube in 0 seconds"), None);
    }

    #[test]
    fn test_is_tool_intent() {
        // Acceptance actions are tools.
        assert!(is_tool_intent(
            "create a folder on desktop and name it exp9"
        ));
        assert!(is_tool_intent("open youtube"));
        assert!(is_tool_intent("open https://example.com/docs"));
        assert!(is_tool_intent("search the web for rust async"));
        // Reminder phrasing is NOT a tool -> stays on the timer path.
        assert!(!is_tool_intent("say hello to me"));
        assert!(!is_tool_intent("remind me to stretch"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(60), "1 minute");
        assert_eq!(format_duration(120), "2 minutes");
        assert_eq!(format_duration(30), "30 seconds");
        assert_eq!(format_duration(7200), "2 hours");
        assert_eq!(format_duration(3600), "1 hour");
        assert_eq!(format_duration(90), "90 seconds");
    }
}
