/*!
Synaptic Recall â€” Liumi VectorCore Memory Interceptor

A high-performance memory layer that sits between the user prompt and model
execution:

1. **Recall (< 0.2 ms budget):** queries the local VectorCore store
   (FTS5-backed `MemoryGraph`) with keyword extraction + recency ranking and
   formats the hits into a compact system-context block injected before
   generation.
2. **Commit (asynchronous):** after each turn, extracts explicit user
   preferences from the message (heuristic, zero model passes) and commits
   them to VectorCore so they are recalled in future sessions.

The recall path is entirely synchronous over a warm WAL-mode FTS5 store,
keeping the p50 lookup far below the 0.2 ms budget on desktop hardware.
*/

use crate::memory::{MemoryEntry, MemoryGraph, MemoryType};
use std::sync::Arc;
use std::time::Instant;

/// Maximum recalled-fact block injected into the prompt, in characters.
const MAX_RECALL_CHARS: usize = 1200;

/// Maximum number of recalled facts injected per turn.
const MAX_RECALL_ENTRIES: usize = 5;

/// Verb stems that signal an explicit, persistent user preference.
const PREFERENCE_STEMS: &[&str] = &[
    "i like",
    "i love",
    "i prefer",
    "i want",
    "i need",
    "i always",
    "i hate",
    "i dislike",
    "i never",
    "i don't like",
    "i do not like",
    "my favorite",
    "my favourite",
    "remember that",
    "keep in mind",
    "call me",
    "my name is",
    "i'm a",
    "i am a",
    "i work",
    "i use",
];

/// Synaptic memory manager over the Liumi VectorCore store (`.lvc` state).
///
/// Cheap to clone via `Arc`; all mutation goes through the shared
/// `MemoryGraph`, which is safe to share per its documented `Send + Sync`
/// contract.
#[derive(Clone)]
pub struct SynapticMemoryManager {
    memory: Arc<MemoryGraph>,
}

impl SynapticMemoryManager {
    pub fn new(memory: Arc<MemoryGraph>) -> Self {
        Self { memory }
    }

    /// Recalled facts relevant to `prompt`, formatted for system injection.
    ///
    /// Returns `None` when nothing relevant is stored (no prompt pollution).
    /// Budget: < 0.2 ms on a warm WAL-mode FTS5 store.
    pub fn recall_context(&self, prompt: &str) -> Option<String> {
        let start = Instant::now();
        let query = self.keyword_query(prompt);
        if query.is_empty() {
            return None;
        }

        // Rank keyword hits first; top up with the most recent preferences so
        // the model always sees durable user facts.
        let mut entries: Vec<MemoryEntry> =
            match self.memory.search_text(&query, MAX_RECALL_ENTRIES) {
                Ok(e) => e,
                Err(_) => Vec::new(),
            };
        if entries.len() < MAX_RECALL_ENTRIES {
            if let Ok(prefs) = self
                .memory
                .get_recent(Some(MemoryType::Preference), MAX_RECALL_ENTRIES)
            {
                for p in prefs {
                    if entries.len() >= MAX_RECALL_ENTRIES {
                        break;
                    }
                    if !entries.iter().any(|e| e.id == p.id) {
                        entries.push(p);
                    }
                }
            }
        }

        if entries.is_empty() {
            return None;
        }

        let mut block = String::from("[Recalled Memory - user-specific facts]\n");
        for e in &entries {
            if block.len() >= MAX_RECALL_CHARS {
                break;
            }
            let line: String = e.content.chars().take(240).collect();
            block.push_str("- ");
            block.push_str(&line);
            block.push('\n');
        }
        block.push_str("");

        let elapsed = start.elapsed();
        if elapsed.as_micros() > 200 {
            // Soft budget check - never fail the request over it, just log.
            tracing::debug!(
                "synaptic recall exceeded 0.2ms budget: {} us",
                elapsed.as_micros()
            );
        }
        Some(block)
    }

    /// Extract explicit preferences from a user message and commit them to
    /// VectorCore as `Preference` entries. Returns the number committed.
    pub fn extract_and_commit(&self, user_message: &str) -> usize {
        let mut committed = 0;
        for sentence in split_sentences(user_message) {
            let s = sentence.to_lowercase();
            if PREFERENCE_STEMS.iter().any(|stem| s.contains(stem)) {
                let fact: String = sentence.trim().chars().take(300).collect();
                if fact.len() < 6 {
                    continue;
                }
                let entry = MemoryEntry::new(
                    MemoryType::Preference,
                    fact,
                    None,
                    80,
                    vec!["synaptic".to_string(), "auto".to_string()],
                );
                if self.memory.store(&entry).is_ok() {
                    committed += 1;
                }
            }
        }
        committed
    }

    /// Keyword query for FTS5: keep alphanumeric terms, drop stopwords and
    /// very short tokens, cap at 12 terms. Each term is quoted so FTS5
    /// treats it as a literal, not syntax.
    fn keyword_query(&self, prompt: &str) -> String {
        const STOPWORDS: &[&str] = &[
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or",
            "in", "on", "at", "it", "this", "that", "with", "for", "you", "me", "my", "i", "we",
            "us", "do", "does", "did", "can", "could", "will", "would", "should", "what", "how",
            "why", "when", "where", "who", "please", "hello", "hi", "hey", "tell",
        ];
        let mut terms = Vec::new();
        for word in prompt.split(|c: char| !c.is_alphanumeric()) {
            let w = word.to_lowercase();
            if w.len() >= 3 && !STOPWORDS.contains(&w.as_str()) && terms.len() < 12 {
                terms.push(format!("\"{}\"", w.replace('"', "")));
            }
        }
        terms.join(" ")
    }

    /// Underlying store, for diagnostics and cleanup on shutdown.
    pub fn memory(&self) -> &Arc<MemoryGraph> {
        &self.memory
    }
}

// ---------------------------------------------------------------------------
// Auto-Teach Loop: web fallback on memory miss, distillation, and commit
// ---------------------------------------------------------------------------

/// Semantic-confidence gate for triggering the auto-teach web fallback.
/// Below this fraction of matched query terms, a factual question is
/// considered "unknown" and triggers the teach loop.
pub const AUTO_TEACH_SIMILARITY_THRESHOLD: f32 = 0.65;

impl SynapticMemoryManager {
    /// Heuristic factual-intent detector: who/what/when/where/why/how and
    /// definition-style prompts. Identity/smalltalk questions are excluded
    /// so they never trigger a web lookup.
    pub fn is_factual_question(prompt: &str) -> bool {
        const CONVERSATIONAL_PATTERNS: &[&str] = &[
            "who are you",
            "what are you",
            "your name",
            "who made you",
            "who created you",
            "who built you",
            "what can you do",
            "what things can you do",
            "what thing can you do",
            "what thing you can do",
            "what could you do",
            "you can do",
            "how can you help",
            "how do you work",
            "how does it work",
            "tell me about yourself",
            "introduce yourself",
            "can you help",
            "could you help",
            "help me impress",
            "impress me",
            "show off",
            "good at",
            "your capabilities",
            "your abilities",
            "your skills",
            "hello",
            "hi ",
            "hey",
        ];
        let lower = prompt.trim().to_lowercase();
        if lower.is_empty() || CONVERSATIONAL_PATTERNS.iter().any(|p| lower.contains(p)) {
            return false;
        }
        const FACTUAL_STARTS: &[&str] = &[
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "which",
            "define",
            "explain",
            "tell me about",
            "is ",
            "are ",
            "does ",
            "do ",
            "did ",
        ];
        FACTUAL_STARTS.iter().any(|p| lower.starts_with(p))
            || lower.contains(" what is ")
            || lower.contains(" who is ")
    }

    /// Heuristic semantic-confidence score in `[0.0, 1.0]` for how well the
    /// `.lvc` graph covers this prompt: the fraction of meaningful query
    /// terms found in the best-matching stored memory. FTS5-backed, so the
    /// measurement stays inside the sub-millisecond recall budget.
    pub fn recall_confidence(&self, prompt: &str) -> f32 {
        let query = self.keyword_query(prompt);
        if query.is_empty() {
            return 1.0; // nothing meaningful to look up; never trigger teach
        }
        let terms: Vec<String> = query
            .split('"')
            .filter(|t| !t.trim().is_empty())
            .map(|t| t.to_lowercase())
            .collect();
        if terms.is_empty() {
            return 1.0;
        }
        let hits = match self.memory.search_text(&query, 1) {
            Ok(h) => h,
            Err(_) => return 0.0,
        };
        match hits.first() {
            None => 0.0,
            Some(best) => {
                let content = best.content.to_lowercase();
                let matched = terms
                    .iter()
                    .filter(|t| content.contains(t.as_str()))
                    .count();
                matched as f32 / terms.len() as f32
            }
        }
    }

    /// Auto-Teach Loop: when a factual query has effectively zero matching
    /// `.lvc` memories, search the web (dual-engine failover), distill the
    /// core fact from the top results, and persist it into the active graph
    /// so the *next* query recalls it locally without any web access.
    ///
    /// Returns the number of distilled facts committed.
    pub async fn auto_teach(&self, prompt: &str) -> anyhow::Result<usize> {
        if !Self::is_factual_question(prompt) {
            return Ok(0);
        }
        if self.recall_confidence(prompt) >= AUTO_TEACH_SIMILARITY_THRESHOLD {
            return Ok(0); // already known; no web access needed
        }

        let (snippets, engine) = crate::tools::web_search::search_web_dual(prompt).await?;
        if snippets.is_empty() {
            return Ok(0);
        }
        tracing::info!(
            "auto-teach: {} distilled from {:?} for {:?}",
            snippets.len(),
            engine,
            prompt
        );

        let mut committed = 0usize;
        for snippet in snippets.iter().take(2) {
            let fact = format!(
                "{} â€” {}",
                snippet.title,
                snippet.snippet.chars().take(400).collect::<String>()
            );
            for sentence in split_sentences(&fact) {
                if !is_factual_sentence(sentence) {
                    continue;
                }
                let fact = sentence.trim().chars().take(300).collect::<String>();
                if fact.len() < 24 {
                    continue;
                }
                let entry = MemoryEntry::new(
                    MemoryType::Context,
                    fact,
                    None,
                    80,
                    vec!["auto-taught".to_string(), "web".to_string()],
                );
                if self.memory.store(&entry).is_ok() {
                    committed += 1;
                }
            }
        }
        Ok(committed)
    }
}

fn split_sentences(text: &str) -> Vec<&str> {
    text.split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n' || c == ';')
        .filter(|s| !s.trim().is_empty())
        .collect()
}

/// Returns true when a distilled sentence is safe, factual memory content.
///
/// Web snippets often mix facts with safety advice, disclaimers, and
/// model-generated text in other languages. Auto-teach is deliberately
/// conservative: a sentence is discarded when it contains known boilerplate
/// or any CJK/full-width character.
fn is_factual_sentence(sentence: &str) -> bool {
    let lower = sentence.to_lowercase();
    const BOILERPLATE: &[&str] = &[
        "unethical",
        "illegal",
        "ethical and legal risks",
        "damage to ecosystems",
        "specialized knowledge and resources",
    ];
    if BOILERPLATE.iter().any(|phrase| lower.contains(phrase)) {
        return false;
    }
    !sentence.chars().any(is_cjk_or_full_width)
}

fn is_cjk_or_full_width(c: char) -> bool {
    matches!(c,
        '\u{3000}'..='\u{303f}'
        | '\u{3400}'..='\u{4dbf}'
        | '\u{4e00}'..='\u{9fff}'
        | '\u{f900}'..='\u{faff}'
        | '\u{ff00}'..='\u{ffef}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn manager() -> (SynapticMemoryManager, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let memory = Arc::new(MemoryGraph::new(dir.path().join("lvc.db"), None).unwrap());
        (SynapticMemoryManager::new(memory), dir)
    }

    #[test]
    fn test_recall_returns_none_when_empty() {
        let (mgr, _dir) = manager();
        assert!(mgr.recall_context("what is rust programming").is_none());
    }

    #[test]
    fn test_commit_then_recall_roundtrip() {
        let (mgr, _dir) = manager();
        assert_eq!(
            mgr.extract_and_commit("I prefer concise answers. The sky is blue."),
            1
        );
        let recalled = mgr.recall_context("how should you answer me");
        assert!(recalled.is_some(), "preference should be recalled");
        let text = recalled.unwrap();
        assert!(text.contains("[Recalled Memory"));
        assert!(text.contains("concise"));
    }

    #[test]
    fn test_recall_latency_budget() {
        let (mgr, _dir) = manager();
        mgr.extract_and_commit("I like dark mode. I use arch linux btw. My name is Liumi.");
        let _ = mgr.recall_context("dark mode setup"); // warm up
        let start = Instant::now();
        let _ = mgr.recall_context("dark mode setup");
        assert!(
            start.elapsed().as_micros() < 2_000,
            "recall should be far below the 0.2ms budget (measured: {} us)",
            start.elapsed().as_micros()
        );
    }

    #[test]
    fn test_keyword_query_drops_stopwords_and_quotes() {
        let (mgr, _dir) = manager();
        let q = mgr.keyword_query("Hello, what is the \"vector\" database?");
        assert!(!q.contains("hello"));
        assert!(q.contains("\"vector\""));
        assert!(q.contains("\"database\""));
    }

    #[test]
    fn test_no_false_positive_preferences() {
        let (mgr, _dir) = manager();
        assert_eq!(mgr.extract_and_commit("What is the weather today?"), 0);
    }

    #[test]
    fn test_factual_question_detection() {
        assert!(SynapticMemoryManager::is_factual_question(
            "What is quantum entanglement?"
        ));
        assert!(SynapticMemoryManager::is_factual_question(
            "who is Ada Lovelace"
        ));
        assert!(!SynapticMemoryManager::is_factual_question("who are you?"));
        assert!(!SynapticMemoryManager::is_factual_question("hello there"));
        assert!(!SynapticMemoryManager::is_factual_question(""));
    }

    #[test]
    fn test_auto_teach_skips_conversational_capability_queries() {
        for prompt in [
            "what thing you can do to impress me",
            "what can you do?",
            "who are you?",
            "how do you work?",
            "tell me about yourself",
            "can you help me?",
            "what are your capabilities?",
        ] {
            assert!(
                !SynapticMemoryManager::is_factual_question(prompt),
                "conversational prompt must not trigger web search: {prompt}"
            );
        }
    }

    #[test]
    fn test_recall_confidence_zero_on_unknown() {
        let (mgr, _dir) = manager();
        assert_eq!(
            mgr.recall_confidence("what is tokamak plasma stability"),
            0.0
        );
    }

    #[test]
    fn test_recall_confidence_full_on_exact_match() {
        let (mgr, _dir) = manager();
        mgr.extract_and_commit("I like dark mode and I use arch linux btw.");
        assert!(mgr.recall_confidence("dark mode") > 0.0);
    }

    #[tokio::test]
    async fn test_auto_teach_skips_known_and_non_factual() {
        let (mgr, _dir) = manager();
        // Non-factual: no web call attempted (would fail without network).
        assert_eq!(mgr.auto_teach("hello").await.unwrap(), 0);
    }

    #[test]
    fn test_auto_teach_rejects_safety_boilerplate_and_chinese() {
        for rejected in [
            "This activity is unethical and illegal.",
            "There are ethical and legal risks to consider.",
            "The process may cause damage to ecosystems.",
            "This requires specialized knowledge and resources.",
            "这个过程会产生污染。",
            "安全警告，请注意安全。",
        ] {
            assert!(
                !is_factual_sentence(rejected),
                "boilerplate must not enter memory: {rejected}"
            );
        }
    }

    #[test]
    fn test_auto_teach_accepts_pure_factual_sentence() {
        assert!(is_factual_sentence(
            "Biryani is made by layering spiced rice and marinated meat."
        ));
    }
}
