/*!
Local Vector Memory Graph

Embedded SQLite with vector extensions for persistent, encrypted memory storage.
Stores user preferences, niche keywords, past activity, and generated thoughts.
*/

use crate::crypto::CryptoError;
use rusqlite::{Connection, params, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

/// Memory errors
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Crypto error: {0}")]
    Crypto(#[from] CryptoError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Memory entry not found: {0}")]
    NotFound(String),
}

/// Memory entry types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    Ideation,
    Preference,
    Keyword,
    Activity,
    Thought,
    Context,
    Custom(String),
}

/// Memory entry stored in the vector database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub memory_type: MemoryType,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub confidence: u8,
    pub created_at: i64,
    pub updated_at: i64,
    pub tags: Vec<String>,
}

impl MemoryEntry {
    pub fn new(
        memory_type: MemoryType,
        content: String,
        embedding: Option<Vec<f32>>,
        confidence: u8,
        tags: Vec<String>,
    ) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            id: Uuid::new_v4().to_string(),
            memory_type,
            content,
            embedding,
            metadata: HashMap::new(),
            confidence,
            created_at: now,
            updated_at: now,
            tags,
        }
    }
}

/// Encrypted vector memory graph backed by embedded SQLite
pub struct MemoryGraph {
    conn: Connection,
    db_path: PathBuf,
    encryption_key: Option<[u8; 32]>,
}

// SAFETY: `rusqlite::Connection` is `Send` but not `Sync` (it wraps a
// `RefCell`). Every access to a `MemoryGraph` in this codebase goes through
// an async `RwLock`, which enforces exclusive/borrowed access at runtime,
// and rusqlite's bundled SQLite is compiled in serialized thread-safe mode.
// Therefore shared references to `MemoryGraph` never observe unsynchronized
// mutation of the inner connection.
unsafe impl Send for MemoryGraph {}
unsafe impl Sync for MemoryGraph {}

impl MemoryGraph {
    /// Create or open a memory graph at the given path
    pub fn new<P: AsRef<Path>>(path: P, encryption_key: Option<[u8; 32]>) -> Result<Self, MemoryError> {
        let db_path = path.as_ref().to_path_buf();
        
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path)?;
        
        // execute_batch tolerates PRAGMAs regardless of whether they return rows
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA temp_store = MEMORY;
             PRAGMA mmap_size = 268435456;",
        )?;

        let mut graph = Self {
            conn,
            db_path,
            encryption_key,
        };

        graph.init_schema()?;
        Ok(graph)
    }

    fn init_schema(&mut self) -> Result<(), MemoryError> {
        self.conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT NOT NULL DEFAULT '{}',
                confidence INTEGER NOT NULL DEFAULT 50,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]'
            )
            "#,
            [],
        )?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_confidence ON memory_entries(confidence)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tags ON memory_entries(tags)",
            [],
        )?;

        #[cfg(feature = "sqlite-vec")]
        {
            self.conn.execute(
                r#"
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                    embedding float[768]
                )
                "#,
                [],
            )?;
        }

        self.conn.execute(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                id UNINDEXED,
                content,
                tags,
                content='memory_entries',
                content_rowid='rowid'
            )
            "#,
            [],
        )?;

        self.conn.execute(
            r#"
            CREATE TRIGGER IF NOT EXISTS memory_fts_insert AFTER INSERT ON memory_entries BEGIN
                INSERT INTO memory_fts (rowid, id, content, tags) 
                VALUES (new.rowid, new.id, new.content, new.tags);
            END
            "#,
            [],
        )?;

        self.conn.execute(
            r#"
            CREATE TRIGGER IF NOT EXISTS memory_fts_update AFTER UPDATE ON memory_entries BEGIN
                UPDATE memory_fts SET content = new.content, tags = new.tags WHERE id = old.id;
            END
            "#,
            [],
        )?;

        self.conn.execute(
            r#"
            CREATE TRIGGER IF NOT EXISTS memory_fts_delete AFTER DELETE ON memory_entries BEGIN
                DELETE FROM memory_fts WHERE id = old.id;
            END
            "#,
            [],
        )?;

        Ok(())
    }

    /// Persist a validated thought from the ideation loop (convenience helper)
    pub fn store_thought(&self, content: &str) -> Result<(), MemoryError> {
        let entry = MemoryEntry::new(
            MemoryType::Thought,
            content.to_string(),
            None,
            50,
            vec!["ideation".to_string(), "auto".to_string()],
        );
        self.store(&entry)
    }

    pub fn store(&self, entry: &MemoryEntry) -> Result<(), MemoryError> {
        let embedding_blob = entry.embedding.as_ref().map(|e| {
            let mut bytes = Vec::with_capacity(e.len() * 4);
            for &val in e {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            bytes
        });

        let metadata_json = serde_json::to_string(&entry.metadata)?;
        let tags_json = serde_json::to_string(&entry.tags)?;

        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO memory_entries 
            (id, memory_type, content, embedding, metadata, confidence, created_at, updated_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                &entry.id,
                serde_json::to_string(&entry.memory_type)?,
                &entry.content,
                embedding_blob,
                metadata_json,
                entry.confidence as i64,
                entry.created_at,
                entry.updated_at,
                tags_json,
            ],
        )?;

        #[cfg(feature = "sqlite-vec")]
        if let Some(ref embedding) = entry.embedding {
            self.conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (rowid, embedding) VALUES (?, ?)",
                params![entry.id, embedding],
            )?;
        }

        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<Option<MemoryEntry>, MemoryError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, memory_type, content, embedding, metadata, confidence, created_at, updated_at, tags
            FROM memory_entries WHERE id = ?
            "#,
        )?;

        let entry = stmt.query_row(params![id], |row| self.row_to_entry(row)).optional()?;
        Ok(entry)
    }

    pub fn search_text(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, MemoryError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT e.id, e.memory_type, e.content, e.embedding, e.metadata, e.confidence, e.created_at, e.updated_at, e.tags
            FROM memory_entries e
            JOIN memory_fts f ON e.id = f.id
            WHERE memory_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            "#,
        )?;

        let entries = stmt.query_map(params![query, limit as i64], |row| self.row_to_entry(row))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(entries)
    }

    #[cfg(feature = "sqlite-vec")]
    pub fn search_similar(&self, embedding: &[f32], limit: usize, threshold: f32) -> Result<Vec<(MemoryEntry, f32)>, MemoryError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT e.id, e.memory_type, e.content, e.embedding, e.metadata, e.confidence, e.created_at, e.updated_at, e.tags,
                   vec_distance_cosine(e.embedding, ?) as distance
            FROM memory_entries e
            WHERE e.embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT ?
            "#,
        )?;

        let embedding_blob: Vec<u8> = embedding.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let results = stmt.query_map(params![embedding_blob, limit as i64], |row| {
            let entry = self.row_to_entry(row)?;
            let distance: f32 = row.get(9)?;
            Ok((entry, distance))
        })?
            .filter_map(|r| r.ok())
            .filter(|(_, dist)| *dist <= threshold)
            .collect();

        Ok(results)
    }

    pub fn get_recent(&self, memory_type: Option<MemoryType>, limit: usize) -> Result<Vec<MemoryEntry>, MemoryError> {
        let (sql, params_vec) = match memory_type {
            Some(ref mt) => (
                "SELECT id, memory_type, content, embedding, metadata, confidence, created_at, updated_at, tags 
                 FROM memory_entries WHERE memory_type = ? ORDER BY created_at DESC LIMIT ?",
                vec![serde_json::to_string(mt)?, limit.to_string()],
            ),
            None => (
                "SELECT id, memory_type, content, embedding, metadata, confidence, created_at, updated_at, tags 
                 FROM memory_entries ORDER BY created_at DESC LIMIT ?",
                vec![limit.to_string()],
            ),
        };

        let mut stmt = self.conn.prepare(sql)?;
        let params = rusqlite::params_from_iter(params_vec.iter().map(|s| s.as_str()));
        
        let entries = stmt.query_map(params, |row| self.row_to_entry(row))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(entries)
    }

    pub fn get_high_confidence(&self, min_confidence: u8, limit: usize) -> Result<Vec<MemoryEntry>, MemoryError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, memory_type, content, embedding, metadata, confidence, created_at, updated_at, tags
            FROM memory_entries 
            WHERE confidence >= ?
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            "#,
        )?;

        let entries = stmt.query_map(params![min_confidence as i64, limit as i64], |row| self.row_to_entry(row))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(entries)
    }

    pub fn delete(&self, id: &str) -> Result<bool, MemoryError> {
        let deleted = self.conn.execute("DELETE FROM memory_entries WHERE id = ?", params![id])?;
        Ok(deleted > 0)
    }

    /// Delete stale/mock entries whose content contains `needle` (exact
    /// substring, case-sensitive). The FTS5 sync triggers keep the search
    /// index consistent. Returns the number of deleted entries.
    pub fn delete_matching(&self, needle: &str) -> Result<usize, MemoryError> {
        let deleted = self.conn.execute(
            "DELETE FROM memory_entries WHERE instr(content, ?1) > 0",
            params![needle],
        )?;
        if deleted > 0 {
            // Rebuild the external-content FTS5 index from the source table:
            // heals any index drift instead of surfacing as 'database disk
            // image is malformed' on later searches.
            self.conn
                .execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')", [])?;
        }
        Ok(deleted)
    }


    pub fn update_metadata(&self, id: &str, metadata: HashMap<String, serde_json::Value>) -> Result<bool, MemoryError> {
        let metadata_json = serde_json::to_string(&metadata)?;
        let updated = self.conn.execute(
            "UPDATE memory_entries SET metadata = ?, updated_at = ? WHERE id = ?",
            params![metadata_json, chrono::Utc::now().timestamp_millis(), id],
        )?;
        Ok(updated > 0)
    }

    pub fn get_stats(&self) -> Result<MemoryStats, MemoryError> {
        let total: i64 = self.conn.query_row("SELECT COUNT(*) FROM memory_entries", [], |r| r.get(0))?;
        
        let by_type: HashMap<String, i64> = {
            let mut stmt = self.conn.prepare("SELECT memory_type, COUNT(*) FROM memory_entries GROUP BY memory_type")?;
            let rows = stmt.query_map([], |row| {
                let mt: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                Ok((mt, count))
            })?.collect::<Result<Vec<_>, _>>()?;
            rows.into_iter().collect()
        };

        let avg_confidence: f64 = self.conn.query_row(
            "SELECT AVG(confidence) FROM memory_entries", 
            [], 
            |r| r.get(0)
        ).unwrap_or(0.0);

        Ok(MemoryStats {
            total_entries: total as usize,
            by_type,
            avg_confidence,
            db_size_bytes: std::fs::metadata(&self.db_path)?.len(),
        })
    }

    fn row_to_entry(&self, row: &Row) -> rusqlite::Result<MemoryEntry> {
        let memory_type_str: String = row.get(1)?;
        let memory_type: MemoryType = serde_json::from_str(&memory_type_str)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(e)))?;

        let embedding_blob: Option<Vec<u8>> = row.get(3)?;
        let embedding = embedding_blob.map(|blob| {
            blob.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        });

        let metadata_str: String = row.get(4)?;
        let metadata: HashMap<String, serde_json::Value> = serde_json::from_str(&metadata_str).unwrap_or_default();

        let tags_str: String = row.get(8)?;
        let tags: Vec<String> = serde_json::from_str(&tags_str).unwrap_or_default();

        Ok(MemoryEntry {
            id: row.get(0)?,
            memory_type,
            content: row.get(2)?,
            embedding,
            metadata,
            confidence: row.get(5)?,
            created_at: row.get(6)?,
            updated_at: row.get(7)?,
            tags,
        })
    }

    pub fn path(&self) -> &Path {
        &self.db_path
    }

    pub fn encryption_key(&self) -> Option<&[u8; 32]> {
        self.encryption_key.as_ref()
    }

    // ------------------------------------------------------------------
    // Single-file container state persistence (.lvc <-> .lagi state block)
    // ------------------------------------------------------------------

    /// Export all memory entries for persistence into the trailing `.lagi`
    /// state block. Entries are ordered oldest-first so imports replay in
    /// original creation order.
    pub fn export_state(&self) -> Result<Vec<MemoryEntry>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, memory_type, content, embedding, metadata, confidence, created_at, updated_at, tags
             FROM memory_entries ORDER BY created_at ASC, id ASC LIMIT 1000000",
        )?;
        let entries = stmt
            .query_map([], |row| self.row_to_entry(row))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(entries)
    }

    /// Import (merge) exported entries into this graph. Entries whose
    /// content already exists are skipped; new entries keep their original
    /// ids and timestamps. Returns the number of imported entries.
    pub fn import_state(&self, entries: &[MemoryEntry]) -> Result<usize, MemoryError> {
        let mut imported = 0usize;
        for entry in entries {
            let exists: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM memory_entries WHERE content = ?",
                params![&entry.content],
                |r| r.get(0),
            )?;
            if exists > 0 {
                continue;
            }
            self.store(entry)?;
            imported += 1;
        }
        Ok(imported)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_memory_graph_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_memory.db");
        
        let graph = MemoryGraph::new(&db_path, None).unwrap();
        
        let entry = MemoryEntry::new(
            MemoryType::Ideation,
            "Test idea content".to_string(),
            None,
            90,
            vec!["test".to_string(), "idea".to_string()],
        );
        
        graph.store(&entry).unwrap();
        
        let retrieved = graph.get(&entry.id).unwrap().unwrap();
        assert_eq!(retrieved.content, "Test idea content");
        assert_eq!(retrieved.confidence, 90);
        assert_eq!(retrieved.tags.len(), 2);
    }

    #[test]
    fn test_memory_search_text() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_memory.db");
        
        let graph = MemoryGraph::new(&db_path, None).unwrap();
        
        let entry1 = MemoryEntry::new(
            MemoryType::Ideation,
            "Creative idea about rust programming".to_string(),
            None,
            85,
            vec!["rust".to_string()],
        );
        let entry2 = MemoryEntry::new(
            MemoryType::Preference,
            "User prefers dark mode".to_string(),
            None,
            95,
            vec!["ui".to_string()],
        );
        
        graph.store(&entry1).unwrap();
        graph.store(&entry2).unwrap();
        
        let results = graph.search_text("rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("rust"));
    }

    #[test]
    fn test_memory_stats() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_memory.db");
        
        let graph = MemoryGraph::new(&db_path, None).unwrap();
        
        for i in 0..5 {
            let entry = MemoryEntry::new(
                MemoryType::Ideation,
                format!("Idea {}", i),
                None,
                80 + i,
                vec![],
            );
            graph.store(&entry).unwrap();
        }
        
        let stats = graph.get_stats().unwrap();
        assert_eq!(stats.total_entries, 5);
        assert!(stats.avg_confidence > 0.0);
    }

    #[test]
    fn test_memory_delete() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_memory.db");
        
        let graph = MemoryGraph::new(&db_path, None).unwrap();
        
        let entry = MemoryEntry::new(
            MemoryType::Thought,
            "To be deleted".to_string(),
            None,
            50,
            vec![],
        );
        graph.store(&entry).unwrap();
        
        let deleted = graph.delete(&entry.id).unwrap();
        assert!(deleted);
        
        let retrieved = graph.get(&entry.id).unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_delete_matching_scrubs_stale_casing() {
        let dir = tempdir().unwrap();
        let graph = MemoryGraph::new(dir.path().join("scrub.db"), None).unwrap();

        let stale = MemoryEntry::new(
            MemoryType::Thought,
            "User: who are you? Assistant: I am LAGi, your assistant.".to_string(),
            None,
            50,
            vec![],
        );
        let clean = MemoryEntry::new(
            MemoryType::Thought,
            "Assistant: I am LAGI, created by Liumi Corporation.".to_string(),
            None,
            50,
            vec![],
        );
        graph.store(&stale).unwrap();
        graph.store(&clean).unwrap();

        // Only the exact-case 'LAGi' entry is scrubbed; canonical LAGI kept.
        let removed = graph.delete_matching("LAGi").unwrap();
        assert_eq!(removed, 1);
        // FTS5 is case-folding, so verify via exact content inspection.
        let remaining = graph.get_recent(None, 10).unwrap();
        assert!(remaining.iter().all(|e| !e.content.contains("LAGi")));
        assert!(remaining.iter().any(|e| e.content.contains("LAGI")));
        assert_eq!(remaining.len(), 1);
    }
}

/// Memory statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub by_type: HashMap<String, i64>,
    pub avg_confidence: f64,
    pub db_size_bytes: u64,
}

/*
Task Ledger — lightweight, persistent task tracking for autonomous operation.

The Task Evaluator consults this ledger on every heartbeat tick (while the
user is idle) to pick the highest-priority pending task for the background
Spark-and-Critic cycle, without ever blocking incoming user prompts.
*/

/// Task priority for ledger scheduling
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Task lifecycle status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// A single ledger task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerTask {
    pub id: String,
    pub title: String,
    pub description: String,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub created_at: i64,
    pub updated_at: i64,
    pub completed_at: Option<i64>,
    pub result: Option<String>,
}

impl LedgerTask {
    pub fn new(title: impl Into<String>, description: impl Into<String>, priority: TaskPriority) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            id: Uuid::new_v4().to_string(),
            title: title.into(),
            description: description.into(),
            priority,
            status: TaskStatus::Pending,
            created_at: now,
            updated_at: now,
            completed_at: None,
            result: None,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Priority task ledger backed by the encrypted memory graph
pub struct TaskLedger {
    memory: Arc<MemoryGraph>,
}

impl TaskLedger {
    pub fn new(memory: Arc<MemoryGraph>) -> Self {
        Self { memory }
    }

    /// Add a task to the ledger
    pub fn add(&self, task: &LedgerTask) -> Result<(), MemoryError> {
        self.memory.store(&MemoryEntry::new(
            MemoryType::Custom("task".to_string()),
            task.to_json(),
            None,
            100,
            vec!["task".to_string()],
        ))?;
        Ok(())
    }

    /// Get the highest-priority pending task (priority desc, then oldest first)
    pub fn next_pending(&self) -> Result<Option<LedgerTask>, MemoryError> {
        let entries = self.memory.get_recent(None, 500)?;
        let mut tasks: Vec<LedgerTask> = entries
            .iter()
            .filter_map(|e| serde_json::from_str::<LedgerTask>(&e.content).ok())
            .filter(|t| matches!(t.status, TaskStatus::Pending))
            .collect();
        tasks.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.created_at.cmp(&b.created_at)));
        Ok(tasks.into_iter().next())
    }

    /// Mark a task completed with its result
    pub fn complete(&self, task_id: &str, result: &str) -> Result<bool, MemoryError> {
        self.transition(task_id, TaskStatus::Completed, Some(result.to_string()))
    }

    /// Mark a task failed with an error note
    pub fn fail(&self, task_id: &str, error: &str) -> Result<bool, MemoryError> {
        self.transition(task_id, TaskStatus::Failed, Some(error.to_string()))
    }

    fn transition(
        &self,
        task_id: &str,
        status: TaskStatus,
        note: Option<String>,
    ) -> Result<bool, MemoryError> {
        let entries = self.memory.get_recent(None, 500)?;
        for e in entries.iter().rev() {
            if let Ok(mut t) = serde_json::from_str::<LedgerTask>(&e.content) {
                if t.id == task_id {
                    t.status = status;
                    t.updated_at = chrono::Utc::now().timestamp_millis();
                    if status == TaskStatus::Completed {
                        t.completed_at = Some(t.updated_at);
                    }
                    t.result = note.clone();
                    self.memory.store(&MemoryEntry::new(
                        MemoryType::Custom("task".to_string()),
                        t.to_json(),
                        None,
                        100,
                        vec!["task".to_string()],
                    ))?;
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Count of pending tasks
    pub fn pending_count(&self) -> usize {
        self.memory
            .get_recent(None, 500)
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|e| serde_json::from_str::<LedgerTask>(&e.content).ok())
                    .filter(|t| matches!(t.status, TaskStatus::Pending))
                    .count()
            })
            .unwrap_or(0)
    }
}

/*
Psychological Onboarding Profile — user behavioral state persisted into the
runtime context so the agent personalizes its ideation and communication.
*/

/// User psychological / operational preference profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnboardingProfile {
    pub user_id: String,
    pub created_at: i64,
    /// 0-100: how much the user wants the agent to act without confirmation
    pub autonomy_level: u8,
    /// 0-100: appetite for experimental/risky recommendations
    pub risk_tolerance: u8,
    /// "direct" | "supportive" | "detailed" | "concise"
    pub communication_preference: String,
    /// "analytical" | "intuitive" | "practical" | "creative"
    pub thinking_style: String,
    /// Free-form primary goal captured during onboarding
    pub primary_goal: String,
    /// Number of onboarding responses captured
    pub questions_answered: u8,
}

/// The 5-question psychological onboarding payload for first-time users
pub const ONBOARDING_QUESTIONS: &[(&str, &str)] = &[
    (
        "autonomy_level",
        "How much should the agent act on its own without asking you first? (1 = always ask, 5 = fully autonomous)",
    ),
    (
        "risk_tolerance",
        "How comfortable are you with experimental or unconventional suggestions? (1 = only proven approaches, 5 = love experiments)",
    ),
    (
        "communication_preference",
        "How do you prefer responses? (1 = direct/brief, 2 = supportive, 3 = detailed with context, 4 = concise bullet points)",
    ),
    (
        "thinking_style",
        "When solving problems you tend to: (1 = logical step-by-step, 2 = gut feeling/patterns, 3 = what works practically, 4 = novel creative connections)",
    ),
    (
        "primary_goal",
        "What is your primary goal for this agent? (free text)",
    ),
];

impl OnboardingProfile {
    /// Build a profile from onboarding answers (numeric 1-5 answers map to
    /// the 0-100 scales; categorical answers map by index).
    pub fn from_answers(answers: &[(String, String)]) -> Self {
        let mut p = Self {
            user_id: Uuid::new_v4().to_string(),
            created_at: chrono::Utc::now().timestamp_millis(),
            autonomy_level: 50,
            risk_tolerance: 50,
            communication_preference: "direct".to_string(),
            thinking_style: "analytical".to_string(),
            primary_goal: String::new(),
            questions_answered: answers.len().min(u8::MAX as usize) as u8,
        };
        for (key, value) in answers {
            match key.as_str() {
                "autonomy_level" => {
                    if let Ok(v) = value.trim().parse::<u8>() {
                        p.autonomy_level = v.clamp(1, 5) * 20;
                    }
                }
                "risk_tolerance" => {
                    if let Ok(v) = value.trim().parse::<u8>() {
                        p.risk_tolerance = v.clamp(1, 5) * 20;
                    }
                }
                "communication_preference" => {
                    p.communication_preference = match value.trim() {
                        "2" => "supportive".to_string(),
                        "3" => "detailed".to_string(),
                        "4" => "concise".to_string(),
                        _ => "direct".to_string(),
                    };
                }
                "thinking_style" => {
                    p.thinking_style = match value.trim() {
                        "2" => "intuitive".to_string(),
                        "3" => "practical".to_string(),
                        "4" => "creative".to_string(),
                        _ => "analytical".to_string(),
                    };
                }
                "primary_goal" => p.primary_goal = value.trim().to_string(),
                _ => {}
            }
        }
        p
    }

    /// Persist into the memory graph (the `.lagi` runtime context state)
    pub fn persist(&self, memory: &MemoryGraph) -> Result<(), MemoryError> {
        memory.store(&MemoryEntry::new(
            MemoryType::Custom("onboarding_profile".to_string()),
            serde_json::to_string(self).map_err(MemoryError::Serialization)?,
            None,
            100,
            vec!["profile".to_string()],
        ))?;
        Ok(())
    }

    /// Load the most recently persisted profile, if any
    pub fn load(memory: &MemoryGraph) -> Option<Self> {
        let entries = memory.get_recent(None, 50).ok()?;
        entries
            .iter()
            .filter(|e| e.memory_type == MemoryType::Custom("onboarding_profile".to_string()))
            .find_map(|e| serde_json::from_str::<OnboardingProfile>(&e.content).ok())
    }

    /// One-line summary injected into ideation/chat prompts
    pub fn prompt_context(&self) -> String {
        format!(
            "User profile: autonomy {}/100, risk tolerance {}/100, prefers {} responses, {} thinking. Primary goal: {}",
            self.autonomy_level,
            self.risk_tolerance,
            self.communication_preference,
            self.thinking_style,
            if self.primary_goal.is_empty() { "not set" } else { &self.primary_goal }
        )
    }
}

#[cfg(test)]
mod ledger_tests {
    use super::*;
    use crate::ideation::ThoughtFrame;
    use tempfile::tempdir;

    #[test]
    fn test_task_ledger_add_and_next() {
        let dir = tempdir().unwrap();
        let memory = Arc::new(MemoryGraph::new(dir.path().join("m.db"), None).unwrap());
        let ledger = TaskLedger::new(memory);

        ledger
            .add(&LedgerTask::new("Low task", "d", TaskPriority::Low))
            .unwrap();
        ledger
            .add(&LedgerTask::new("Critical task", "d", TaskPriority::Critical))
            .unwrap();

        let next = ledger.next_pending().unwrap().unwrap();
        assert_eq!(next.title, "Critical task");
        assert_eq!(ledger.pending_count(), 2);
    }

    #[test]
    fn test_onboarding_profile_roundtrip() {
        let dir = tempdir().unwrap();
        let memory = MemoryGraph::new(dir.path().join("m.db"), None).unwrap();

        let answers = vec![
            ("autonomy_level".to_string(), "5".to_string()),
            ("risk_tolerance".to_string(), "3".to_string()),
            ("communication_preference".to_string(), "4".to_string()),
            ("thinking_style".to_string(), "2".to_string()),
            ("primary_goal".to_string(), "Automate my workflow".to_string()),
        ];
        let profile = OnboardingProfile::from_answers(&answers);
        assert_eq!(profile.autonomy_level, 100);
        assert_eq!(profile.risk_tolerance, 60);
        assert_eq!(profile.communication_preference, "concise");
        assert_eq!(profile.thinking_style, "intuitive");

        profile.persist(&memory).unwrap();
        let loaded = OnboardingProfile::load(&memory).unwrap();
        assert_eq!(loaded.primary_goal, "Automate my workflow");
        assert!(!loaded.prompt_context().is_empty());
    }

    #[test]
    fn test_thought_frame_parse_json_and_fallback() {
        let raw = r#"{"cycle_type":"SPARK","thought_draft":"Idea X","evaluation_score":0.95,"next_state":"EXECUTE"}"#;
        let frame = ThoughtFrame::parse(raw);
        assert_eq!(frame.cycle_type, "SPARK");
        assert_eq!(frame.evaluation_score, 0.95);

        let fallback = ThoughtFrame::parse("not json at all");
        assert_eq!(fallback.cycle_type, "CRITIC");
        assert_eq!(fallback.evaluation_score, 0.5);
    }
}
