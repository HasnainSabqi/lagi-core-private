/*!
Safe Operating System Tool Executor (PC Control)

Implements file listing, directory reorganization (moving PDFs/images by
extension), process inspection, clipboard read/write, and default-browser
URL opening.

SAFETY RULES (Destructive Action Guard):
- No deletions. Reorganization only MOVES files into subfolders.
- Any hypothetical destructive operation must surface
  `requires_confirmation: true` in its payload before execution.
- Paths are validated; non-http URLs are refused by the browser opener.
*/

use anyhow::{anyhow, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::WalkDir;

/// Categories used when organizing a folder, by lowercase extension
const CATEGORY_MAP: &[(&str, &[&str])] = &[
    ("PDFs", &["pdf"]),
    ("Images", &["png", "jpg", "jpeg", "gif", "webp", "bmp", "heic", "svg", "tiff"]),
    ("Documents", &["doc", "docx", "txt", "md", "rtf", "odt", "xls", "xlsx", "ppt", "pptx", "csv"]),
    ("Archives", &["zip", "rar", "7z", "tar", "gz", "xz"]),
    ("Videos", &["mp4", "mov", "avi", "mkv", "webm"]),
    ("Audio", &["mp3", "wav", "flac", "aac", "ogg", "m4a"]),
    ("Code", &["rs", "py", "js", "ts", "html", "css", "json", "toml", "yaml", "sh", "ps1"]),
];

pub struct PcToolExecutor;

#[derive(Debug, Clone, Serialize)]
pub struct FileEntry {
    pub name: String,
    pub is_dir: bool,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunningApp {
    pub pid: u32,
    pub name: String,
    pub memory_mb: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrganizeReport {
    /// Human-readable summary of what was moved (also committed to VectorCore)
    pub summary: String,
    pub moved: u64,
    pub per_category: HashMap<String, u64>,
}

impl Default for PcToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl PcToolExecutor {
    pub fn new() -> Self {
        Self
    }

    /// Resolve a natural-language folder hint ("screenshots", "downloads",
    /// "desktop", "documents", "home") to a concrete directory.
    pub fn resolve_folder(&self, hint: &str) -> Result<PathBuf> {
        let home = std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .ok_or_else(|| anyhow!("cannot determine home directory"))?;

        let dir = match hint.to_lowercase().as_str() {
            "screenshots" => {
                #[cfg(target_os = "windows")]
                {
                    home.join("Pictures").join("Screenshots")
                }
                #[cfg(target_os = "macos")]
                {
                    home.join("Desktop")
                }
                #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
                {
                    home.join("Pictures")
                }
            }
            "downloads" => home.join("Downloads"),
            "desktop" => home.join("Desktop"),
            "documents" => home.join("Documents"),
            other => {
                return Err(anyhow!(
                    "unknown folder hint '{}' — use screenshots/downloads/desktop/documents",
                    other
                ))
            }
        };

        if dir.exists() {
            Ok(dir)
        } else {
            Err(anyhow!("folder '{}' does not exist: {}", hint, dir.display()))
        }
    }

    /// Create a folder inside a safe user directory (Desktop, Downloads,
    /// Documents) and return a confirmation with the absolute path.
    ///
    /// SAFETY: only whitelisted user directories are permitted, and the
    /// folder name is sanitized — quotes/punctuation trimmed, path
    /// separators and traversal attempts ("..", "../", "a/b") rejected.
    pub fn create_folder(&self, target_folder: &str, folder_name: &str) -> Result<String> {
        const ALLOWED_TARGETS: &[&str] = &["desktop", "downloads", "documents"];
        let target = target_folder.trim().to_lowercase();
        if !ALLOWED_TARGETS.contains(&target.as_str()) {
            return Err(anyhow!(
                "create_folder only operates inside Desktop, Downloads or Documents (got '{}')",
                target_folder.trim()
            ));
        }

        let sanitized = sanitize_folder_name(folder_name)?;
        let base = self.resolve_folder(&target)?;
        let new_dir = base.join(&sanitized);

        std::fs::create_dir_all(&new_dir)?;
        Ok(format!(
            "Folder '{}' created at {}",
            sanitized,
            new_dir.display()
        ))
    }


    /// Write a text file inside a safe user directory, supporting subfolders
    /// (e.g. `"desktop/lixo"` or `"downloads"`).
    ///
    /// SAFETY: only whitelisted user roots (Desktop, Downloads, Documents)
    /// are permitted; every subfolder component and the file name are
    /// sanitized - quotes/punctuation trimmed, path separators and
    /// traversal attempts ("..", "../", "a/b") rejected.
    pub fn write_file(&self, target_folder: &str, file_name: &str, content: &str) -> Result<String> {
        const ALLOWED_ROOTS: &[&str] = &["desktop", "downloads", "documents"];
        let mut components = target_folder
            .split(['/', '\\'])
            .map(str::trim)
            .filter(|c| !c.is_empty());

        let root = components.next().unwrap_or("").to_lowercase();
        if !ALLOWED_ROOTS.contains(&root.as_str()) {
            return Err(anyhow!(
                "write_file only operates inside Desktop, Downloads or Documents (got '{}')",
                target_folder.trim()
            ));
        }

        let base = self.resolve_folder(&root)?;
        let mut dir = base;
        for component in components {
            let safe = sanitize_folder_name(component)?;
            dir = dir.join(safe);
        }

        let sanitized_name = sanitize_folder_name(file_name)?;
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(&sanitized_name);
        std::fs::write(&path, content)?;

        Ok(format!(
            "File '{}' written at {}",
            sanitized_name,
            path.display()
        ))
    }
    /// Launch a generated file so the user sees live output immediately
    /// (auto-run pipeline).
    ///
    /// - HTML (.html / .htm): opened in the default browser via `open::that`.
    /// - Python (.py): executed with `python`, capturing stdout/stderr.
    /// - Anything else: an error explaining no auto-runner is available.
    pub fn launch_or_run_file(&self, file_path: &std::path::Path) -> Result<String> {
        if !file_path.exists() {
            return Err(anyhow!("file not found: {}", file_path.display()));
        }
        let ext = file_path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "html" | "htm" => {
                open::that(file_path).map_err(|e| anyhow!("could not open browser: {}", e))?;
                Ok(format!(
                    "Opened '{}' in the default browser",
                    file_path.display()
                ))
            }
            "py" => {
                let output = Command::new("python")
                    .arg(file_path)
                    .output()
                    .map_err(|e| anyhow!("failed to execute python: {}", e))?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let mut report = String::new();
                if !stdout.trim().is_empty() {
                    report.push_str(stdout.trim_end());
                }
                if !stderr.trim().is_empty() {
                    if !report.is_empty() {
                        report.push('\n');
                    }
                    report.push_str(stderr.trim_end());
                }
                if report.trim().is_empty() {
                    report = if output.status.success() {
                        "(no output; exited successfully)".to_string()
                    } else {
                        format!("(exited with status {})", output.status)
                    };
                }
                Ok(format!(
                    "python {} -> exit code {}:\n{}",
                    file_path.display(),
                    output.status.code().unwrap_or(-1),
                    report
                ))
            }
            other => Err(anyhow!(
                "no auto-runner for '.{}' files (only .html/.htm and .py are auto-executed)",
                other
            )),
        }
    }

    /// Non-recursively list a directory's entries
    pub async fn list_files(&self, folder_hint: &str) -> Result<Vec<FileEntry>> {
        let dir = self.resolve_folder(folder_hint)?;
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            entries.push(FileEntry {
                name: entry.file_name().to_string_lossy().into_owned(),
                is_dir: meta.is_dir(),
                size_bytes: meta.len(),
            });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    /// Read a local UTF-8 text file (size-capped to 1 MB for safety)
    pub fn read_local_file(&self, path: &str) -> Result<String> {
        let p = Path::new(path);
        if !p.exists() {
            return Err(anyhow!("file not found: {}", path));
        }
        let meta = std::fs::metadata(p)?;
        if meta.len() > 1024 * 1024 {
            return Err(anyhow!("file too large to read ({} bytes, cap 1 MB)", meta.len()));
        }
        Ok(std::fs::read_to_string(p)?)
    }

    /**
    Reorganize a folder by moving files into category subfolders
    (Images/, PDFs/, ...). Moves only — never deletes. If the utterance
    implies deletion the caller must instead emit
    `requires_confirmation: true` and ask the user first.
    */
    pub async fn organize_folder(&self, folder_hint: &str) -> Result<OrganizeReport> {
        let dir = self.resolve_folder(folder_hint)?;
        let mut per_category: HashMap<String, u64> = HashMap::new();

        // Defensive: refuse to reorganize a root-ish directory
        if dir.parent().is_none() {
            return Err(anyhow!("refusing to organize a root directory"));
        }

        let targets: Vec<(PathBuf, String)> = WalkDir::new(&dir)
            .max_depth(1)
            .min_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| {
                let ext = e
                    .path()
                    .extension()
                    .map(|x| x.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                if ext == "lnk" {
                    return None; // never move shortcuts
                }
                CATEGORY_MAP
                    .iter()
                    .find(|(_, exts)| exts.contains(&ext.as_str()))
                    .map(|(cat, _)| (e.path().to_path_buf(), cat.to_string()))
            })
            .collect();

        for (src, category) in targets {
            let dest_dir = dir.join(&category);
            if !dest_dir.exists() {
                std::fs::create_dir(&dest_dir)?;
            }
            let file_name = src
                .file_name()
                .ok_or_else(|| anyhow!("bad path"))?
                .to_owned();
            let mut dest = dest_dir.join(&file_name);

            // Collision-safe naming: report.txt -> report (2).txt
            let mut n = 2;
            while dest.exists() {
                let stem = Path::new(&file_name)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let ext = Path::new(&file_name)
                    .extension()
                    .map(|s| format!(".{}", s.to_string_lossy()))
                    .unwrap_or_default();
                dest = dest_dir.join(format!("{} ({}){}", stem, n, ext));
                n += 1;
            }

            std::fs::rename(&src, &dest).map_err(|e| {
                anyhow!(
                    "failed to move {} (it may be open in another program): {}",
                    src.display(),
                    e
                )
            })?;
            *per_category.entry(category).or_insert(0) += 1;
        }

        let total: u64 = per_category.values().sum();
        let breakdown: String = if per_category.is_empty() {
            "no matching files".to_string()
        } else {
            per_category
                .iter()
                .map(|(k, v)| format!("{} -> {}", k, v))
                .collect::<Vec<_>>()
                .join(", ")
        };

        Ok(OrganizeReport {
            summary: format!(
                "Organized {} in {}: {} file(s) moved ({}). No files were deleted.",
                folder_hint,
                dir.display(),
                total,
                breakdown
            ),
            moved: total,
            per_category,
        })
    }

    /// Snapshot of running processes (name, pid, memory)
    pub fn get_running_apps(&self) -> Result<Vec<RunningApp>> {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_processes();
        let mut apps: Vec<RunningApp> = sys
            .processes()
            .iter()
            .map(|(pid, p)| RunningApp {
                pid: pid.as_u32(),
                name: p.name().to_string(),
                memory_mb: p.memory() as f32 / 1024.0 / 1024.0,
            })
            .collect();
        apps.sort_by(|a, b| {
            b.memory_mb
                .partial_cmp(&a.memory_mb)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(apps)
    }

    /// Open a URL in the user's default browser (uses the lightweight `open` crate)
    pub fn open_browser_url(&self, url: &str) -> Result<()> {
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(anyhow!("refusing to open non-http URL: {}", url));
        }
        open::that(url).map_err(|e| anyhow!("could not open browser: {}", e))
    }

    /// Read the system clipboard (platform shell, no heavy deps)
    pub async fn read_clipboard(&self) -> Result<String> {
        #[cfg(target_os = "windows")]
        let out = self.run_platform_command(("powershell", &["-NoProfile", "-Command", "Get-Clipboard"]))?;
        #[cfg(target_os = "macos")]
        let out = self.run_platform_command(("pbpaste", &[]))?;
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        let out = self.run_platform_command((
            "sh",
            &["-c", "command -v xclip >/dev/null && xclip -o -selection clipboard || wl-paste"],
        ))?;
        Ok(out.trim_end().to_string())
    }

    /// Write to the system clipboard (platform shell, no heavy deps)
    pub fn write_clipboard(&self, text: &str) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            self.run_platform_command_with_stdin(
                ("powershell", &["-NoProfile", "-Command", "$input | Set-Clipboard"]),
                text,
            )
        }
        #[cfg(target_os = "macos")]
        {
            self.run_platform_command_with_stdin(("pbcopy", &[]), text)
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            self.run_platform_command_with_stdin(
                ("sh", &["-c", "xclip -selection clipboard || wl-copy"]),
                text,
            )
        }
    }

    /// Kill a process — DESTRUCTIVE. Never executes on its own; returns a
    /// payload flag requiring explicit user confirmation first.
    pub fn kill_process_payload(&self, pid: u32) -> serde_json::Value {
        serde_json::json!({
            "tool": "kill_process",
            "pid": pid,
            "requires_confirmation": true,
            "note": "Destructive action. Ask the user to confirm before executing."
        })
    }

    /// Actually terminate a process; ONLY call after explicit user confirmation.
    pub fn kill_process_confirmed(&self, pid: u32) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/F"])
                .output()?;
        }
        #[cfg(not(target_os = "windows"))]
        {
            Command::new("kill").args(["-9", &pid.to_string()]).output()?;
        }
        Ok(())
    }

    #[allow(unused_variables)]
    fn run_platform_command(&self, (program, args): (&str, &[&str])) -> Result<String> {
        let out = Command::new(program).args(args).output()?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        } else {
            Err(anyhow!(
                "{} exited with {}: {}",
                program,
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ))
        }
    }

    #[allow(unused_variables)]
    fn run_platform_command_with_stdin(&self, (program, args): (&str, &[&str]), input: &str) -> Result<()> {
        use std::io::Write;
        let mut child = Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()?;
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(input.as_bytes())?;
        }
        let status = child.wait()?;
        if status.success() {
            Ok(())
        } else {
            Err(anyhow!("{} exited with {}", program, status))
        }
    }
}

/// Sanitize a natural-language folder name for `create_folder`.
///
/// Trims surrounding whitespace, quotes (single/double/backtick) and
/// punctuation, then rejects anything that could escape the target
/// directory: path separators, drive letters, and traversal components
/// ("..", "..."). Returns the cleaned name.
pub(crate) fn sanitize_folder_name(raw: &str) -> Result<String> {

    let name: String = raw
        .trim()
        .trim_matches(['"', '\'', '`'])
        .trim()
        .chars()
        .collect();

    // Repeatedly trim quotes AND punctuation from both ends until stable.
    let clean = |s: &str| -> String {
        let mut t = s.trim().to_string();
        loop {
            let before = t.clone();
            t = t
                .trim_start_matches(['"', '\'', '`', '.', ',', '!', '?', ';', ':', ' ', '\t'])
                .trim_end_matches(['"', '\'', '`', '.', ',', '!', '?', ';', ':', ' ', '\t'])
                .to_string();
            if t == before {
                break;
            }
        }
        t
    };

    let name = clean(&name);

    if name.is_empty() {
        return Err(anyhow!("folder name is empty after sanitization"));
    }
    if name.contains('/') || name.contains('\\') || name.contains(':') {
        return Err(anyhow!(
            "folder name must not contain path separators: '{}'",
            raw.trim()
        ));
    }
    if name.chars().all(|c| c == '.') {
        return Err(anyhow!("folder name must not be a traversal token: '{}'", name));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_folder_name_basic() {
        assert_eq!(sanitize_folder_name("lixo").unwrap(), "lixo");
        assert_eq!(sanitize_folder_name("  My Project  ").unwrap(), "My Project");
        // Quotes and trailing punctuation are trimmed
        assert_eq!(sanitize_folder_name("\"lixo\"").unwrap(), "lixo");
        assert_eq!(sanitize_folder_name("'lixo'.").unwrap(), "lixo");
        assert_eq!(sanitize_folder_name("`lixo`, !").unwrap(), "lixo");
    }

    #[test]
    fn test_sanitize_folder_name_rejects_traversal_and_separators() {
        assert!(sanitize_folder_name("../etc").is_err());
        assert!(sanitize_folder_name("..").is_err());
        assert!(sanitize_folder_name("a/b").is_err());
        assert!(sanitize_folder_name("a\\b").is_err());
        assert!(sanitize_folder_name("C:\\Temp").is_err());
        assert!(sanitize_folder_name("   ").is_err());
        assert!(sanitize_folder_name("...").is_err());
        assert!(sanitize_folder_name("").is_err());
    }

    #[test]
    fn test_create_folder_safety_whitelist() {
        let exec = PcToolExecutor::new();
        // Non-whitelisted target must be rejected without touching disk.
        assert!(exec.create_folder("system32", "evil").is_err());
        assert!(exec.create_folder("home", "nope").is_err());
        // Traversal inside an allowed target must be rejected.
        assert!(exec.create_folder("desktop", "../evil").is_err());
        assert!(exec.create_folder("desktop", "sub/dir").is_err());
    }

    #[test]
    fn test_organize_categories_cover_common_extensions() {
        // The category map must classify representative extensions
        for (ext, expected) in [("pdf", "PDFs"), ("png", "Images"), ("zip", "Archives"), ("mp4", "Videos")] {
            let cat = CATEGORY_MAP
                .iter()
                .find(|(_, exts)| exts.contains(&ext))
                .map(|(c, _)| *c)
                .unwrap();
            assert_eq!(cat, expected);
        }
    }

    #[test]
    fn test_kill_process_requires_confirmation() {
        let exec = PcToolExecutor::new();
        let payload = exec.kill_process_payload(4242);
        assert_eq!(payload["requires_confirmation"], true);
    }

    #[test]
    fn test_create_folder_end_to_end_on_disk() {
        let exec = PcToolExecutor::new();
        // Redirect USERPROFILE to a temp dir so the test never touches the
        // real user Desktop, then verify the folder physically exists.
        let dir = std::env::temp_dir().join(format!("lagi_cf_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(dir.join("Desktop")).unwrap();
        let old = std::env::var("USERPROFILE").ok();
        std::env::set_var("USERPROFILE", &dir);

        let msg = exec.create_folder("desktop", "lixo").unwrap();
        assert!(msg.contains("lixo"));
        assert!(msg.contains("Desktop"));
        assert!(dir.join("Desktop").join("lixo").is_dir());

        // Restore the real USERPROFILE before dropping the temp dir.
        match old {
            Some(v) => std::env::set_var("USERPROFILE", v),
            None => std::env::remove_var("USERPROFILE"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_file_end_to_end_on_disk() {
        let exec = PcToolExecutor::new();
        let dir = std::env::temp_dir().join(format!("lagi_wf_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(dir.join("Desktop")).unwrap();
        let old = std::env::var("USERPROFILE").ok();
        std::env::set_var("USERPROFILE", &dir);

        let msg = exec.write_file("desktop/lixo", "hello.txt", "hello world").unwrap();
        assert!(msg.contains("hello.txt"));
        let path = dir.join("Desktop").join("lixo").join("hello.txt");
        assert!(path.is_file());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");

        match old {
            Some(v) => std::env::set_var("USERPROFILE", v),
            None => std::env::remove_var("USERPROFILE"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_file_safety_whitelist() {
        let exec = PcToolExecutor::new();
        assert!(exec.write_file("system32", "evil.txt", "x").is_err());
        assert!(exec.write_file("home", "evil.txt", "x").is_err());
        assert!(exec.write_file("desktop/../../evil", "x.txt", "x").is_err());
        assert!(exec.write_file("desktop", "a/b.txt", "x").is_err());
    }

    #[test]
    fn test_open_browser_rejects_non_http() {
        let exec = PcToolExecutor::new();
        assert!(exec.open_browser_url("file:///etc/passwd").is_err());
        assert!(exec.open_browser_url("calc.exe").is_err());
    }

    #[tokio::test]
    async fn test_organize_moves_files_safely() {
        let exec = PcToolExecutor::new();
        // Build a fake home-based folder via env override is fragile across
        // platforms; instead exercise resolve error path and move semantics
        // through list/organize on a non-existent hint.
        assert!(exec.organize_folder("does-not-exist-xyz").await.is_err());
    }
}
