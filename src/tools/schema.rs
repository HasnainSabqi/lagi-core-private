//! Strict model-facing tool catalog and action-call parser.
//!
//! LAGI models emit tool calls as a single fenced block:
//! ```text
//! ```action
//! {"tool": "write_file", "args": {"folder": "desktop/hello", "file_name": "note.txt", "content": "hello"}}
//! ```
//! ```
//!
//! Unknown tools and extra argument fields are rejected before execution.

use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::Value;

/// Maximum number of autonomous tool actions allowed in one user turn.
pub const MAX_REACT_STEPS: usize = 5;

/// One strictly typed tool schema in the model-facing catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolSchema {
    pub name: &'static str,
    pub description: &'static str,
    pub required_args: &'static [&'static str],
    pub optional_args: &'static [&'static str],
}

/// The complete catalog exposed to the model.
pub const TOOL_CATALOG: &[ToolSchema] = &[
    ToolSchema {
        name: "create_folder",
        description: "Create a folder inside Desktop, Downloads, or Documents.",
        required_args: &["folder", "name"],
        optional_args: &[],
    },
    ToolSchema {
        name: "write_file",
        description: "Write UTF-8 text to a file in a safe folder.",
        required_args: &["folder", "file_name", "content"],
        optional_args: &[],
    },
    ToolSchema {
        name: "web_search",
        description: "Search the web and return titles, URLs, and snippets.",
        required_args: &["query"],
        optional_args: &[],
    },
    ToolSchema {
        name: "open_browser",
        description: "Open an http(s) URL in the default browser.",
        required_args: &["url"],
        optional_args: &[],
    },
    ToolSchema {
        name: "read_file",
        description: "Read a UTF-8 text file inside a safe user directory.",
        required_args: &["path"],
        optional_args: &[],
    },
];

/// Dense model-facing tool schema and ReAct protocol.
pub const TOOL_CATALOG_PROMPT: &str = r#"Tools:
create_folder(folder,name)
write_file(folder,file_name,content)
web_search(query)
open_browser(url)
read_file(path)
Folders/files/paths must use desktop,downloads,documents or their subfolders; URLs use http(s).
For each action emit only:
{"tool":"name","args":{"key":"value"}}
Then wait for [Observation: ...]. One action per response. No filler."#;

/// The parsed contents of one model action block.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionCall {
    pub tool: String,
    pub args: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawActionCall {
    tool: String,
    args: Value,
}

/// Extract the first fenced action block from a model response.
///
/// `action` is the canonical fence name. `json`, `tool_call`, and `tool-call`
/// are also accepted because smaller models commonly use those names. An
/// unlabeled fence is accepted as well. Fences that do not contain an action
/// object are ignored.
pub fn extract_action_block(response: &str) -> Option<&str> {
    let mut remaining = response;
    while let Some(start) = remaining.find("```") {
        let after_fence = &remaining[start + 3..];
        let line_end = after_fence.find('\n').unwrap_or(after_fence.len());
        let language = after_fence[..line_end].trim();
        let body_start = (start + 3 + line_end).min(remaining.len());
        let body = &remaining[body_start..];
        let Some(end) = body.find("```") else {
            // Streaming may stop as soon as the action's closing brace appears,
            // before the model emits the closing fence.
            let candidate = body.trim();
            if (language.eq_ignore_ascii_case("action") || is_flexible_action_fence(language))
                && looks_like_action(candidate)
            {
                return Some(candidate);
            }
            return None;
        };
        let candidate = body[..end].trim();
        if language.eq_ignore_ascii_case("action") {
            return Some(candidate);
        }
        if is_flexible_action_fence(language) && looks_like_action(candidate) {
            return Some(candidate);
        }
        remaining = &body[end + 3..];
    }
    None
}

fn is_flexible_action_fence(language: &str) -> bool {
    language.is_empty()
        || language.eq_ignore_ascii_case("json")
        || language.eq_ignore_ascii_case("tool_call")
        || language.eq_ignore_ascii_case("tool-call")
}

fn looks_like_action(candidate: &str) -> bool {
    serde_json::from_str::<Value>(candidate)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .map(|object| object.contains_key("tool") && object.contains_key("args"))
        .unwrap_or(false)
}

/// Parse and strictly validate one action block.
///
/// Both fenced and bare JSON action objects are supported. Strict validation
/// of the tool name and its arguments is unchanged.
pub fn parse_action_call(response: &str) -> Result<Option<ActionCall>> {
    let block = extract_action_block(response).or_else(|| {
        let candidate = response.trim();
        if looks_like_action(candidate) {
            Some(candidate)
        } else {
            None
        }
    });
    let Some(block) = block else {
        return Ok(None);
    };
    let raw: RawActionCall =
        serde_json::from_str(block).map_err(|err| anyhow!("invalid action JSON: {}", err))?;
    validate_action_call(&raw.tool, &raw.args)?;
    Ok(Some(ActionCall {
        tool: raw.tool,
        args: raw.args,
    }))
}

/// Return true when the text contains a complete, valid action object.
///
/// This is used by streaming inference to stop as soon as an action becomes
/// executable. It accepts bare JSON and action-shaped fenced JSON.
pub fn has_complete_action(text: &str) -> bool {
    let trimmed = text.trim();
    if looks_like_action(trimmed) {
        if let Ok(raw) = serde_json::from_str::<RawActionCall>(trimmed) {
            if validate_action_call(&raw.tool, &raw.args).is_ok() {
                return true;
            }
        }
    }

    let mut remaining = text;
    while let Some(start) = remaining.find('{') {
        let after = &remaining[start..];
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        let mut end = None;
        for (index, ch) in after.char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            match ch {
                '"' => in_string = true,
                '{' => depth += 1,
                '}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        end = Some(index + ch.len_utf8());
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(end) = end {
            let candidate = &after[..end];
            if looks_like_action(candidate) {
                if let Ok(raw) = serde_json::from_str::<RawActionCall>(candidate) {
                    if validate_action_call(&raw.tool, &raw.args).is_ok() {
                        return true;
                    }
                }
            }
            remaining = &after[end..];
        } else {
            break;
        }
    }
    false
}

/// Validate a tool name and its exact argument object.
pub fn validate_action_call(tool: &str, args: &Value) -> Result<()> {
    let schema = TOOL_CATALOG
        .iter()
        .find(|schema| schema.name == tool)
        .ok_or_else(|| anyhow!("unknown tool '{}'", tool))?;
    let object = args
        .as_object()
        .ok_or_else(|| anyhow!("args for '{}' must be a JSON object", tool))?;

    for required in schema.required_args {
        let value = object
            .get(*required)
            .ok_or_else(|| anyhow!("tool '{}' requires argument '{}'", tool, required))?;
        require_string(tool, required, value)?;
    }
    for optional in schema.optional_args {
        if let Some(value) = object.get(*optional) {
            require_string(tool, optional, value)?;
        }
    }

    let allowed: std::collections::HashSet<&str> = schema
        .required_args
        .iter()
        .chain(schema.optional_args.iter())
        .copied()
        .collect();
    if let Some(extra) = object.keys().find(|key| !allowed.contains(key.as_str())) {
        return Err(anyhow!("unknown argument '{}' for tool '{}'", extra, tool));
    }
    Ok(())
}

fn require_string(tool: &str, argument: &str, value: &Value) -> Result<()> {
    if value.is_string() {
        Ok(())
    } else {
        Err(anyhow!(
            "argument '{}' for tool '{}' must be a string",
            argument,
            tool
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_valid_action_block() {
        let response = "working\n```action\n{\"tool\":\"create_folder\",\"args\":{\"folder\":\"desktop\",\"name\":\"hello\"}}\n```";
        let parsed = parse_action_call(response).unwrap().unwrap();
        assert_eq!(parsed.tool, "create_folder");
        assert_eq!(parsed.args["name"], json!("hello"));
    }

    #[test]
    fn ignores_non_action_fences() {
        let response = "```json\n{\"message\":\"not an action\"}\n```";
        assert!(parse_action_call(response).unwrap().is_none());
    }

    #[test]
    fn parses_bare_json_action() {
        let response =
            "{\"tool\":\"create_folder\",\"args\":{\"folder\":\"desktop\",\"name\":\"hello\"}}";
        let parsed = parse_action_call(response).unwrap().unwrap();
        assert_eq!(parsed.tool, "create_folder");
        assert_eq!(parsed.args["name"], json!("hello"));
    }

    #[test]
    fn parses_flexible_action_fences() {
        for language in ["json", "tool_call", "tool-call", ""] {
            let response = format!(
                "```{}\n{{\"tool\":\"create_folder\",\"args\":{{\"folder\":\"desktop\",\"name\":\"hello\"}}}}\n```",
                language
            );
            let parsed = parse_action_call(&response).unwrap().unwrap();
            assert_eq!(
                parsed.tool, "create_folder",
                "failed for fence: {}",
                language
            );
        }
    }

    #[test]
    fn ignores_bare_non_action_json() {
        let response = "{\"message\":\"hello\"}";
        assert!(parse_action_call(response).unwrap().is_none());
    }

    #[test]
    fn detects_complete_action_for_streaming_early_exit() {
        assert!(has_complete_action(
            "{\"tool\":\"create_folder\",\"args\":{\"folder\":\"desktop\",\"name\":\"hello\"}}"
        ));
        assert!(has_complete_action(
            "```action\n{\"tool\":\"create_folder\",\"args\":{\"folder\":\"desktop\",\"name\":\"hello\"}}"
        ));
        assert!(!has_complete_action(
            "{\"tool\":\"create_folder\",\"args\":{\"folder\":\"desktop\""
        ));
        assert!(!has_complete_action("{\"message\":\"hello\"}"));
    }

    #[test]
    fn parses_unterminated_streaming_action_fence() {
        let response = "```action\n{\"tool\":\"create_folder\",\"args\":{\"folder\":\"desktop\",\"name\":\"hello\"}}";
        let parsed = parse_action_call(response).unwrap().unwrap();
        assert_eq!(parsed.tool, "create_folder");
    }

    #[test]
    fn rejects_unknown_tool() {
        let response = "```action\n{\"tool\":\"delete_everything\",\"args\":{}}\n```";
        assert!(parse_action_call(response).is_err());
    }

    #[test]
    fn rejects_missing_and_extra_arguments() {
        let missing = "```action\n{\"tool\":\"write_file\",\"args\":{\"folder\":\"desktop\",\"file_name\":\"a.txt\"}}\n```";
        let extra =
            "```action\n{\"tool\":\"web_search\",\"args\":{\"query\":\"gold\",\"limit\":3}}\n```";
        assert!(parse_action_call(missing).is_err());
        assert!(parse_action_call(extra).is_err());
    }

    #[test]
    fn rejects_non_string_arguments() {
        let response = "```action\n{\"tool\":\"web_search\",\"args\":{\"query\":42}}\n```";
        assert!(parse_action_call(response).is_err());
    }
}
