/*!
Zero-API-Key Web Search & Page Reader

Queries the DuckDuckGo HTML Lite endpoint with a native reqwest client
(rustls, no native TLS, no browser binaries — zero-bloat guarantee), parses
the returned HTML with regexes, and returns clean title/url/snippet triples.
A readability-style stripper converts arbitrary HTML into plain-text
markdown for the `read_webpage` tool.

All results are committed as episodic records to Liumi VectorCore by the
caller (see `tools::ToolDispatcher`).
*/

use anyhow::{anyhow, Result};
use regex::Regex;
use std::time::Duration;

const USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

/// One clean search result
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebSnippet {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| anyhow!("http client: {}", e))
}

/// Decode the HTML entities DDG uses (&, &#x27;, ...)
fn unescape_entities(s: &str) -> String {
    let amp = ['&', 'a', 'm', 'p', ';'].iter().collect::<String>();
    let lt = ['&', 'l', 't', ';'].iter().collect::<String>();
    let gt = ['&', 'g', 't', ';'].iter().collect::<String>();
    let quot = ['&', 'q', 'u', 'o', 't', ';'].iter().collect::<String>();
    let apos27 = ['&', '#', 'x', '2', '7', ';'].iter().collect::<String>();
    let apos39 = ['&', '#', '3', '9', ';'].iter().collect::<String>();
    let nbsp = ['&', 'n', 'b', 's', 'p', ';'].iter().collect::<String>();
    s.replace(&amp, "&")
        .replace(&lt, "<")
        .replace(&gt, ">")
        .replace(&quot, "\"")
        .replace(&apos27, "'")
        .replace(&apos39, "'")
        .replace(&nbsp, " ")
}

/// Strip all tags and collapse whitespace
fn strip_tags(html: &str) -> String {
    let tag_re = Regex::new(r"(?is)<(script|style)[^>]*>.*?</(script|style)>").unwrap();
    let any_tag = Regex::new(r"<[^>]+>").unwrap();
    let ws = Regex::new(r"\s+").unwrap();
    let cleaned = tag_re.replace_all(html, " ");
    let text = any_tag.replace_all(&cleaned, " ");
    ws.replace_all(&text, " ").trim().to_string()
}

/// Search DuckDuckGo HTML Lite and return 3-5 clean snippets.
pub async fn search_web(query: &str) -> Result<Vec<WebSnippet>> {
    if query.trim().is_empty() {
        return Err(anyhow!("empty search query"));
    }

    let client = http_client()?;
    let body = client
        .post("https://html.duckduckgo.com/html/")
        .header("Referer", "https://html.duckduckgo.com/")
        .form(&[("q", query.trim())])
        .send()
        .await
        .map_err(|e| anyhow!("search request failed: {}", e))?
        .error_for_status()
        .map_err(|e| anyhow!("search returned {}", e))?
        .text()
        .await
        .map_err(|e| anyhow!("failed to read search body: {}", e))?;

    parse_ddg_html(&body)
}

/// Parse DDG HTML Lite result blocks: `<a class="result__a" href="...">Title</a>`
/// followed by `<a class="result__snippet">Snippet</a>`.
fn parse_ddg_html(body: &str) -> Result<Vec<WebSnippet>> {
    let link_re = Regex::new(r#"(?is)<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#)
        .expect("static regex");
    let snip_re =
        Regex::new(r#"(?is)<a[^>]*class="result__snippet"[^>]*>(.*?)</a>"#).expect("static regex");

    let mut results: Vec<WebSnippet> = Vec::new();
    let mut snippets = snip_re.captures_iter(body);

    for cap in link_re.captures_iter(body) {
        let raw_url = cap.get(1).map(|m| m.as_str()).unwrap_or_default();
        let title = strip_tags(&unescape_entities(
            cap.get(2).map(|m| m.as_str()).unwrap_or_default(),
        ));

        // DDG wraps URLs in a redirect: //duckduckgo.com/l/?uddg=<encoded>&rut=...
        let url = if let Some(idx) = raw_url.find("uddg=") {
            let rest = &raw_url[idx + 5..];
            let end = rest.find('&').unwrap_or(rest.len());
            percent_decode(&rest[..end])
        } else if raw_url.starts_with("//") {
            format!("https:{}", raw_url)
        } else {
            raw_url.to_string()
        };

        let snippet = snippets.next().map(|s| {
            strip_tags(&unescape_entities(
                s.get(1).map(|m| m.as_str()).unwrap_or_default(),
            ))
        });

        if title.is_empty() || url.is_empty() {
            continue;
        }

        results.push(WebSnippet {
            title,
            url,
            snippet: snippet.unwrap_or_default(),
        });

        if results.len() >= 5 {
            break;
        }
    }

    if results.is_empty() {
        return Err(anyhow!("no results parsed (rate limit or markup change)"));
    }
    Ok(results)
}

/// Minimal percent-decoder for DDG redirect URLs
fn percent_decode(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Fetch a URL and return readability-style plain text (markdown-ish).
/// Keeps paragraph breaks, drops scripts/styles/nav noise, caps at ~32 KB.
pub async fn read_webpage(url: &str) -> Result<String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(anyhow!("only http(s) URLs are supported"));
    }

    let client = http_client()?;
    let html = client
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow!("fetch failed: {}", e))?
        .error_for_status()
        .map_err(|e| anyhow!("server returned {}", e))?
        .text()
        .await
        .map_err(|e| anyhow!("failed to decode body: {}", e))?;

    Ok(html_to_markdown(&html))
}

/// Readability-style HTML → markdown converter using regexes only
/// (no scraper / html5ever tree — keeps the binary small).
pub fn html_to_markdown(html: &str) -> String {
    // Drop non-content blocks entirely (regex crate has no backreferences,
    // so enumerate each tag pair explicitly)
    let mut text = html.to_string();
    for tag in [
        "script", "style", "noscript", "head", "nav", "footer", "aside", "svg", "form", "iframe",
    ] {
        let re = Regex::new(&format!(r"(?is)<{}\b[^>]*>.*?</{}>", tag, tag)).expect("static regex");
        text = re.replace_all(&text, " ").to_string();
    }
    let comments = Regex::new(r"(?s)<!--.*?-->").expect("static regex");
    let h1 = Regex::new(r"(?is)<h1[^>]*>(.*?)</h1>").expect("static regex");
    let h2 = Regex::new(r"(?is)<h2[^>]*>(.*?)</h2>").expect("static regex");
    let h3 = Regex::new(r"(?is)<h3[^>]*>(.*?)</h3>").expect("static regex");
    let li = Regex::new(r"(?is)<li[^>]*>(.*?)</li>").expect("static regex");
    let para = Regex::new(r"(?is)<[^>]*(p|div|br|section|article|tr|table|ul|ol|blockquote)[^>]*>")
        .expect("static regex");
    let any_tag = Regex::new(r"<[^>]+>").expect("static regex");
    let ws = Regex::new(r"[ \t]+").expect("static regex");
    let blank = Regex::new(r"\n{3,}").expect("static regex");

    text = comments.replace_all(&text, "").to_string();

    // Headers get markdown prefixes; list items get bullets.
    text = h1
        .replace_all(&text, |c: &regex::Captures| {
            format!(
                "\n# {}\n",
                strip_tags(c.get(1).map(|m| m.as_str()).unwrap_or(""))
            )
        })
        .to_string();
    text = h2
        .replace_all(&text, |c: &regex::Captures| {
            format!(
                "\n## {}\n",
                strip_tags(c.get(1).map(|m| m.as_str()).unwrap_or(""))
            )
        })
        .to_string();
    text = h3
        .replace_all(&text, |c: &regex::Captures| {
            format!(
                "\n### {}\n",
                strip_tags(c.get(1).map(|m| m.as_str()).unwrap_or(""))
            )
        })
        .to_string();
    text = li
        .replace_all(&text, |c: &regex::Captures| {
            format!(
                "\n- {}",
                strip_tags(c.get(1).map(|m| m.as_str()).unwrap_or(""))
            )
        })
        .to_string();

    // Remaining block tags become newlines
    text = para.replace_all(&text, "\n").to_string();
    text = any_tag.replace_all(&text, "").to_string();
    text = unescape_entities(&text);

    // Normalize whitespace while preserving paragraph structure
    let lines: Vec<String> = text
        .lines()
        .map(|l| ws.replace_all(l.trim(), " ").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let joined = blank.replace_all(&lines.join("\n"), "\n\n").to_string();

    // Cap output so the 8K-token context isn't blown
    joined.chars().take(32_000).collect()
}

// ---------------------------------------------------------------------------
// Dual-Engine Failover (Primary: DDG HTML Lite -> Secondary: SearXNG/Wikipedia)
// ---------------------------------------------------------------------------

/// Which engine produced a result set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum SearchEngine {
    DuckDuckGo,
    Searxng,
    Wikipedia,
}

/// One Wikipedia REST search hit
#[derive(Debug, serde::Deserialize)]
struct WikiSearchPage {
    title: String,
    #[serde(default)]
    excerpt: String,
    #[serde(default)]
    key: String,
}

/// Search a self-hosted or public SearXNG instance (configured via the
/// `LAGI_SEARXNG_URL` env var) using its JSON API.
async fn search_searxng(query: &str) -> Result<Vec<WebSnippet>> {
    let base = std::env::var("LAGI_SEARXNG_URL")
        .map_err(|_| anyhow!("LAGI_SEARXNG_URL not configured"))?;
    let client = http_client()?;
    let url = format!("{}/search", base.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .query(&[("q", query), ("format", "json")])
        .send()
        .await
        .map_err(|e| anyhow!("searxng request failed: {}", e))?
        .error_for_status()
        .map_err(|e| anyhow!("searxng returned {}", e))?;
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| anyhow!("searxng bad json: {}", e))?;
    let mut out = Vec::new();
    if let Some(results) = body.get("results").and_then(|r| r.as_array()) {
        for r in results.iter().take(5) {
            out.push(WebSnippet {
                title: r
                    .get("title")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                url: r
                    .get("url")
                    .and_then(|u| u.as_str())
                    .unwrap_or("")
                    .to_string(),
                snippet: r
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string(),
            });
        }
    }
    Ok(out.into_iter().filter(|s| !s.title.is_empty()).collect())
}

/// Search Wikipedia via the keyless REST API (secondary fallback).
async fn search_wikipedia(query: &str) -> Result<Vec<WebSnippet>> {
    let client = http_client()?;
    let resp = client
        .get("https://en.wikipedia.org/w/rest.php/v1/search/page")
        .query(&[("q", query.trim()), ("limit", "5")])
        .send()
        .await
        .map_err(|e| anyhow!("wikipedia request failed: {}", e))?
        .error_for_status()
        .map_err(|e| anyhow!("wikipedia returned {}", e))?;
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| anyhow!("wikipedia bad json: {}", e))?;
    let pages: Vec<WikiSearchPage> = body
        .get("pages")
        .and_then(|p| serde_json::from_value(p.clone()).ok())
        .unwrap_or_default();
    Ok(pages
        .into_iter()
        .map(|p| WebSnippet {
            title: p.title,
            url: format!("https://en.wikipedia.org/wiki/{}", p.key),
            snippet: strip_tags(&unescape_entities(&p.excerpt)),
        })
        .filter(|s| !s.title.is_empty())
        .collect())
}

/// Run a primary search and fall back to a secondary search when the primary
/// fails or returns no usable results. This generic helper is public so the
/// production audit can inject deterministic transport faults without relying
/// on external internet availability.
pub async fn search_with_failover<P, S, PF, SF>(
    primary: P,
    secondary: S,
) -> Result<(Vec<WebSnippet>, SearchEngine)>
where
    P: FnOnce() -> PF,
    S: FnOnce() -> SF,
    PF: std::future::Future<Output = Result<Vec<WebSnippet>>>,
    SF: std::future::Future<Output = Result<Vec<WebSnippet>>>,
{
    match primary().await {
        Ok(snippets) if !snippets.is_empty() => {
            return Ok((snippets, SearchEngine::DuckDuckGo));
        }
        Ok(_) => tracing::warn!("primary search returned 0 results; failing over"),
        Err(e) => tracing::warn!("primary search failed ({}); failing over", e),
    }

    let snippets = secondary().await?;
    Ok((snippets, SearchEngine::Wikipedia))
}

/// Dual-engine failover search.
///
/// 1. Primary: DuckDuckGo HTML Lite.
/// 2. Secondary: SearXNG instance (if `LAGI_SEARXNG_URL` is set), then the
///    keyless Wikipedia REST API.
///
/// Failover triggers on transport errors, HTTP error statuses (incl. DDG
/// rate-limiting), or an empty result set from the primary.
pub async fn search_web_dual(query: &str) -> Result<(Vec<WebSnippet>, SearchEngine)> {
    if query.trim().is_empty() {
        return Err(anyhow!("empty search query"));
    }

    match search_web(query).await {
        Ok(snippets) if !snippets.is_empty() => {
            return Ok((snippets, SearchEngine::DuckDuckGo));
        }
        Ok(_) => tracing::warn!("ddg returned 0 results; failing over"),
        Err(e) => tracing::warn!("ddg failed ({}); failing over", e),
    }

    if std::env::var("LAGI_SEARXNG_URL").is_ok() {
        match search_searxng(query).await {
            Ok(snippets) if !snippets.is_empty() => {
                return Ok((snippets, SearchEngine::Searxng));
            }
            Ok(_) => tracing::warn!("searxng returned 0 results; falling back to wikipedia"),
            Err(e) => tracing::warn!("searxng failed ({}); falling back to wikipedia", e),
        }
    }

    let snippets = search_wikipedia(query).await?;
    Ok((snippets, SearchEngine::Wikipedia))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ddg_html() {
        let html = r#"
        <div class="result">
        <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fblog.rust-lang.org%2F2026%2F09%2F18%2FRust-1.90.0.html&rut=abc">Rust 1.90.0 released</a>
        <a class="result__snippet" href="...">Announcing <b>Rust</b> 1.90.0 with new features & fixes.</a>
        </div>"#;
        let out = parse_ddg_html(html).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].url,
            "https://blog.rust-lang.org/2026/09/18/Rust-1.90.0.html"
        );
        assert_eq!(out[0].title, "Rust 1.90.0 released");
        assert_eq!(
            out[0].snippet,
            "Announcing Rust 1.90.0 with new features & fixes."
        );
    }

    #[test]
    fn test_html_to_markdown() {
        let html = "<html><head><style>x{}</style></head><body>\
                    <h2>Hello</h2><p>World & <b>friends</b></p>\
                    <ul><li>one</li><li>two</li></ul>\
                    <script>evil()</script></body></html>";
        let md = html_to_markdown(html);
        assert!(md.contains("## Hello"));
        assert!(md.contains("World & friends"));
        assert!(md.contains("- one"));
        assert!(!md.contains("evil"));
        assert!(!md.contains("x{}"));
    }

    #[test]
    fn test_percent_decode() {
        assert_eq!(percent_decode("a%20b%2Fc+d"), "a b/c d");
    }
}
