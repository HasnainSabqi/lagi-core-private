/*!
Cross-Platform Desktop Notification Dispatcher

Dispatches native toast banners when autonomous ideation produces a
high-confidence thought (evaluation score >= 0.85):

- Windows: PowerShell WinRT toast via the AppUserModelID fallback chain
- macOS:   osascript `display notification`
- Linux:   notify-send

SILENT FALLBACK: if notifications fail (headless server, disabled
permissions, missing binaries) we log a trace and continue execution —
a notification failure can never crash LAGI.
*/

use tracing::{debug, trace};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

/// Ideation confidence threshold for surfacing a desktop toast
pub const IDEATION_NOTIFICATION_THRESHOLD: f32 = 0.85;

/// Show a native desktop notification. Never panics, never blocks long.
pub fn show_desktop_notification(title: &str, message: &str) {
    let title = sanitize(title);
    let message = sanitize(message);

    let result = dispatch_os_notification(&title, &message);
    match result {
        Ok(()) => debug!("desktop notification shown: {}", title),
        Err(err) => trace!("desktop notification failed silently (non-fatal): {}", err),
    }
}

/// Fire-and-forget async wrapper so callers never stall on notification I/O
pub fn show_desktop_notification_async(title: String, message: String) {
    tokio::spawn(async move {
        // Run the blocking subprocess call off the async workers
        let _ = tokio::task::spawn_blocking(move || {
            show_desktop_notification(&title, &message);
        })
        .await;
    });
}

/// Strip characters that could break shell/script quoting
fn sanitize(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '\0' | '\r' | '\n' | '\x07'))
        .take(400)
        .collect()
}

fn dispatch_os_notification(title: &str, message: &str) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        // Toast via WinRT through an inline PowerShell script. Uses the
        // PowerShell AUMID which works without an app registration.
        let script = format!(
            r#"
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$template = @"
<toast><visual><binding template="ToastText02"><text id="1">{}</text><text id="2">{}</text></binding></visual></toast>
"@
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Microsoft.Windows.PowerShell").Show($toast)
"#,
            escape_ps(title),
            escape_ps(message)
        );
        std::process::Command::new("powershell")
            .args(["-NoProfile", "-WindowStyle", "Hidden", "-Command", &script])
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .output()
            .map_err(|e| format!("powershell spawn failed: {}", e))
            .and_then(|out| {
                if out.status.success() {
                    Ok(())
                } else {
                    Err(format!(
                        "toast exited with {}: {}",
                        out.status,
                        String::from_utf8_lossy(&out.stderr)
                    ))
                }
            })
    }

    #[cfg(target_os = "macos")]
    {
        let script = format!(
            "display notification \"{}\" with title \"{}\"",
            message.replace('\\', "\\\\").replace('"', "\\\""),
            title.replace('\\', "\\\\").replace('"', "\\\"")
        );
        std::process::Command::new("osascript")
            .args(["-e", &script])
            .output()
            .map_err(|e| format!("osascript spawn failed: {}", e))
            .and_then(|out| {
                if out.status.success() {
                    Ok(())
                } else {
                    Err(format!(
                        "osascript: {}",
                        String::from_utf8_lossy(&out.stderr)
                    ))
                }
            })
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        std::process::Command::new("notify-send")
            .args(["-a", "LAGI", title, message])
            .output()
            .map_err(|e| format!("notify-send unavailable: {}", e))
            .and_then(|out| {
                if out.status.success() {
                    Ok(())
                } else {
                    Err(format!(
                        "notify-send: {}",
                        String::from_utf8_lossy(&out.stderr)
                    ))
                }
            })
    }
}

#[cfg(target_os = "windows")]
fn escape_ps(s: &str) -> String {
    s.replace('&', "&").replace('<', "<").replace('>', ">")
}

/// Convenience helper used by the ideation loop: only fire when the
/// critic-validated score crosses the notification threshold.
pub fn maybe_notify_ideation(thought: &str, score: f32) {
    if score >= IDEATION_NOTIFICATION_THRESHOLD {
        show_desktop_notification_async(
            "💡 LAGI Ideation".to_string(),
            thought.chars().take(280).collect(),
        );
    } else {
        trace!(
            "ideation score {} below notification threshold {} — silent",
            score,
            IDEATION_NOTIFICATION_THRESHOLD
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_strips_control_chars() {
        assert_eq!(sanitize("hi\r\nthere\0"), "hithere");
        let long = "x".repeat(1000);
        assert_eq!(sanitize(&long).len(), 400);
    }

    #[test]
    fn test_maybe_notify_below_threshold_is_silent() {
        // Must not panic and must not spawn anything at score 0.4
        maybe_notify_ideation("low-confidence idea", 0.4);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_escape_ps() {
        assert_eq!(escape_ps("a<b&c>"), "a<b&c>");
    }
}
