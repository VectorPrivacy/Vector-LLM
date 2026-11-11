//! What people attach, made readable for the model: pictures as images it can look at, and
//! documents (text, config, code, PDFs, Word files) as text in the conversation. Anything else
//! becomes a one-line note that it was sent.

use std::io::Read as _;
use std::path::Path;
use std::time::Duration;

use log::warn;
use vector_sdk::vector_core::image_prep::{prepare, ImageUse};
use vector_sdk::{Message, VectorBot};

/// Larger files aren't fetched at all.
const MAX_DOWNLOAD: u64 = 25 * 1024 * 1024;
/// How much of one document the model reads.
const MAX_TEXT_CHARS: usize = 60_000;
/// Pictures looked at from one message.
const MAX_IMAGES: usize = 4;

/// Extensions read as text, and the code-fence language each gets.
const TEXT_EXTS: &[&str] = &[
    "txt", "md", "markdown", "rst", "org", "tex", "log", "csv", "tsv", "json", "jsonl", "ndjson", "yaml", "yml", "toml",
    "ini", "conf", "cfg", "config", "env", "properties", "xml", "html", "htm", "css", "scss", "sass", "less", "svg", "js",
    "mjs", "cjs", "ts", "tsx", "jsx", "vue", "svelte", "py", "pyi", "rb", "php", "pl", "lua", "r", "jl", "go", "rs", "c",
    "h", "cc", "cpp", "cxx", "hpp", "hh", "cs", "java", "kt", "kts", "scala", "swift", "m", "mm", "dart", "zig", "nim",
    "ex", "exs", "erl", "hs", "ml", "clj", "sql", "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd", "mk", "cmake",
    "gradle", "dockerfile", "nix", "diff", "patch", "srt", "vtt", "ics", "vcf", "gitignore", "editorconfig", "lock",
];

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "gif", "webp", "bmp", "tif", "tiff", "avif", "heic", "heif"];

/// What a message's attachments came to.
#[derive(Default)]
pub struct Read {
    /// Each document in a fenced block, and a note for whatever couldn't be read.
    pub text: String,
    /// Pictures saved under the images directory, by file name.
    pub images: Vec<String>,
}

impl Read {
    fn note(&mut self, line: String) {
        if !self.text.is_empty() {
            self.text.push_str("\n\n");
        }
        self.text.push_str(&line);
    }
}

enum Kind {
    Image,
    Pdf,
    Docx,
    Text,
    Other,
}

fn kind(ext: &str, bytes: &[u8]) -> Kind {
    if IMAGE_EXTS.contains(&ext) {
        Kind::Image
    } else if ext == "pdf" || bytes.starts_with(b"%PDF") {
        Kind::Pdf
    } else if ext == "docx" {
        Kind::Docx
    } else if TEXT_EXTS.contains(&ext) || looks_like_text(bytes) {
        Kind::Text
    } else {
        Kind::Other
    }
}

/// Valid UTF-8 with no NUL bytes over its first 8 KB: a text file whatever its name says.
fn looks_like_text(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(8192)];
    if head.is_empty() || head.contains(&0) {
        return false;
    }
    match std::str::from_utf8(head) {
        Ok(_) => true,
        // Cut mid-character at the 8 KB edge is still text.
        Err(e) => e.error_len().is_none(),
    }
}

/// A document as the model reads it: fenced, and cut at [`MAX_TEXT_CHARS`].
fn block(name: &str, lang: &str, text: &str) -> String {
    let text = text.trim();
    let total = text.chars().count();
    let body: String = text.chars().take(MAX_TEXT_CHARS).collect();
    let fence = if body.contains("```") { "````" } else { "```" };
    let mut out = format!("📎 {name}:\n{fence}{lang}\n{body}\n{fence}");
    if total > MAX_TEXT_CHARS {
        out.push_str(&format!("\n(only the first {MAX_TEXT_CHARS} of its {total} characters are shown)"));
    }
    out
}

/// The text of a Word document: its paragraphs, tables and line breaks, without the markup.
fn docx_text(bytes: &[u8]) -> Option<String> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).ok()?;
    let mut xml = String::new();
    zip.by_name("word/document.xml").ok()?.take(20 * 1024 * 1024).read_to_string(&mut xml).ok()?;
    let xml = xml.replace("</w:p>", "\n").replace("<w:tab/>", "\t").replace("<w:br/>", "\n").replace("</w:tc>", "\t");
    let mut text = String::with_capacity(xml.len() / 4);
    let mut in_tag = false;
    for c in xml.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => text.push(c),
            _ => {}
        }
    }
    Some(text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&"))
}

/// The text of a PDF, read off the async runtime and given up on after a minute.
async fn pdf_text(bytes: Vec<u8>) -> Option<String> {
    let reading = tokio::task::spawn_blocking(move || pdf_extract::extract_text_from_mem(&bytes).ok());
    tokio::time::timeout(Duration::from_secs(60), reading).await.ok()?.ok()?
}

/// Read every attachment on `msg`. Pictures are fitted and saved in `images_dir` for the model to
/// look at when `vision` is on.
pub async fn read(bot: &VectorBot, msg: &Message, images_dir: &Path, vision: bool) -> Read {
    let mut out = Read::default();
    for (i, att) in msg.attachments.iter().enumerate() {
        let ext = att.extension.trim_start_matches('.').to_ascii_lowercase();
        let name = if att.name.trim().is_empty() { format!("a .{ext} file") } else { att.name.clone() };
        if att.size > MAX_DOWNLOAD {
            out.note(format!("[{name}: too large to open, {} MB]", att.size / (1024 * 1024)));
            continue;
        }
        let bytes = match bot.download_attachment_from(att, msg.npub.as_deref()).await {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!("couldn't download {name}: {e}");
                out.note(format!("[{name}: couldn't be downloaded]"));
                continue;
            }
        };
        match kind(&ext, &bytes) {
            Kind::Image if !vision => out.note(format!("[sent a picture, {name}, which you can't see]")),
            Kind::Image if out.images.len() >= MAX_IMAGES => out.note(format!("[sent another picture, {name}, beyond the {MAX_IMAGES} you can look at at once]")),
            Kind::Image => {
                // Fitted to 1500 px and a few hundred KB: plenty to see by, and a modest bill.
                let fitted = tokio::task::spawn_blocking(move || prepare(&bytes, ImageUse::Banner)).await;
                match fitted {
                    Ok(Ok(picture)) => {
                        let file = format!("{}-{i}.{}", msg.id, picture.extension);
                        match std::fs::create_dir_all(images_dir).and_then(|_| std::fs::write(images_dir.join(&file), &picture.bytes)) {
                            Ok(()) => {
                                out.images.push(file);
                                out.note(format!("[sent a picture: {name}]"));
                            }
                            Err(e) => {
                                warn!("couldn't save {name}: {e}");
                                out.note(format!("[sent a picture, {name}, which couldn't be opened]"));
                            }
                        }
                    }
                    _ => out.note(format!("[sent a picture, {name}, which couldn't be opened]")),
                }
            }
            Kind::Pdf => match pdf_text(bytes).await {
                Some(text) if !text.trim().is_empty() => out.note(block(&name, "", &text)),
                Some(_) => out.note(format!("[{name}: a PDF with no text in it, likely scanned pages]")),
                None => out.note(format!("[{name}: a PDF that couldn't be read]")),
            },
            Kind::Docx => match docx_text(&bytes) {
                Some(text) if !text.trim().is_empty() => out.note(block(&name, "", &text)),
                _ => out.note(format!("[{name}: a Word document that couldn't be read]")),
            },
            Kind::Text => {
                let lang = if TEXT_EXTS.contains(&ext.as_str()) { ext.as_str() } else { "" };
                out.note(block(&name, lang, &String::from_utf8_lossy(&bytes)));
            }
            Kind::Other => out.note(format!("[{name}: a file you can't open]")),
        }
    }
    out
}

/// A saved picture as a `data:` URL for the model.
pub fn data_url(images_dir: &Path, file: &str) -> Option<String> {
    use base64::Engine;
    let mime = match file.rsplit('.').next()? {
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "image/jpeg",
    };
    let bytes = std::fs::read(images_dir.join(file)).ok()?;
    Some(format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_are_told_apart_by_name_then_by_content() {
        assert!(matches!(kind("png", b""), Kind::Image));
        assert!(matches!(kind("", b"%PDF-1.7 ..."), Kind::Pdf));
        assert!(matches!(kind("conf", b"\x00binary"), Kind::Text), "a named config is text");
        assert!(matches!(kind("", b"server {\n  listen 80;\n}\n"), Kind::Text), "unnamed, but text");
        assert!(matches!(kind("bin", b"\x7fELF\x00\x00"), Kind::Other));
    }

    #[test]
    fn a_long_document_is_cut_and_says_so() {
        let text = "a".repeat(MAX_TEXT_CHARS + 10);
        let out = block("big.txt", "txt", &text);
        assert!(out.starts_with("📎 big.txt:\n```txt\n"));
        assert!(out.ends_with(&format!("(only the first {MAX_TEXT_CHARS} of its {} characters are shown)", MAX_TEXT_CHARS + 10)));
        assert!(block("x.md", "md", "has ``` inside").contains("````md"), "a fence inside gets a longer one around it");
    }

    #[test]
    fn a_word_document_reads_as_its_paragraphs() {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        zip.start_file("word/document.xml", zip::write::SimpleFileOptions::default()).unwrap();
        std::io::Write::write_all(&mut zip, br#"<w:document><w:body><w:p><w:r><w:t>Hello &amp; welcome</w:t></w:r></w:p><w:p><w:r><w:t>Second</w:t></w:r></w:p></w:body></w:document>"#).unwrap();
        let bytes = zip.finish().unwrap().into_inner();
        assert_eq!(docx_text(&bytes).unwrap().trim(), "Hello & welcome\nSecond");
    }
}
