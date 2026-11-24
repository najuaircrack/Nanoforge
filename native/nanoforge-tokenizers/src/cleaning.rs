// Native data-cleaning + dedup for Nanoforge.
//
// Mirrors the semantics of `nanoforge.data.cleaning` (NFKC normalize, newline
// canonicalization, optional whitespace collapse, exact dedup, simhash near-dedup)
// but runs the per-text work in parallel via Rayon. Python keeps an identical
// fallback so behavior is unchanged when the extension is absent.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

const SIMHASH_BITS: usize = 64;

/// Normalize one text: strip NULs, optional NFKC, canonical newlines,
/// optional whitespace collapse, then trim + append a trailing newline.
/// Matches `clean_text` in the Python module exactly.
fn clean_one(text: &str, normalize_unicode: bool, collapse_whitespace: bool) -> String {
    let without_nul: String = text.chars().filter(|&c| c != '\0').collect();
    let normalized = if normalize_unicode {
        without_nul.nfkc().collect::<String>()
    } else {
        without_nul
    };
    let newlined = normalized.replace("\r\n", "\n").replace('\r', "\n");

    let body = if collapse_whitespace {
        collapse(&newlined)
    } else {
        newlined
    };
    let mut out = body.trim().to_string();
    out.push('\n');
    out
}

/// Collapse runs of spaces/tabs to one space and 4+ newlines to exactly three.
fn collapse(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space_run = false;
    let mut newline_run = 0usize;
    for ch in text.chars() {
        if ch == ' ' || ch == '\t' {
            if !space_run {
                out.push(' ');
                space_run = true;
            }
            newline_run = 0;
        } else if ch == '\n' {
            newline_run += 1;
            if newline_run <= 3 {
                out.push('\n');
            }
            space_run = false;
        } else {
            out.push(ch);
            space_run = false;
            newline_run = 0;
        }
    }
    out
}

/// 64-bit FNV-1a hash of a byte slice. Used both for exact-dedup keys and as
/// the per-token hash feeding simhash. Deterministic within and across runs.
#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Compute a 64-bit simhash over ASCII-lowercased `\w+` tokens, matching the
/// Python `simhash` (which weights each bit by +1/-1 per token hash bit).
fn simhash(text: &str) -> u64 {
    let mut weights = [0i32; SIMHASH_BITS];
    for token in word_tokens(text) {
        let value = fnv1a(token.as_bytes());
        for (bit, w) in weights.iter_mut().enumerate() {
            if value & (1u64 << bit) != 0 {
                *w += 1;
            } else {
                *w -= 1;
            }
        }
    }
    let mut out = 0u64;
    for (bit, &w) in weights.iter().enumerate() {
        if w > 0 {
            out |= 1u64 << bit;
        }
    }
    out
}

/// Split on non-word characters and lowercase, emulating `re.findall(r"\w+")`
/// for the ASCII range (Unicode word chars pass through as their own bytes).
fn word_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[inline]
fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Clean a batch of texts in parallel. Returns (cleaned_text, exact_hash) pairs.
/// The hash is FNV-1a of the cleaned text — used by the caller for exact dedup.
#[pyfunction]
pub fn clean_text_batch(
    texts: Vec<String>,
    normalize_unicode: bool,
    collapse_whitespace: bool,
) -> Vec<(String, u64)> {
    texts
        .par_iter()
        .map(|t| {
            let cleaned = clean_one(t, normalize_unicode, collapse_whitespace);
            let hash = fnv1a(cleaned.as_bytes());
            (cleaned, hash)
        })
        .collect()
}

/// Compute simhash fingerprints for a batch of (already-cleaned) texts in parallel.
#[pyfunction]
pub fn simhash_batch(texts: Vec<String>) -> Vec<u64> {
    texts.par_iter().map(|t| simhash(t)).collect()
}

/// Stateful streaming deduplicator. Holds the exact-dedup hash set and the
/// near-dedup fingerprint list across calls, so Python can feed batches one
/// at a time without loading the whole corpus into memory.
#[pyclass]
pub struct StreamingDeduplicator {
    seen_exact: HashSet<u64>,
    seen_near: Vec<u64>,
    near_window: usize,
    hamming_threshold: u32,
    normalize_unicode: bool,
    collapse_whitespace: bool,
    do_exact: bool,
    do_near: bool,
}

#[pymethods]
impl StreamingDeduplicator {
    #[new]
    #[pyo3(signature = (
        normalize_unicode=true,
        collapse_whitespace=false,
        deduplicate=true,
        near_deduplicate=false,
        near_window=50000,
        hamming_threshold=3,
    ))]
    fn new(
        normalize_unicode: bool,
        collapse_whitespace: bool,
        deduplicate: bool,
        near_deduplicate: bool,
        near_window: usize,
        hamming_threshold: u32,
    ) -> Self {
        Self {
            seen_exact: HashSet::new(),
            seen_near: Vec::new(),
            near_window,
            hamming_threshold,
            normalize_unicode,
            collapse_whitespace,
            do_exact: deduplicate,
            do_near: near_deduplicate,
        }
    }

    /// Process a batch of raw texts. Returns (cleaned_text, keep) pairs.
    /// Texts that are too short (< min_chars) or duplicates get keep=false.
    #[pyo3(signature = (texts, min_chars, max_chars=None))]
    fn process_batch(
        &mut self,
        texts: Vec<String>,
        min_chars: usize,
        max_chars: Option<usize>,
    ) -> Vec<(String, bool)> {
        // 1. Clean in parallel.
        let cleaned: Vec<(String, u64, usize)> = texts
            .par_iter()
            .map(|t| {
                let mut c = clean_one(t, self.normalize_unicode, self.collapse_whitespace);
                let original_chars = c.chars().count();
                if let Some(limit) = max_chars {
                    if c.chars().count() > limit {
                        c = c
                            .chars()
                            .take(limit)
                            .collect::<String>()
                            .trim_end()
                            .to_string();
                        c.push('\n');
                    }
                }
                let h = fnv1a(c.as_bytes());
                (c, h, original_chars)
            })
            .collect();

        // 2. Dedup sequentially (state must be consistent across the batch).
        let mut out = Vec::with_capacity(cleaned.len());
        for (text, hash, original_chars) in cleaned {
            if original_chars < min_chars {
                out.push((text, false));
                continue;
            }
            if self.do_exact {
                if self.seen_exact.contains(&hash) {
                    out.push((text, false));
                    continue;
                }
                self.seen_exact.insert(hash);
            }
            if self.do_near {
                let fp = simhash(&text);
                let start = self.seen_near.len().saturating_sub(self.near_window);
                let is_near_dup = self.seen_near[start..]
                    .iter()
                    .any(|&old| hamming(fp, old) <= self.hamming_threshold);
                if is_near_dup {
                    out.push((text, false));
                    continue;
                }
                self.seen_near.push(fp);
            }
            out.push((text, true));
        }
        out
    }

    fn exact_seen_count(&self) -> usize {
        self.seen_exact.len()
    }
    fn near_seen_count(&self) -> usize {
        self.seen_near.len()
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(clean_text_batch, module)?)?;
    module.add_function(wrap_pyfunction!(simhash_batch, module)?)?;
    module.add_class::<StreamingDeduplicator>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_minimum_counts_characters_not_utf8_bytes() {
        let mut deduper = StreamingDeduplicator::new(true, false, true, false, 50000, 3);
        let results = deduper.process_batch(vec!["界界界".to_string()], 8, None);
        assert!(!results[0].1);
    }

    #[test]
    fn normalization_matches_python_golden_examples() {
        assert_eq!(clean_one("  Ａ\r\nB\0 ", true, false), "A\nB\n");
        assert_eq!(clean_one("a  \tb\n\n\n\nc", false, true), "a b\n\n\nc\n");
    }

    #[test]
    fn exact_dedup_operates_after_truncation() {
        let mut deduper = StreamingDeduplicator::new(true, false, true, false, 50000, 3);
        let results = deduper.process_batch(
            vec!["abcdef1".to_string(), "abcdef2".to_string()],
            1,
            Some(6),
        );
        assert_eq!(results[0], ("abcdef\n".to_string(), true));
        assert!(!results[1].1);
    }

    #[test]
    fn deterministic_simhash_uses_shared_fnv1a() {
        assert_eq!(fnv1a(b"hello"), 0xa430d84680aabd0b);
        assert_eq!(simhash("alpha alpha"), fnv1a(b"alpha"));
    }
}
