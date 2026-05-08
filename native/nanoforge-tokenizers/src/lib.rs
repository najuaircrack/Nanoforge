
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader};

// ── Token ID constants ────────────────────────────────────────────────────────

const PAD_ID: u32 = 0;
const BOS_ID: u32 = 1;
const EOS_ID: u32 = 2;
const OFFSET: u32 = 4;
const BPE_BYTE_OFFSET: u32 = 14;
const SPECIAL_TOKENS: [&str; 14] = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|endoftext|>",
];

// ── Heap entry ────────────────────────────────────────────────────────────────

/// An entry in the max-priority queue used to select the best merge pair.
/// Tie-breaking on `pair` guarantees deterministic training across runs.
#[derive(Eq, PartialEq)]
struct HeapEntry {
    count: i64,
    pair: (u32, u32),
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            // Lower numeric pair wins ties (deterministic, matches insertion order
            // semantics when pairs first appear at the same frequency).
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── Word-frequency builders ───────────────────────────────────────────────────

/// Build a frequency table from a slice of strings without splitting on
/// whitespace (each string is treated as one sequence).  Uses Rayon's
/// parallel fold+reduce so threads share no mutable state.
fn word_freq_from_texts_whole(texts: &[String]) -> HashMap<Vec<u8>, usize> {
    texts
        .par_iter()
        .fold(HashMap::new, |mut map, text| {
            if !text.is_empty() {
                *map.entry(text.as_bytes().to_vec()).or_insert(0) += 1;
            }
            map
        })
        .reduce(HashMap::new, merge_freq_maps)
}

/// Like `word_freq_from_texts_whole` but splits on ASCII whitespace first,
/// enabling much better deduplication for natural-language corpora.
fn word_freq_from_texts_split(texts: &[String]) -> HashMap<Vec<u8>, usize> {
    texts
        .par_iter()
        .fold(HashMap::new, |mut map, text| {
            for word in text.split_ascii_whitespace() {
                *map.entry(word.as_bytes().to_vec()).or_insert(0) += 1;
            }
            map
        })
        .reduce(HashMap::new, merge_freq_maps)
}

/// Stream a file line-by-line, never holding more than one line in memory
/// beyond the frequency table itself.  Pass `split = true` for word-level
/// tokenisation, `split = false` to treat each line as one sequence.
fn word_freq_from_reader<R: BufRead>(reader: R, split: bool) -> HashMap<Vec<u8>, usize> {
    let mut freq: HashMap<Vec<u8>, usize> = HashMap::new();
    for line in reader.lines().flatten() {
        if split {
            for word in line.split_ascii_whitespace() {
                *freq.entry(word.as_bytes().to_vec()).or_insert(0) += 1;
            }
        } else if !line.is_empty() {
            *freq.entry(line.into_bytes()).or_insert(0) += 1;
        }
    }
    freq
}

#[inline]
fn merge_freq_maps(
    mut a: HashMap<Vec<u8>, usize>,
    b: HashMap<Vec<u8>, usize>,
) -> HashMap<Vec<u8>, usize> {
    for (k, v) in b {
        *a.entry(k).or_insert(0) += v;
    }
    a
}

// ── Training corpus ───────────────────────────────────────────────────────────

/// The mutable state threaded through the training loop.
struct TrainingCorpus {
    /// One entry per unique byte sequence: (current token IDs, corpus frequency).
    words: Vec<(Vec<u32>, usize)>,

    /// Weighted pair counts across all words: count = Σ freq(w) for each w
    /// that contains the pair.  Maintained incrementally; never rebuilt.
    pair_counts: HashMap<(u32, u32), i64>,

    /// Inverted index: pair → set of word indices that contain it.
    /// May contain stale entries (a word that no longer contains the pair after
    /// a prior merge); staleness is detected cheaply before processing.
    pair_index: HashMap<(u32, u32), Vec<usize>>,
}

impl TrainingCorpus {
    fn from_texts(texts: &[String], split_on_whitespace: bool) -> Self {
        let word_freq = if split_on_whitespace {
            word_freq_from_texts_split(texts)
        } else {
            word_freq_from_texts_whole(texts)
        };
        Self::build(word_freq)
    }

    fn from_file(path: &str, split_on_whitespace: bool) -> std::io::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::with_capacity(1 << 20, file); // 1 MiB read buffer
        let word_freq = word_freq_from_reader(reader, split_on_whitespace);
        Ok(Self::build(word_freq))
    }

    fn build(word_freq: HashMap<Vec<u8>, usize>) -> Self {
        // Convert byte sequences → initial symbol ID sequences.
        // Single-byte sequences can never produce a BPE merge and are skipped.
        let words: Vec<(Vec<u32>, usize)> = word_freq
            .into_iter()
            .filter(|(bytes, _)| bytes.len() >= 2)
            .map(|(bytes, freq)| {
                let symbols: Vec<u32> = bytes
                    .iter()
                    .map(|b| u32::from(*b) + BPE_BYTE_OFFSET)
                    .collect();
                (symbols, freq)
            })
            .collect();

        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::with_capacity(1 << 16);
        let mut pair_index: HashMap<(u32, u32), Vec<usize>> = HashMap::with_capacity(1 << 16);

        for (word_idx, (symbols, freq)) in words.iter().enumerate() {
            adjust_pair_counts(symbols, *freq as i64, &mut pair_counts);
            for w in symbols.windows(2) {
                pair_index.entry((w[0], w[1])).or_default().push(word_idx);
            }
        }

        // Deduplicate: a word with a repeated pair would have pushed itself
        // multiple times into the same pair_index bucket.
        for indices in pair_index.values_mut() {
            indices.sort_unstable();
            indices.dedup();
        }

        Self { words, pair_counts, pair_index }
    }

    /// Snapshot the current pair counts into the priority queue.
    fn build_heap(&self) -> BinaryHeap<HeapEntry> {
        self.pair_counts
            .iter()
            .map(|(&pair, &count)| HeapEntry { count, pair })
            .collect()
    }
}

// ── Core helpers ──────────────────────────────────────────────────────────────

/// Add `delta` (positive or negative) to every pair that appears as a
/// consecutive window in `symbols`.  Called with +freq when adding a word's
/// contribution and -freq when removing it before an in-place merge.
#[inline]
fn adjust_pair_counts(
    symbols: &[u32],
    delta: i64,
    counts: &mut HashMap<(u32, u32), i64>,
) {
    for w in symbols.windows(2) {
        *counts.entry((w[0], w[1])).or_insert(0) += delta;
    }
}

/// Two-pointer in-place merge.  Replaces every occurrence of `pair` with
/// `token_id` without allocating a new Vec.  Returns whether any replacement
/// occurred (useful for skipping stale pair_index entries).
fn merge_in_place(symbols: &mut Vec<u32>, pair: (u32, u32), token_id: u32) -> bool {
    let mut write = 0usize;
    let mut read = 0usize;
    let mut changed = false;
    while read < symbols.len() {
        if read + 1 < symbols.len()
            && symbols[read] == pair.0
            && symbols[read + 1] == pair.1
        {
            symbols[write] = token_id;
            read += 2;
            changed = true;
        } else {
            symbols[write] = symbols[read];
            read += 1;
        }
        write += 1;
    }
    symbols.truncate(write);
    changed
}

// ── Training loop ─────────────────────────────────────────────────────────────

fn run_training(
    corpus: &mut TrainingCorpus,
    max_merges: usize,
    min_frequency: usize,
    progress_cb: Option<&PyObject>,  // ← NEW
    py: Python<'_>,    
) -> Vec<(u32, u32)> {
    let min_freq_i64 = min_frequency as i64;
    let mut heap = corpus.build_heap();
    let mut merges: Vec<(u32, u32)> = Vec::with_capacity(max_merges);

    'merge: for step  in 0..max_merges {
        // ── 1. Find the best pair via the priority queue ──────────────────────
        // The heap uses lazy deletion: an entry is valid only when its stored
        // count equals the current count in pair_counts.
         // ── Report progress every 100 merges ─────────────────────────────────
        if let Some(cb) = progress_cb {
            if step % 100 == 0 {
                let _ = cb.call1(py, (step as u32, max_merges as u32));
            }
        }

        let (best_count, best_pair) = loop {
            let Some(HeapEntry { count, pair }) = heap.pop() else {
                break 'merge;
            };
            let live = corpus.pair_counts.get(&pair).copied().unwrap_or(0);
            if live > 0 && live == count {
                break (count, pair);
            }
        };

        if best_count < min_freq_i64 {
            break;
        }

        let (best_count, best_pair) = loop {
            let Some(HeapEntry { count, pair }) = heap.pop() else {
                break 'merge; // heap exhausted
            };
            let live = corpus.pair_counts.get(&pair).copied().unwrap_or(0);
            if live > 0 && live == count {
                break (count, pair); // valid, non-stale entry
            }
            // Stale entry (a prior merge reduced the count). Discard and retry.
        };

        if best_count < min_freq_i64 {
            break; // remaining pairs are too rare
        }

        let token_id = BPE_BYTE_OFFSET + 256 + merges.len() as u32;

        // ── 2. Collect affected word indices and remove from index ─────────────
        let affected = corpus.pair_index.remove(&best_pair).unwrap_or_default();

        // New-pair delta accumulator: tracks count changes to push back to heap.
        let mut heap_updates: HashMap<(u32, u32), i64> = HashMap::new();

        // ── 3. For each affected word: update counts and merge in-place ───────
        for &word_idx in &affected {
            // Borrow symbols immutably first to run the staleness check.
            {
                let (symbols, _) = &corpus.words[word_idx];
                let actually_contains = symbols
                    .windows(2)
                    .any(|w| w[0] == best_pair.0 && w[1] == best_pair.1);
                if !actually_contains {
                    // pair_index entry is stale from a previous merge. Skip.
                    continue;
                }
            }

            let freq = corpus.words[word_idx].1 as i64;

            // 3a. Remove this word's current pair contributions globally.
            //     (We'll re-add the post-merge contributions in 3c.)
            {
                let (symbols, _) = &corpus.words[word_idx];
                adjust_pair_counts(symbols, -freq, &mut corpus.pair_counts);
            }

            // 3b. In-place merge — no allocation.
            {
                let (symbols, _) = &mut corpus.words[word_idx];
                merge_in_place(symbols, best_pair, token_id);
            }

            // 3c. Re-add updated pair contributions and update the index.
            {
                let (symbols, _) = &corpus.words[word_idx];
                adjust_pair_counts(symbols, freq, &mut corpus.pair_counts);

                for w in symbols.windows(2) {
                    let pair = (w[0], w[1]);
                    *heap_updates.entry(pair).or_insert(0) += freq;
                    corpus.pair_index.entry(pair).or_default().push(word_idx);
                }
            }
        }

        // ── 4. Evict zero / negative counts (defensive cleanup) ───────────────
        // This keeps pair_counts compact; entries that reached 0 are gone.
        corpus.pair_counts.retain(|_, c| *c > 0);

        // ── 5. Deduplicate pair_index buckets that just received new entries ───
        for pair in heap_updates.keys() {
            if let Some(indices) = corpus.pair_index.get_mut(pair) {
                indices.sort_unstable();
                indices.dedup();
            }
        }

        // ── 6. Push updated pairs onto the heap (lazy: old entries will be ────
        //       discarded as stale when they are later popped)
        for (pair, _delta) in heap_updates {
            if let Some(&count) = corpus.pair_counts.get(&pair) {
                heap.push(HeapEntry { count, pair });
            }
        }
        // Final 100% callback
        if let Some(cb) = progress_cb {
            let _ = cb.call1(py, (merges.len() as u32, max_merges as u32));
        }

        merges.push(best_pair);
    }
    

    merges
}

// ── PyO3 types ────────────────────────────────────────────────────────────────

#[pyclass]
struct ByteTokenizer;

#[pymethods]
impl ByteTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    #[getter]
    fn pad_id(&self) -> u32 {
        PAD_ID
    }
    #[getter]
    fn bos_id(&self) -> u32 {
        BOS_ID
    }
    #[getter]
    fn eos_id(&self) -> u32 {
        EOS_ID
    }
    #[getter]
    fn unk_id(&self) -> u32 {
        3
    }
    #[getter]
    fn vocab_size(&self) -> u32 {
        260
    }

    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        encode_bytes(text.as_bytes(), add_bos, add_eos)
    }

    fn encode_batch(&self, texts: Vec<String>, add_bos: bool, add_eos: bool) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .map(|t| encode_bytes(t.as_bytes(), add_bos, add_eos))
            .collect()
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let bytes: Vec<u8> = ids
            .into_iter()
            .filter_map(|idx| {
                if (OFFSET..OFFSET + 256).contains(&idx) {
                    Some((idx - OFFSET) as u8)
                } else {
                    None
                }
            })
            .collect();
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }
}

// ── Serialisation schema (unchanged so existing .json files stay compatible) ──

#[derive(Serialize, Deserialize)]
struct BpeJson {
    #[serde(rename = "type")]
    kind: String,
    version: u32,
    special_tokens: Vec<String>,
    merges: Vec<[u32; 2]>,
}

// ── ByteLevelBpeTokenizer ─────────────────────────────────────────────────────

#[pyclass]
struct ByteLevelBpeTokenizer {
    merges: Vec<(u32, u32)>,
    pair_to_id: HashMap<(u32, u32), u32>,
    // wrap in Option so it's only populated on first decode call
    id_to_piece: Option<HashMap<u32, Vec<u8>>>,
}

#[pymethods]
impl ByteLevelBpeTokenizer {
    #[new]
    fn new(path: Option<String>) -> PyResult<Self> {
        match path {
            Some(p) => Self::from_file(&p),
            None => Ok(Self::build_tables(Vec::new())),
        }
    }

    // ── Training entry points ─────────────────────────────────────────────────

    /// Train from an in-memory list of strings.
    ///
    /// `split_on_whitespace` (default `false`) mirrors the original behaviour.
    /// Set it to `true` for word-level BPE (much better deduplication on
    /// natural-language text).
    #[staticmethod]
    #[pyo3(signature = (texts, vocab_size, min_frequency=1, split_on_whitespace=false, progress_cb=None))]
    fn train_from_texts(
        py: Python<'_>,
        texts: Vec<String>,
        vocab_size: usize,
        min_frequency: usize,
        split_on_whitespace: bool,
        progress_cb: Option<PyObject>,  // ← NEW
    ) -> PyResult<Self> {
        let mut corpus = TrainingCorpus::from_texts(&texts, split_on_whitespace);
        drop(texts);
        let max_merges = vocab_size.saturating_sub((BPE_BYTE_OFFSET + 256) as usize);
        let merges = run_training(&mut corpus, max_merges, min_frequency.max(1), progress_cb.as_ref(), py);
        Ok(Self::build_tables(merges))
    }

    #[staticmethod]
    #[pyo3(signature = (path, vocab_size, min_frequency=1, split_on_whitespace=true, progress_cb=None))]
    fn train_from_file(
        py: Python<'_>,
        path: &str,
        vocab_size: usize,
        min_frequency: usize,
        split_on_whitespace: bool,
        progress_cb: Option<PyObject>,  // ← NEW
    ) -> PyResult<Self> {
        let mut corpus = TrainingCorpus::from_file(path, split_on_whitespace)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        let max_merges = vocab_size.saturating_sub((BPE_BYTE_OFFSET + 256) as usize);
        let merges = run_training(&mut corpus, max_merges, min_frequency.max(1), progress_cb.as_ref(), py);
        Ok(Self::build_tables(merges))
    }

    

    // ── Getters ───────────────────────────────────────────────────────────────

    #[getter]
    fn pad_id(&self) -> u32 {
        PAD_ID
    }
    #[getter]
    fn bos_id(&self) -> u32 {
        BOS_ID
    }
    #[getter]
    fn eos_id(&self) -> u32 {
        EOS_ID
    }
    #[getter]
    fn unk_id(&self) -> u32 {
        3
    }
    #[getter]
    fn vocab_size(&self) -> u32 {
        BPE_BYTE_OFFSET + 256 + self.merges.len() as u32
    }

    // ── Encode / decode (unchanged behaviour) ─────────────────────────────────

   // In ByteLevelBpeTokenizer impl, replace encode():
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut a: Vec<u32> = text
            .as_bytes()
            .iter()
            .map(|b| u32::from(*b) + BPE_BYTE_OFFSET)
            .collect();
        let mut b: Vec<u32> = Vec::with_capacity(a.len());

        for (idx, &pair) in self.merges.iter().enumerate() {
            apply_merge_inplace(&a, &mut b, pair, BPE_BYTE_OFFSET + 256 + idx as u32);
            std::mem::swap(&mut a, &mut b);
        }

        if add_bos { a.insert(0, BOS_ID); }
        if add_eos { a.push(EOS_ID); }
        a
    }

    fn encode_batch(&self, texts: Vec<String>, add_bos: bool, add_eos: bool) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .map(|t| self.encode(t, add_bos, add_eos))
            .collect()
    }

    fn decode(&mut self, ids: Vec<u32>) -> PyResult<String> {
        let table = self.get_or_build_pieces();
        let mut bytes = Vec::new();
        for idx in ids {
            if let Some(piece) = table.get(&idx) {
                bytes.extend_from_slice(piece);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    fn save(&self, path: &str) -> PyResult<()> {
        let payload = BpeJson {
            kind: "nanoforge_python_bpe".to_string(),
            version: 1,
            special_tokens: SPECIAL_TOKENS.iter().map(|t| t.to_string()).collect(),
            merges: self.merges.iter().map(|&(l, r)| [l, r]).collect(),
        };
        let json = serde_json::to_string(&payload)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        fs::write(path, json)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
    }
}

impl ByteLevelBpeTokenizer {

    fn get_or_build_pieces(&mut self) -> &HashMap<u32, Vec<u8>> {
        self.id_to_piece.get_or_insert_with(|| {
            let mut map: HashMap<u32, Vec<u8>> = (0u32..256)
                .map(|b| (BPE_BYTE_OFFSET + b, vec![b as u8]))
                .collect();
            for (idx, &pair) in self.merges.iter().enumerate() {
                let token_id = BPE_BYTE_OFFSET + 256 + idx as u32;
                let mut piece = map.get(&pair.0).cloned().unwrap_or_default();
                if let Some(right) = map.get(&pair.1) {
                    piece.extend_from_slice(right);
                }
                map.insert(token_id, piece);
            }
            map
        })
    }

    fn from_file(path: &str) -> PyResult<Self> {
        let text = fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        let payload: BpeJson = serde_json::from_str(&text)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if payload.kind != "nanoforge_python_bpe" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Tokenizer is not a Nanoforge byte-level BPE artifact.",
            ));
        }
        let merges = payload.merges.into_iter().map(|p| (p[0], p[1])).collect();
        Ok(Self::build_tables(merges))
    }

    fn build_tables(merges: Vec<(u32, u32)>) -> Self {
        let mut pair_to_id: HashMap<(u32, u32), u32> =
            HashMap::with_capacity(merges.len());
        let mut id_to_piece: HashMap<u32, Vec<u8>> = (0u32..256)
            .map(|b| (BPE_BYTE_OFFSET + b, vec![b as u8]))
            .collect();

        for (idx, &pair) in merges.iter().enumerate() {
            let token_id = BPE_BYTE_OFFSET + 256 + idx as u32;
            pair_to_id.insert(pair, token_id);
            let mut piece = id_to_piece.get(&pair.0).cloned().unwrap_or_default();
            if let Some(right) = id_to_piece.get(&pair.1) {
                piece.extend_from_slice(right);
            }
            id_to_piece.insert(token_id, piece);
        }

        Self { merges, pair_to_id, id_to_piece: Some(id_to_piece) }
    }
    
}

// ── Module-level free functions ───────────────────────────────────────────────

/// Encode a byte string as token IDs using the flat byte tokenizer.
fn encode_bytes(bytes: &[u8], add_bos: bool, add_eos: bool) -> Vec<u32> {
    let extra = usize::from(add_bos) + usize::from(add_eos);
    let mut ids = Vec::with_capacity(bytes.len() + extra);
    if add_bos {
        ids.push(BOS_ID);
    }
    ids.extend(bytes.iter().map(|b| u32::from(*b) + OFFSET));
    if add_eos {
        ids.push(EOS_ID);
    }
    ids
}
fn apply_merge_inplace(src: &[u32], dst: &mut Vec<u32>, pair: (u32, u32), token_id: u32) {
    dst.clear();
    let mut idx = 0;
    while idx < src.len() {
        if idx + 1 < src.len() && src[idx] == pair.0 && src[idx + 1] == pair.1 {
            dst.push(token_id);
            idx += 2;
        } else {
            dst.push(src[idx]);
            idx += 1;
        }
    }
}


#[pyfunction]
fn sanitize_utf8(data: &[u8]) -> String {
    String::from_utf8_lossy(data).replace('\0', "")
}

#[pyfunction]
fn encode_byte_text(text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
    encode_bytes(text.as_bytes(), add_bos, add_eos)
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
fn nanoforge_tokenizers(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<ByteTokenizer>()?;
    module.add_class::<ByteLevelBpeTokenizer>()?;
    module.add_function(wrap_pyfunction!(sanitize_utf8, module)?)?;
    module.add_function(wrap_pyfunction!(encode_byte_text, module)?)?;
    Ok(())
}