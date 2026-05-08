use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

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
            .map(|text| encode_bytes(text.as_bytes(), add_bos, add_eos))
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

#[derive(Serialize, Deserialize)]
struct BpeJson {
    #[serde(rename = "type")]
    kind: String,
    version: u32,
    special_tokens: Vec<String>,
    merges: Vec<[u32; 2]>,
}

#[pyclass]
struct ByteLevelBpeTokenizer {
    merges: Vec<(u32, u32)>,
    pair_to_id: HashMap<(u32, u32), u32>,
    id_to_piece: HashMap<u32, Vec<u8>>,
}

#[pymethods]
impl ByteLevelBpeTokenizer {
    #[new]
    fn new(path: Option<String>) -> PyResult<Self> {
        match path {
            Some(path) => Self::from_file(&path),
            None => Ok(Self::from_merges(Vec::new())),
        }
    }

    #[staticmethod]
    fn train_from_texts(texts: Vec<String>, vocab_size: usize, min_frequency: usize) -> PyResult<Self> {
        Ok(train_byte_bpe(texts, vocab_size, min_frequency.max(1)))
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
        BPE_BYTE_OFFSET + 256 + self.merges.len() as u32
    }

    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut ids: Vec<u32> = text
            .as_bytes()
            .iter()
            .map(|byte| u32::from(*byte) + BPE_BYTE_OFFSET)
            .collect();
        for (idx, pair) in self.merges.iter().enumerate() {
            ids = merge_pair(&ids, *pair, BPE_BYTE_OFFSET + 256 + idx as u32);
        }
        if add_bos {
            ids.insert(0, BOS_ID);
        }
        if add_eos {
            ids.push(EOS_ID);
        }
        ids
    }

    fn encode_batch(&self, texts: Vec<String>, add_bos: bool, add_eos: bool) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .map(|text| self.encode(text, add_bos, add_eos))
            .collect()
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let mut bytes = Vec::new();
        for idx in ids {
            if let Some(piece) = self.id_to_piece.get(&idx) {
                bytes.extend_from_slice(piece);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let payload = BpeJson {
            kind: "nanoforge_python_bpe".to_string(),
            version: 1,
            special_tokens: SPECIAL_TOKENS.iter().map(|token| token.to_string()).collect(),
            merges: self.merges.iter().map(|(left, right)| [*left, *right]).collect(),
        };
        let text = serde_json::to_string(&payload)
            .map_err(|exc| pyo3::exceptions::PyValueError::new_err(exc.to_string()))?;
        fs::write(path, text).map_err(|exc| pyo3::exceptions::PyOSError::new_err(exc.to_string()))
    }
}

impl ByteLevelBpeTokenizer {
    fn from_file(path: &str) -> PyResult<Self> {
        let text = fs::read_to_string(path)
            .map_err(|exc| pyo3::exceptions::PyOSError::new_err(exc.to_string()))?;
        let payload: BpeJson = serde_json::from_str(&text)
            .map_err(|exc| pyo3::exceptions::PyValueError::new_err(exc.to_string()))?;
        if payload.kind != "nanoforge_python_bpe" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Tokenizer is not a Nanoforge byte-level BPE artifact.",
            ));
        }
        let merges = payload.merges.into_iter().map(|pair| (pair[0], pair[1])).collect();
        Ok(Self::from_merges(merges))
    }

    fn from_merges(merges: Vec<(u32, u32)>) -> Self {
        let mut pair_to_id = HashMap::new();
        let mut id_to_piece: HashMap<u32, Vec<u8>> = (0..256)
            .map(|byte| (BPE_BYTE_OFFSET + byte, vec![byte as u8]))
            .collect();
        for (idx, pair) in merges.iter().enumerate() {
            let token_id = BPE_BYTE_OFFSET + 256 + idx as u32;
            pair_to_id.insert(*pair, token_id);
            let mut piece = id_to_piece.get(&pair.0).cloned().unwrap_or_default();
            if let Some(right) = id_to_piece.get(&pair.1) {
                piece.extend_from_slice(right);
            }
            id_to_piece.insert(token_id, piece);
        }
        Self {
            merges,
            pair_to_id,
            id_to_piece,
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

fn encode_bytes(bytes: &[u8], add_bos: bool, add_eos: bool) -> Vec<u32> {
    let extra = usize::from(add_bos) + usize::from(add_eos);
    let mut ids = Vec::with_capacity(bytes.len() + extra);
    if add_bos {
        ids.push(BOS_ID);
    }
    ids.extend(bytes.iter().map(|byte| u32::from(*byte) + OFFSET));
    if add_eos {
        ids.push(EOS_ID);
    }
    ids
}

fn train_byte_bpe(texts: Vec<String>, vocab_size: usize, min_frequency: usize) -> ByteLevelBpeTokenizer {
    let mut words: Vec<Vec<u32>> = texts
        .iter()
        .filter_map(|text| {
            let ids: Vec<u32> = text
                .as_bytes()
                .iter()
                .map(|byte| u32::from(*byte) + BPE_BYTE_OFFSET)
                .collect();
            if ids.is_empty() {
                None
            } else {
                Some(ids)
            }
        })
        .collect();
    let max_merges = vocab_size.saturating_sub((BPE_BYTE_OFFSET + 256) as usize);
    let mut merges = Vec::new();
    for _ in 0..max_merges {
        let counts = words
            .par_iter()
            .map(pair_counts)
            .reduce(HashMap::new, |mut left, right| {
                for (pair, count) in right {
                    *left.entry(pair).or_insert(0) += count;
                }
                left
            });
        let Some((pair, count)) = counts.into_iter().max_by_key(|(_, count)| *count) else {
            break;
        };
        if count < min_frequency {
            break;
        }
        let token_id = BPE_BYTE_OFFSET + 256 + merges.len() as u32;
        words.par_iter_mut().for_each(|word| {
            *word = merge_pair(word, pair, token_id);
        });
        merges.push(pair);
    }
    ByteLevelBpeTokenizer::from_merges(merges)
}

fn pair_counts(word: &Vec<u32>) -> HashMap<(u32, u32), usize> {
    let mut counts = HashMap::new();
    for pair in word.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
    }
    counts
}

fn merge_pair(ids: &[u32], pair: (u32, u32), token_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut idx = 0;
    while idx < ids.len() {
        if idx + 1 < ids.len() && ids[idx] == pair.0 && ids[idx + 1] == pair.1 {
            out.push(token_id);
            idx += 2;
        } else {
            out.push(ids[idx]);
            idx += 1;
        }
    }
    out
}

#[pymodule]
fn nanoforge_tokenizers(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<ByteTokenizer>()?;
    module.add_class::<ByteLevelBpeTokenizer>()?;
    module.add_function(wrap_pyfunction!(sanitize_utf8, module)?)?;
    module.add_function(wrap_pyfunction!(encode_byte_text, module)?)?;
    Ok(())
}
