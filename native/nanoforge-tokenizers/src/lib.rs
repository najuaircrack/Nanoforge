use pyo3::prelude::*;
use rayon::prelude::*;

const PAD_ID: u32 = 0;
const BOS_ID: u32 = 1;
const EOS_ID: u32 = 2;
const OFFSET: u32 = 4;

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

#[pymodule]
fn nanoforge_tokenizers(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<ByteTokenizer>()?;
    module.add_function(wrap_pyfunction!(sanitize_utf8, module)?)?;
    module.add_function(wrap_pyfunction!(encode_byte_text, module)?)?;
    Ok(())
}
