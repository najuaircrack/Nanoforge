use memmap2::{Mmap, MmapOptions};
use pyo3::prelude::*;
use std::fs::File;

#[pyclass]
pub struct MmapCorpusReader {
    mapping: Option<Mmap>,
    position: usize,
    max_record_bytes: usize,
    #[pyo3(get)]
    records: usize,
    #[pyo3(get)]
    bytes_read: usize,
    #[pyo3(get)]
    invalid_utf8_records: usize,
    #[pyo3(get)]
    skipped_records: usize,
}

#[pymethods]
impl MmapCorpusReader {
    #[new]
    #[pyo3(signature = (path, max_record_bytes=1048576, max_source_bytes=268435456))]
    fn new(path: &str, max_record_bytes: usize, max_source_bytes: u64) -> PyResult<Self> {
        if max_record_bytes == 0 || max_source_bytes == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Mmap limits must be positive",
            ));
        }
        let file = File::open(path)
            .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?;
        let size = file
            .metadata()
            .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?
            .len();
        if size > max_source_bytes {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Mmap source byte limit exceeded",
            ));
        }
        // The public contract requires an immutable corpus while the mapping
        // exists. Mapping is read-only, and no writable pointer is exposed.
        let mapping = if size == 0 {
            None
        } else {
            Some(
                unsafe { MmapOptions::new().map(&file) }
                    .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?,
            )
        };
        Ok(Self {
            mapping,
            position: 0,
            max_record_bytes,
            records: 0,
            bytes_read: 0,
            invalid_utf8_records: 0,
            skipped_records: 0,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<String>> {
        let Some(mapping) = self.mapping.as_ref() else {
            return Ok(None);
        };
        if self.position >= mapping.len() {
            return Ok(None);
        }
        let start = self.position;
        let end = mapping[start..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|index| start + index + 1)
            .unwrap_or(mapping.len());
        self.position = end;
        self.bytes_read += end - start;
        if end - start > self.max_record_bytes {
            self.skipped_records += 1;
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Mmap record byte limit exceeded",
            ));
        }
        let text = match std::str::from_utf8(&mapping[start..end]) {
            Ok(text) => text.to_string(),
            Err(_) => {
                self.invalid_utf8_records += 1;
                String::from_utf8_lossy(&mapping[start..end]).into_owned()
            }
        };
        self.records += 1;
        Ok(Some(text))
    }

    fn close(&mut self) {
        self.mapping = None;
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MmapCorpusReader>()?;
    Ok(())
}
