use pyo3::prelude::*;
use pyo3::types::PyModule;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

#[pyclass]
struct SimpleTokenizer {}

#[pymethods]
impl SimpleTokenizer {
    #[new]
    fn new() -> Self { Self {} }

    /// Tokenize text by Unicode word boundaries with NFC normalization.
    fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = text.nfc().collect::<String>();
        normalized
            .unicode_words()
            .map(|w| w.to_string())
            .collect()
    }
}

/// Load a Hugging Face tokenizer from a tokenizer.json and encode text.
#[pyfunction]
fn hf_encode(tokenizer_json_path: &str, text: &str, add_special_tokens: bool) -> PyResult<(Vec<u32>, Vec<String>)> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("failed to load tokenizer: {}", e)))?;
    let encoding = tokenizer
        .encode(text, add_special_tokens)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("encode error: {}", e)))?;
    Ok((encoding.get_ids().to_vec(), encoding.get_tokens().to_vec()))
}

#[pymodule]
fn rust_tokenizer(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SimpleTokenizer>()?;
    m.add_function(wrap_pyfunction!(hf_encode, m)?)?;
    Ok(())
}
