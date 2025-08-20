use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::path::Path;
use tokenizers::{Tokenizer as HFTokenizer, EncodeInput};
use tokenizers::models::{bpe::BPE, wordpiece::WordPiece, unigram::Unigram};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;
use anyhow::{Result, Context};

pub mod tokenizer;
pub mod utils;

use crate::tokenizer::{RustTokenizer, TokenizerType};

/// Python 模块定义
#[pymodule]
fn rust_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(probe_tokenizer_path, m)?)?;
    Ok(())
}

/// Python 绑定的 Tokenizer 类
#[pyclass]
struct PyTokenizer {
    inner: RustTokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// 创建新的 tokenizer 实例
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let inner = RustTokenizer::new(model_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to create tokenizer: {}", e)))?;
        Ok(PyTokenizer { inner })
    }

    /// 编码文本为 token IDs
    fn encode(&self, text: &str, bos: bool, eos: bool) -> PyResult<Vec<u32>> {
        self.inner.encode(text, bos, eos)
            .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))
    }

    /// 编码文本段（处理前导空格）
    fn encode_segment(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner.encode_segment(text)
            .map_err(|e| PyValueError::new_err(format!("Segment encoding failed: {}", e)))
    }

    /// 编码文本（不添加前缀空格）
    fn encode_wo_prefix_space(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner.encode_wo_prefix_space(text)
            .map_err(|e| PyValueError::new_err(format!("Encoding without prefix space failed: {}", e)))
    }

    /// 解码 token IDs 为文本
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids)
            .map_err(|e| PyValueError::new_err(format!("Decoding failed: {}", e)))
    }

    /// 批量编码多个文本
    fn batch_encode(&self, texts: Vec<&str>, bos: bool, eos: bool) -> PyResult<Vec<Vec<u32>>> {
        self.inner.batch_encode(&texts, bos, eos)
            .map_err(|e| PyValueError::new_err(format!("Batch encoding failed: {}", e)))
    }

    /// 批量解码多个 token ID 序列
    fn batch_decode(&self, ids_batch: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        let ids_refs: Vec<&[u32]> = ids_batch.iter().map(|v| v.as_slice()).collect();
        self.inner.batch_decode(&ids_refs)
            .map_err(|e| PyValueError::new_err(format!("Batch decoding failed: {}", e)))
    }

    /// 保存 tokenizer 到指定目录
    fn save(&self, save_dir: &str) -> PyResult<()> {
        self.inner.save(save_dir)
            .map_err(|e| PyValueError::new_err(format!("Failed to save tokenizer: {}", e)))
    }

    /// 获取词汇表大小
    #[getter]
    fn n_words(&self) -> u32 {
        self.inner.vocab_size()
    }

    /// 获取 BOS token ID
    #[getter]
    fn bos_id(&self) -> Option<u32> {
        self.inner.bos_id()
    }

    /// 获取 EOS token ID
    #[getter]
    fn eos_id(&self) -> Option<u32> {
        self.inner.eos_id()
    }

    /// 获取 tokenizer 类型
    #[getter]
    fn tokenizer_type(&self) -> String {
        match self.inner.tokenizer_type() {
            TokenizerType::BPE => "bpe".to_string(),
            TokenizerType::WordPiece => "wordpiece".to_string(),
            TokenizerType::Unigram => "unigram".to_string(),
            TokenizerType::SentencePiece => "sentencepiece".to_string(),
        }
    }

    /// 检查是否需要在段前添加空格
    #[getter]
    fn need_space_before_segment(&self) -> bool {
        self.inner.need_space_before_segment()
    }
}

/// 探测 tokenizer 路径
#[pyfunction]
fn probe_tokenizer_path(pretrained_path: &str) -> PyResult<Option<String>> {
    Ok(utils::probe_tokenizer_path(pretrained_path))
}