use anyhow::{Result, Context, bail};
use std::path::{Path, PathBuf};
use tokenizers::{Tokenizer as HFTokenizer, EncodeInput, Encoding};
use tokenizers::models::{bpe::BPE, wordpiece::WordPiece, unigram::Unigram};
use rayon::prelude::*;
use std::fs;
use serde_json;

#[derive(Debug, Clone, Copy)]
pub enum TokenizerType {
    BPE,
    WordPiece,
    Unigram,
    SentencePiece,
}

/// Rust 实现的高性能 Tokenizer
pub struct RustTokenizer {
    tokenizer: HFTokenizer,
    tokenizer_type: TokenizerType,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    need_space_before_segment: bool,
}

impl RustTokenizer {
    /// 创建新的 tokenizer 实例
    pub fn new(model_path: &str) -> Result<Self> {
        let path = Path::new(model_path);
        
        // 判断 tokenizer 类型并加载
        let (tokenizer, tokenizer_type) = if model_path.ends_with(".model") {
            // SentencePiece 模型
            Self::load_sentencepiece_model(path)?
        } else if path.join("tokenizer.json").exists() {
            // HuggingFace tokenizer
            Self::load_hf_tokenizer(path)?
        } else if path.join("vocab.txt").exists() {
            // WordPiece tokenizer (BERT-style)
            Self::load_wordpiece_tokenizer(path)?
        } else {
            bail!("Unsupported tokenizer format at path: {}", model_path);
        };

        // 获取特殊 token IDs
        let bos_id = tokenizer.token_to_id("[BOS]")
            .or_else(|| tokenizer.token_to_id("<s>"))
            .or_else(|| tokenizer.token_to_id("[CLS]"));
            
        let eos_id = tokenizer.token_to_id("[EOS]")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("[SEP]"));

        let mut rust_tokenizer = RustTokenizer {
            tokenizer,
            tokenizer_type,
            bos_id,
            eos_id,
            need_space_before_segment: false,
        };

        // 探测 tokenizer 风格
        rust_tokenizer.probe_tokenizer_style();

        Ok(rust_tokenizer)
    }

    /// 加载 SentencePiece 模型
    fn load_sentencepiece_model(path: &Path) -> Result<(HFTokenizer, TokenizerType)> {
        // 使用 Unigram 模型来模拟 SentencePiece
        let tokenizer = HFTokenizer::from_file(path)
            .with_context(|| format!("Failed to load SentencePiece model from {:?}", path))?;
        Ok((tokenizer, TokenizerType::SentencePiece))
    }

    /// 加载 HuggingFace tokenizer
    fn load_hf_tokenizer(path: &Path) -> Result<(HFTokenizer, TokenizerType)> {
        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = HFTokenizer::from_file(&tokenizer_path)
            .with_context(|| format!("Failed to load HF tokenizer from {:?}", tokenizer_path))?;
        
        // 检测模型类型
        let config_path = path.join("tokenizer_config.json");
        let tokenizer_type = if config_path.exists() {
            let config_str = fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;
            
            match config.get("model_type").and_then(|v| v.as_str()) {
                Some("BPE") => TokenizerType::BPE,
                Some("WordPiece") => TokenizerType::WordPiece,
                Some("Unigram") => TokenizerType::Unigram,
                _ => TokenizerType::BPE, // 默认
            }
        } else {
            TokenizerType::BPE
        };
        
        Ok((tokenizer, tokenizer_type))
    }

    /// 加载 WordPiece tokenizer
    fn load_wordpiece_tokenizer(path: &Path) -> Result<(HFTokenizer, TokenizerType)> {
        let vocab_path = path.join("vocab.txt");
        let vocab = fs::read_to_string(&vocab_path)
            .with_context(|| format!("Failed to read vocab file from {:?}", vocab_path))?;
        
        let mut tokenizer = HFTokenizer::new(WordPiece::from_file(&vocab_path)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build WordPiece model: {:?}", e))?);
        
        Ok((tokenizer, TokenizerType::WordPiece))
    }

    /// 编码文本
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {:?}", e))?;
        
        let mut ids = encoding.get_ids().to_vec();
        
        if bos {
            if let Some(bos_id) = self.bos_id {
                ids.insert(0, bos_id);
            }
        }
        
        if eos {
            if let Some(eos_id) = self.eos_id {
                ids.push(eos_id);
            }
        }
        
        Ok(ids)
    }

    /// 编码文本段
    pub fn encode_segment(&self, text: &str) -> Result<Vec<u32>> {
        let text = text.trim_start();
        if self.need_space_before_segment {
            self.encode(&format!(" {}", text), false, false)
        } else {
            self.encode(text, false, false)
        }
    }

    /// 编码文本（不添加前缀空格）
    pub fn encode_wo_prefix_space(&self, text: &str) -> Result<Vec<u32>> {
        if self.need_space_before_segment {
            self.encode(text, false, false)
        } else {
            // 尝试找到一个前缀，使得它能独立地被分词
            let prefixes = ["@", "\n", "\\", "=", ">", "`"];
            
            for prefix in &prefixes {
                let prefix_tokens = self.encode(prefix, false, false)?;
                let cat_tokens = self.encode(&format!("{}{}", prefix, text), false, false)?;
                
                if cat_tokens.len() >= prefix_tokens.len() &&
                   &cat_tokens[..prefix_tokens.len()] == prefix_tokens.as_slice() {
                    return Ok(cat_tokens[prefix_tokens.len()..].to_vec());
                }
            }
            
            bail!(
                "All prefixes are merged into '{}' during tokenization. \
                This is weird behavior, please report this issue.",
                text
            )
        }
    }

    /// 解码 token IDs
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {:?}", e))
    }

    /// 批量编码（并行处理）
    pub fn batch_encode(&self, texts: &[&str], bos: bool, eos: bool) -> Result<Vec<Vec<u32>>> {
        texts.par_iter()
            .map(|text| self.encode(text, bos, eos))
            .collect()
    }

    /// 批量解码（并行处理）
    pub fn batch_decode(&self, ids_batch: &[&[u32]]) -> Result<Vec<String>> {
        ids_batch.par_iter()
            .map(|ids| self.decode(ids))
            .collect()
    }

    /// 保存 tokenizer
    pub fn save(&self, save_dir: &str) -> Result<()> {
        let dir = Path::new(save_dir);
        fs::create_dir_all(dir)?;
        
        let tokenizer_path = dir.join("tokenizer.json");
        self.tokenizer.save(&tokenizer_path, false)
            .map_err(|e| anyhow::anyhow!("Failed to save tokenizer: {:?}", e))?;
        
        // 保存配置
        let config = serde_json::json!({
            "tokenizer_type": match self.tokenizer_type {
                TokenizerType::BPE => "BPE",
                TokenizerType::WordPiece => "WordPiece",
                TokenizerType::Unigram => "Unigram",
                TokenizerType::SentencePiece => "SentencePiece",
            },
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "need_space_before_segment": self.need_space_before_segment,
        });
        
        let config_path = dir.join("tokenizer_config.json");
        fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
        
        Ok(())
    }

    /// 探测 tokenizer 风格
    fn probe_tokenizer_style(&mut self) {
        let sentence1 = self.encode("Hi my darling", false, false).unwrap_or_default();
        let sentence2 = self.encode("my darling", false, false).unwrap_or_default();
        
        if sentence1.len() >= sentence2.len() &&
           &sentence1[sentence1.len() - sentence2.len()..] == sentence2.as_slice() {
            self.need_space_before_segment = false;
        } else {
            let sentence3 = self.encode(" my darling", false, false).unwrap_or_default();
            if sentence1.len() >= sentence3.len() &&
               &sentence1[sentence1.len() - sentence3.len()..] == sentence3.as_slice() {
                self.need_space_before_segment = true;
            }
        }
    }

    // Getter 方法
    pub fn vocab_size(&self) -> u32 {
        self.tokenizer.get_vocab_size(true) as u32
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    pub fn tokenizer_type(&self) -> TokenizerType {
        self.tokenizer_type
    }

    pub fn need_space_before_segment(&self) -> bool {
        self.need_space_before_segment
    }
}