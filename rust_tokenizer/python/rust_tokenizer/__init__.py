"""
Rust Tokenizer - 高性能 Rust 实现的 tokenizer，提供 Python 接口

这个包提供了一个用 Rust 编写的高性能 tokenizer，支持多种分词模型：
- BPE (Byte Pair Encoding)
- WordPiece
- Unigram (SentencePiece)
- 自定义模型

主要特性：
- 高性能：比纯 Python 实现快 10-100 倍
- 内存安全：利用 Rust 的内存安全保证
- 并行处理：支持批量编码/解码的并行处理
- 兼容性：与现有的 HuggingFace 和 SentencePiece 模型兼容
"""

from .rust_tokenizer import PyTokenizer as Tokenizer, probe_tokenizer_path
from typing import List, Optional, Union
import logging

__version__ = "0.1.0"
__all__ = ["Tokenizer", "RustTokenizerWrapper", "probe_tokenizer_path_from_pretrained"]

logger = logging.getLogger(__name__)


class RustTokenizerWrapper:
    """
    Python 包装器，提供与原始 Python tokenizer 兼容的接口
    """
    
    def __init__(self, model_path: str):
        """
        创建 tokenizer
        
        Args:
            model_path: tokenizer 模型路径
                - .model 文件：SentencePiece 模型
                - 包含 tokenizer.json 的目录：HuggingFace tokenizer
                - 包含 vocab.txt 的目录：WordPiece tokenizer
        """
        self._tokenizer = Tokenizer(model_path)
        self.tokenizer_type = self._tokenizer.tokenizer_type
        
        # 兼容性属性
        self.bos_id = self._tokenizer.bos_id
        self.eos_id = self._tokenizer.eos_id
        self.n_words = self._tokenizer.n_words
        self.need_space_before_segment = self._tokenizer.need_space_before_segment
        
        logger.info(
            f"Loaded Rust tokenizer - Type: {self.tokenizer_type} | "
            f"Vocab size: {self.n_words} | BOS: {self.bos_id} | EOS: {self.eos_id}"
        )
    
    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """编码文本为 token IDs"""
        return self._tokenizer.encode(text, bos, eos)
    
    def decode(self, ids: List[int]) -> str:
        """解码 token IDs 为文本"""
        return self._tokenizer.decode(ids)
    
    def encode_segment(self, text: str) -> List[int]:
        """编码文本段（处理前导空格）"""
        return self._tokenizer.encode_segment(text)
    
    def encode_wo_prefix_space(self, text: str) -> List[int]:
        """编码文本（不添加前缀空格）"""
        return self._tokenizer.encode_wo_prefix_space(text)
    
    def batch_encode(
        self, 
        texts: List[str], 
        bos: bool = False, 
        eos: bool = False
    ) -> List[List[int]]:
        """批量编码多个文本（并行处理）"""
        return self._tokenizer.batch_encode(texts, bos, eos)
    
    def batch_decode(self, ids_batch: List[List[int]]) -> List[str]:
        """批量解码多个 token ID 序列（并行处理）"""
        return self._tokenizer.batch_decode(ids_batch)
    
    def save(self, save_dir: str):
        """保存 tokenizer 到指定目录"""
        self._tokenizer.save(save_dir)
        logger.info(f"Saved tokenizer to {save_dir}")
    
    @property
    def vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.n_words
    
    def __repr__(self) -> str:
        return (
            f"RustTokenizerWrapper(type={self.tokenizer_type}, "
            f"vocab_size={self.n_words}, bos_id={self.bos_id}, eos_id={self.eos_id})"
        )


def probe_tokenizer_path_from_pretrained(pretrained_path: str) -> Optional[str]:
    """
    探测预训练模型目录中的 tokenizer 路径
    
    Args:
        pretrained_path: 预训练模型目录路径
    
    Returns:
        找到的 tokenizer 路径，如果没找到则返回 None
    """
    return probe_tokenizer_path(pretrained_path)