#!/usr/bin/env python3
"""
Rust Tokenizer 单元测试
"""

import pytest
from pathlib import Path
import tempfile
import json

# 注意：实际测试时需要先构建并安装 rust_tokenizer
# 运行: maturin develop --release

def test_import():
    """测试模块导入"""
    from rust_tokenizer import RustTokenizerWrapper, probe_tokenizer_path_from_pretrained
    assert RustTokenizerWrapper is not None
    assert probe_tokenizer_path_from_pretrained is not None


class TestRustTokenizer:
    """Rust Tokenizer 测试类"""
    
    @pytest.fixture
    def mock_tokenizer_path(self, tmp_path):
        """创建模拟的 tokenizer 文件"""
        # 创建一个简单的 tokenizer.json 文件
        tokenizer_config = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "[UNK]",
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "vocab": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[CLS]": 2,
                    "[SEP]": 3,
                    "[MASK]": 4,
                    "hello": 5,
                    "world": 6,
                    "test": 7,
                },
                "merges": []
            }
        }
        
        tokenizer_path = tmp_path / "tokenizer.json"
        with open(tokenizer_path, 'w') as f:
            json.dump(tokenizer_config, f)
        
        # 创建 tokenizer_config.json
        config = {
            "model_type": "BPE",
            "bos_token": "[CLS]",
            "eos_token": "[SEP]",
        }
        config_path = tmp_path / "tokenizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        return str(tmp_path)
    
    def test_tokenizer_creation(self, mock_tokenizer_path):
        """测试 tokenizer 创建"""
        from rust_tokenizer import RustTokenizerWrapper
        
        # 注意：这个测试需要实际的 tokenizer 文件
        # 在实际测试中，您需要提供真实的模型文件
        # tokenizer = RustTokenizerWrapper(mock_tokenizer_path)
        # assert tokenizer is not None
        # assert tokenizer.n_words > 0
        pass  # 暂时跳过，需要真实模型文件
    
    def test_encode_decode(self):
        """测试编码和解码"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass
    
    def test_batch_operations(self):
        """测试批量操作"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass
    
    def test_probe_tokenizer_path(self, tmp_path):
        """测试 tokenizer 路径探测"""
        from rust_tokenizer import probe_tokenizer_path_from_pretrained
        
        # 创建不同类型的 tokenizer 文件
        # SentencePiece 风格
        spm_path = tmp_path / "spm_model"
        spm_path.mkdir()
        (spm_path / "tokenizer.model").touch()
        
        result = probe_tokenizer_path_from_pretrained(str(spm_path))
        assert result is not None
        assert "tokenizer.model" in result
        
        # HuggingFace 风格
        hf_path = tmp_path / "hf_model"
        hf_path.mkdir()
        (hf_path / "tokenizer.json").touch()
        (hf_path / "tokenizer_config.json").touch()
        
        result = probe_tokenizer_path_from_pretrained(str(hf_path))
        assert result is not None
        
        # WordPiece 风格
        wp_path = tmp_path / "wp_model"
        wp_path.mkdir()
        (wp_path / "vocab.txt").touch()
        
        result = probe_tokenizer_path_from_pretrained(str(wp_path))
        assert result is not None
        
        # 不存在的路径
        result = probe_tokenizer_path_from_pretrained(str(tmp_path / "nonexistent"))
        assert result is None
    
    def test_tokenizer_properties(self):
        """测试 tokenizer 属性"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass
    
    def test_special_tokens(self):
        """测试特殊 token 处理"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass
    
    def test_segment_encoding(self):
        """测试段编码"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass
    
    def test_save_load(self, tmp_path):
        """测试保存和加载"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass


class TestPerformance:
    """性能测试"""
    
    @pytest.mark.benchmark
    def test_encoding_performance(self):
        """测试编码性能"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass
    
    @pytest.mark.benchmark
    def test_batch_encoding_performance(self):
        """测试批量编码性能"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass
    
    @pytest.mark.benchmark
    def test_decoding_performance(self):
        """测试解码性能"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass


class TestCompatibility:
    """兼容性测试"""
    
    def test_api_compatibility(self):
        """测试 API 兼容性"""
        from rust_tokenizer import RustTokenizerWrapper
        
        # 检查必要的方法存在
        assert hasattr(RustTokenizerWrapper, 'encode')
        assert hasattr(RustTokenizerWrapper, 'decode')
        assert hasattr(RustTokenizerWrapper, 'encode_segment')
        assert hasattr(RustTokenizerWrapper, 'encode_wo_prefix_space')
        assert hasattr(RustTokenizerWrapper, 'save')
    
    def test_chinese_text(self):
        """测试中文文本处理"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass
    
    def test_mixed_language(self):
        """测试混合语言文本"""
        # 需要真实的 tokenizer 模型文件来运行这个测试
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])