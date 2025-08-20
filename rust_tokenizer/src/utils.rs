use std::path::Path;

/// 探测 tokenizer 路径
pub fn probe_tokenizer_path(pretrained_path: &str) -> Option<String> {
    let path = Path::new(pretrained_path);
    
    // 尝试查找 SentencePiece 风格的 tokenizer
    println!("Trying to find SentencePiece-style tokenizer at {:?}", path.join("tokenizer.model"));
    if path.join("tokenizer.model").exists() {
        println!("Found tokenizer.model, using it.");
        return Some(path.join("tokenizer.model").to_string_lossy().to_string());
    }
    println!("Not found");
    
    // 尝试查找 HuggingFace 风格的 tokenizer
    println!("Trying to find HuggingFace-style tokenizer at {:?}", path.join("tokenizer.json"));
    if path.join("tokenizer.json").exists() && path.join("tokenizer_config.json").exists() {
        println!("Found tokenizer.json and tokenizer_config.json, using them.");
        return Some(pretrained_path.to_string());
    }
    println!("Not found");
    
    // 尝试查找 WordPiece 风格的 tokenizer (BERT-style)
    println!("Trying to find WordPiece-style tokenizer at {:?}", path.join("vocab.txt"));
    if path.join("vocab.txt").exists() {
        println!("Found vocab.txt, using it.");
        return Some(pretrained_path.to_string());
    }
    println!("Not found");
    
    println!("No usable tokenizer found");
    None
}