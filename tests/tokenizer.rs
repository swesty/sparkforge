//! `tokenizer::Decoder` on a small byte-level BPE `tokenizer.json` laid out like Llama 3's.
use sparkforge::tokenizer::Decoder;

const JSON: &str = r#"{
  "added_tokens": [{"id": 5, "content": "<|begin_of_text|>", "special": true}],
  "model": {"type": "BPE", "vocab": {"Hello": 0, "Ġworld": 1, ".": 2, "ĊĊ": 3, "Ã©": 4}, "merges": []}
}"#;

#[test]
fn byte_level_tokens_decode_to_text() {
    let d = Decoder::from_json(JSON).unwrap();
    assert_eq!(d.vocab_len(), 6);
    // Ġ is a space and Ċ a newline in GPT-2's byte table; Ã© is the two UTF-8 bytes of é.
    assert_eq!(d.decode(&[5, 0, 1, 2, 3]), "<|begin_of_text|>Hello world.\n\n");
    assert_eq!(d.decode(&[4]), "é");
    // An id the file doesn't define decodes to nothing.
    assert_eq!(d.decode(&[0, 99]), "Hello");
}

#[test]
fn only_bpe_tokenizers_are_read() {
    let e = Decoder::from_json(r#"{"model": {"type": "Unigram", "vocab": []}}"#).err().unwrap().to_string();
    assert!(e.contains("only BPE"), "{e}");
}
