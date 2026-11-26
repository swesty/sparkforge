//! `tokenizer::Decoder` on a small byte-level BPE `tokenizer.json` laid out like Llama 3's.
use sparkforge::tokenizer::Decoder;

const JSON: &str = r#"{
  "added_tokens": [{"id": 5, "content": "<|begin_of_text|>", "special": true}],
  "model": {"type": "BPE", "vocab": {"Hello": 0, "Ġworld": 1, ".": 2, "ĊĊ": 3, "Ã©": 4}, "merges": []}
}"#;

#[test]
fn byte_level_tokens_decode_to_text() {
    let d = Decoder::from_json(JSON).unwrap();
    assert_eq!(d.len(), 6);
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

#[test]
fn a_huge_token_id_costs_no_memory_and_an_out_of_range_one_is_refused() {
    // One entry with id 1e9: stored as one map entry, not a billion-element table.
    let d = Decoder::from_json(r#"{"model": {"type": "BPE", "vocab": {"x": 1000000000}}}"#).unwrap();
    assert_eq!(d.len(), 1);
    assert_eq!(d.decode(&[1_000_000_000]), "x");
    let e = Decoder::from_json(r#"{"model": {"type": "BPE", "vocab": {"x": 4294967296}}}"#).err().unwrap().to_string();
    assert!(e.contains("does not fit"), "{e}");
    let e = Decoder::from_json(r#"{"model": {"type": "BPE", "vocab": {"x": -1}}}"#).err().unwrap().to_string();
    assert!(e.contains("non-negative"), "{e}");
}
