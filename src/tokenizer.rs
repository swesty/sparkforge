//! Token ids to text for a byte-level BPE `tokenizer.json` (Llama 3, GPT-2 style): each vocabulary
//! entry is a string of stand-in characters, one per byte, mapped back through GPT-2's
//! `bytes_to_unicode` table; added (special) tokens decode to their content as written. Decoding
//! only: `generate` takes its prompt as token ids.
use anyhow::{Context, Result};
use std::collections::HashMap;

pub struct Decoder {
    /// Bytes of each token id the file defines. A map, not a vector indexed by id, so storage
    /// follows the entries in the file rather than the largest id it names.
    tokens: HashMap<u32, Vec<u8>>,
}

fn token_id(v: &serde_json::Value, what: &str) -> Result<u32> {
    let id = v.as_u64().with_context(|| format!("{what} id is not a non-negative integer"))?;
    u32::try_from(id).with_context(|| format!("{what} id {id} does not fit a token id"))
}

/// GPT-2's byte-to-character table, inverted: stand-in character -> byte.
fn char_to_byte() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter().zip(&cs).map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8)).collect()
}

impl Decoder {
    pub fn from_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text).context("tokenizer.json is not JSON")?;
        let model = v.get("model").context("tokenizer.json has no model")?;
        anyhow::ensure!(model.get("type").and_then(|t| t.as_str()) == Some("BPE"), "only BPE tokenizers are supported");
        let vocab = model.get("vocab").and_then(|v| v.as_object()).context("tokenizer.json has no model.vocab")?;
        let map = char_to_byte();
        let mut tokens = HashMap::new();
        for (s, id) in vocab {
            let id = token_id(id, "a vocabulary")?;
            let mut bytes = vec![];
            for c in s.chars() {
                match map.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => bytes.extend(c.to_string().as_bytes()),
                }
            }
            tokens.insert(id, bytes);
        }
        for t in v.get("added_tokens").and_then(|a| a.as_array()).into_iter().flatten() {
            let id = token_id(t.get("id").context("an added token has no id")?, "an added token")?;
            let content = t.get("content").and_then(|c| c.as_str()).context("an added token has no content")?;
            tokens.insert(id, content.as_bytes().to_vec());
        }
        Ok(Self { tokens })
    }

    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?)
    }

    /// Token ids the file defines.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The text of `ids`, invalid UTF-8 (a token boundary inside a character) shown as U+FFFD. An
    /// id the file doesn't define shows as `<unknown token ID>`, so a mismatched tokenizer can't
    /// print plausible but incomplete text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = vec![];
        for i in ids {
            match self.tokens.get(i) {
                Some(b) => bytes.extend_from_slice(b),
                None => bytes.extend_from_slice(format!("<unknown token {i}>").as_bytes()),
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The ids in `ids` the file doesn't define, in order of first appearance.
    pub fn unknown(&self, ids: &[u32]) -> Vec<u32> {
        let mut out: Vec<u32> = vec![];
        for &i in ids {
            if !self.tokens.contains_key(&i) && !out.contains(&i) {
                out.push(i);
            }
        }
        out
    }
}
