//! Token ids to text for a byte-level BPE `tokenizer.json` (Llama 3, GPT-2 style): each vocabulary
//! entry is a string of stand-in characters, one per byte, mapped back through GPT-2's
//! `bytes_to_unicode` table; added (special) tokens decode to their content as written. Decoding
//! only: `generate` takes its prompt as token ids.
use anyhow::{Context, Result};
use std::collections::HashMap;

pub struct Decoder {
    /// Bytes of each token id; empty for an id the file doesn't define.
    tokens: Vec<Vec<u8>>,
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
        let mut tokens: Vec<Vec<u8>> = vec![];
        let mut put = |id: usize, bytes: Vec<u8>| {
            if tokens.len() <= id {
                tokens.resize(id + 1, vec![]);
            }
            tokens[id] = bytes;
        };
        for (s, id) in vocab {
            let id = id.as_u64().context("a vocabulary id is not a number")? as usize;
            let mut bytes = vec![];
            for c in s.chars() {
                match map.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => bytes.extend(c.to_string().as_bytes()),
                }
            }
            put(id, bytes);
        }
        for t in v.get("added_tokens").and_then(|a| a.as_array()).into_iter().flatten() {
            let id = t.get("id").and_then(|i| i.as_u64()).context("an added token has no id")? as usize;
            let content = t.get("content").and_then(|c| c.as_str()).context("an added token has no content")?;
            put(id, content.as_bytes().to_vec());
        }
        Ok(Self { tokens })
    }

    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?)
    }

    pub fn vocab_len(&self) -> usize {
        self.tokens.len()
    }

    /// The text of `ids`, invalid UTF-8 (a token boundary inside a character) shown as U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.tokens.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
