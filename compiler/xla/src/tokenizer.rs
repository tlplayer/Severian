use serde_json::Value;
use std::{
    collections::HashMap,
    ffi::{c_char, c_void, CStr, CString},
    fs,
    path::Path,
    sync::{Mutex, OnceLock},
};

struct ByteBpe {
    vocab: HashMap<String, i64>,
    tokens: HashMap<i64, String>,
    merges: HashMap<(String, String), usize>,
    byte_encoder: Vec<char>,
    byte_decoder: HashMap<char, u8>,
}

static TOKENIZERS: OnceLock<Mutex<HashMap<String, ByteBpe>>> = OnceLock::new();

unsafe extern "C" {
    fn __sev_collection_new(kind: i64) -> *mut c_void;
    fn __sev_collection_push(collection: *mut c_void, value: *mut c_void);
    fn __sev_box_i64(value: i64) -> *mut c_void;
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("Severian tokenizer error: {message}");
    std::process::abort()
}

fn bytes_to_unicode() -> (Vec<char>, HashMap<char, u8>) {
    let mut bytes = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect::<Vec<_>>();
    let mut codepoints = bytes.iter().map(|&byte| u32::from(byte)).collect::<Vec<_>>();
    let mut extra = 0_u32;
    for byte in 0_u8..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + extra);
            extra += 1;
        }
    }
    let mut encoder = vec!['\0'; 256];
    let mut decoder = HashMap::new();
    for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
        let encoded = char::from_u32(codepoint).unwrap();
        encoder[byte as usize] = encoded;
        decoder.insert(encoded, byte);
    }
    (encoder, decoder)
}

impl ByteBpe {
    fn load(model_path: &str) -> Result<Self, String> {
        let vocab_text = fs::read_to_string(Path::new(model_path).join("vocab.json"))
            .map_err(|error| format!("could not read vocab.json: {error}"))?;
        let vocab_value: Value = serde_json::from_str(&vocab_text)
            .map_err(|error| format!("invalid vocab.json: {error}"))?;
        let vocab_object = vocab_value
            .as_object()
            .ok_or_else(|| "vocab.json is not an object".to_string())?;
        let mut vocab = HashMap::with_capacity(vocab_object.len());
        let mut tokens = HashMap::with_capacity(vocab_object.len());
        for (token, id) in vocab_object {
            let id = id
                .as_i64()
                .ok_or_else(|| format!("token {token:?} has a non-integer id"))?;
            vocab.insert(token.clone(), id);
            tokens.insert(id, token.clone());
        }

        let merges_text = fs::read_to_string(Path::new(model_path).join("merges.txt"))
            .map_err(|error| format!("could not read merges.txt: {error}"))?;
        let mut merges = HashMap::new();
        for (rank, line) in merges_text.lines().enumerate() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((left, right)) = line.split_once(' ') else {
                return Err(format!("invalid BPE merge at line {}", rank + 1));
            };
            merges.insert((left.to_string(), right.to_string()), rank);
        }
        let (byte_encoder, byte_decoder) = bytes_to_unicode();
        Ok(Self {
            vocab,
            tokens,
            merges,
            byte_encoder,
            byte_decoder,
        })
    }

    fn pretokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let positions = text
            .char_indices()
            .map(|(offset, _)| offset)
            .chain(std::iter::once(text.len()))
            .collect::<Vec<_>>();
        let chars = text.chars().collect::<Vec<_>>();
        let mut pieces = Vec::new();
        let mut cursor = 0;
        while cursor < chars.len() {
            let start = cursor;

            if chars[cursor] == '\'' {
                let rest = chars[cursor + 1..]
                    .iter()
                    .take(2)
                    .collect::<String>()
                    .to_ascii_lowercase();
                let width = ["s", "t", "m", "d"]
                    .iter()
                    .find(|suffix| rest.starts_with(**suffix))
                    .map(|suffix| suffix.len())
                    .or_else(|| ["re", "ve", "ll"].iter().find(|suffix| rest.starts_with(**suffix)).map(|_| 2));
                if let Some(width) = width {
                    cursor += 1 + width;
                    pieces.push(&text[positions[start]..positions[cursor]]);
                    continue;
                }
            }

            let prefix = !matches!(chars[cursor], '\r' | '\n')
                && !chars[cursor].is_alphabetic()
                && !chars[cursor].is_numeric()
                && cursor + 1 < chars.len()
                && chars[cursor + 1].is_alphabetic();
            if chars[cursor].is_alphabetic() || prefix {
                if prefix {
                    cursor += 1;
                }
                while cursor < chars.len() && chars[cursor].is_alphabetic() {
                    cursor += 1;
                }
                pieces.push(&text[positions[start]..positions[cursor]]);
                continue;
            }

            if chars[cursor].is_numeric() {
                cursor += 1;
                pieces.push(&text[positions[start]..positions[cursor]]);
                continue;
            }

            let space_before_punctuation = chars[cursor] == ' '
                && cursor + 1 < chars.len()
                && !chars[cursor + 1].is_whitespace()
                && !chars[cursor + 1].is_alphabetic()
                && !chars[cursor + 1].is_numeric();
            let punctuation = !chars[cursor].is_whitespace()
                && !chars[cursor].is_alphabetic()
                && !chars[cursor].is_numeric();
            if punctuation || space_before_punctuation {
                if space_before_punctuation {
                    cursor += 1;
                }
                while cursor < chars.len()
                    && !chars[cursor].is_whitespace()
                    && !chars[cursor].is_alphabetic()
                    && !chars[cursor].is_numeric()
                {
                    cursor += 1;
                }
                while cursor < chars.len() && matches!(chars[cursor], '\r' | '\n') {
                    cursor += 1;
                }
                pieces.push(&text[positions[start]..positions[cursor]]);
                continue;
            }

            if chars[cursor].is_whitespace() {
                cursor += 1;
                while cursor < chars.len() && chars[cursor].is_whitespace() {
                    cursor += 1;
                }
                pieces.push(&text[positions[start]..positions[cursor]]);
                continue;
            }

            cursor += 1;
            pieces.push(&text[positions[start]..positions[cursor]]);
        }
        pieces
    }

    fn encode_piece(&self, piece: &str) -> Vec<i64> {
        let encoded = piece
            .as_bytes()
            .iter()
            .map(|byte| self.byte_encoder[*byte as usize])
            .collect::<String>();
        let mut symbols = encoded.chars().map(String::from).collect::<Vec<_>>();
        while symbols.len() > 1 {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    self.merges
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (index, *rank))
                })
                .min_by_key(|(_, rank)| *rank);
            let Some((index, _)) = best else { break };
            let right = symbols.remove(index + 1);
            symbols[index].push_str(&right);
        }
        symbols
            .iter()
            .map(|symbol| {
                *self
                    .vocab
                    .get(symbol)
                    .unwrap_or_else(|| fail(format!("BPE produced unknown token {symbol:?}")))
            })
            .collect()
    }

    fn encode(&self, text: &str) -> Vec<i64> {
        self.pretokenize(text)
            .into_iter()
            .flat_map(|piece| self.encode_piece(piece))
            .collect()
    }

    fn decode(&self, ids: &[i64]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            let token = self
                .tokens
                .get(id)
                .unwrap_or_else(|| fail(format!("unknown token id {id}")));
            for character in token.chars() {
                bytes.push(
                    *self
                        .byte_decoder
                        .get(&character)
                        .unwrap_or_else(|| fail(format!("token {id} is not byte-level text"))),
                );
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn with_tokenizer<T>(model_path: &str, operation: impl FnOnce(&ByteBpe) -> T) -> T {
    let mut tokenizers = TOKENIZERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !tokenizers.contains_key(model_path) {
        let tokenizer = ByteBpe::load(model_path).unwrap_or_else(|error| fail(error));
        tokenizers.insert(model_path.to_string(), tokenizer);
    }
    operation(tokenizers.get(model_path).unwrap())
}

unsafe fn input_string<'a>(pointer: *const c_char, name: &str) -> &'a str {
    if pointer.is_null() {
        fail(format!("null {name}"));
    }
    CStr::from_ptr(pointer)
        .to_str()
        .unwrap_or_else(|error| fail(format!("invalid UTF-8 in {name}: {error}")))
}

#[no_mangle]
pub unsafe extern "C" fn __sev_qwen_tokenize(
    model_path: *const c_char,
    text: *const c_char,
) -> *mut c_void {
    let model_path = input_string(model_path, "model path");
    let text = input_string(text, "tokenizer input");
    let ids = with_tokenizer(model_path, |tokenizer| tokenizer.encode(text));
    let result = __sev_collection_new(0);
    for id in ids {
        __sev_collection_push(result, __sev_box_i64(id));
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn __sev_qwen_decode_token(
    model_path: *const c_char,
    id: i64,
) -> *mut c_char {
    let model_path = input_string(model_path, "model path");
    let decoded = with_tokenizer(model_path, |tokenizer| tokenizer.decode(&[id]));
    CString::new(decoded)
        .unwrap_or_else(|error| fail(format!("decoded token contains NUL: {error}")))
        .into_raw()
}

#[cfg(test)]
mod tests {
    use super::ByteBpe;

    #[test]
    fn qwen_reference_phrase() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../benchmarks/inference/models/Qwen2.5-3B-Instruct");
        if !std::path::Path::new(path).join("vocab.json").exists() {
            return;
        }
        let tokenizer = ByteBpe::load(path).unwrap();
        assert_eq!(tokenizer.encode("Roses are red,"), [49, 19696, 525, 2518, 11]);
        assert_eq!(tokenizer.decode(&[348]), " v");
    }
}
