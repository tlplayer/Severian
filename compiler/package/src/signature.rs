use crate::PackageError;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub fn sha256_file(path: &Path) -> Result<String, PackageError> {
    let bytes = fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn verify_sha256(path: &Path, expected: &str) -> Result<(), PackageError> {
    validate_sha256(expected)?;
    let actual = sha256_file(path)?;
    if actual != expected.to_ascii_lowercase() {
        return Err(PackageError::Manifest(format!(
            "checksum mismatch for {}: expected {expected}, got {actual}",
            path.display()
        )));
    }
    Ok(())
}

pub fn verify_ed25519(
    signing_keys: &[String],
    payload: &[u8],
    encoded_signature: &str,
) -> Result<(), PackageError> {
    let signature_bytes = decode_hex(encoded_signature, 64, "signature")?;
    let signature = Signature::from_slice(&signature_bytes)
        .map_err(|_| PackageError::Manifest("invalid Ed25519 signature encoding".into()))?;
    for encoded_key in signing_keys {
        let Ok(bytes) = decode_hex(encoded_key, 32, "signing key") else {
            continue;
        };
        let Ok(key_bytes) = <[u8; 32]>::try_from(bytes.as_slice()) else {
            continue;
        };
        if VerifyingKey::from_bytes(&key_bytes)
            .is_ok_and(|key| key.verify(payload, &signature).is_ok())
        {
            return Ok(());
        }
    }
    Err(PackageError::Manifest(
        "external package signature is not valid for any trusted publisher key".into(),
    ))
}

pub fn validate_sha256(value: &str) -> Result<(), PackageError> {
    decode_hex(value, 32, "SHA-256 checksum").map(|_| ())
}

pub fn validate_ed25519_public_key(value: &str) -> Result<(), PackageError> {
    let bytes = decode_hex(value, 32, "Ed25519 public key")?;
    let key = <[u8; 32]>::try_from(bytes.as_slice())
        .map_err(|_| PackageError::Manifest("invalid Ed25519 public key".into()))?;
    VerifyingKey::from_bytes(&key)
        .map(|_| ())
        .map_err(|_| PackageError::Manifest("invalid Ed25519 public key".into()))
}

fn decode_hex(value: &str, expected_bytes: usize, label: &str) -> Result<Vec<u8>, PackageError> {
    if value.len() != expected_bytes * 2 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PackageError::Manifest(format!(
            "invalid {label}; expected {} hexadecimal characters",
            expected_bytes * 2
        )));
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("hexadecimal is ASCII");
            u8::from_str_radix(text, 16)
                .map_err(|_| PackageError::Manifest(format!("invalid {label}")))
        })
        .collect()
}
