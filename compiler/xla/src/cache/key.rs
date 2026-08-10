use crate::{
    options::{serialize_compile_options, XlaCompileOptions},
    stablehlo::StableHloModule,
};
use std::fmt;

/// Deterministic non-cryptographic 128-bit cache key.
///
/// This intentionally does not use `DefaultHasher`, whose algorithm is not a
/// stable persistence contract. Two independently seeded FNV-1a streams give a
/// compact cache key without introducing a hashing dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheKey {
    high: u64,
    low: u64,
}

impl CacheKey {
    pub const fn new(high: u64, low: u64) -> Self {
        Self { high, low }
    }

    pub const fn high(self) -> u64 {
        self.high
    }

    pub const fn low(self) -> u64 {
        self.low
    }

    pub fn hex(self) -> String {
        format!("{:016x}{:016x}", self.high, self.low)
    }

    pub fn shard(self) -> String {
        format!("{:02x}", (self.high >> 56) as u8)
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.hex())
    }
}

pub struct CacheKeyBuilder {
    high: Fnv64,
    low: Fnv64,
}

impl Default for CacheKeyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheKeyBuilder {
    pub fn new() -> Self {
        Self {
            high: Fnv64::new(0xcbf29ce484222325),
            low: Fnv64::new(0x84222325cbf29ce4),
        }
    }

    pub fn field(mut self, name: &str, bytes: impl AsRef<[u8]>) -> Self {
        let bytes = bytes.as_ref();

        self.write_len(name.len());
        self.write(name.as_bytes());
        self.write_len(bytes.len());
        self.write(bytes);
        self
    }

    pub fn text(self, name: &str, value: &str) -> Self {
        self.field(name, value.as_bytes())
    }

    pub fn integer(self, name: &str, value: i64) -> Self {
        self.field(name, value.to_le_bytes())
    }

    pub fn boolean(self, name: &str, value: bool) -> Self {
        self.field(name, [u8::from(value)])
    }

    pub fn stablehlo(self, module: &StableHloModule) -> Self {
        self.field("stablehlo", module.bytes())
    }

    pub fn compile_options(
        self,
        options: &XlaCompileOptions,
    ) -> Result<Self, crate::XlaError> {
        let bytes = serialize_compile_options(options)
            .map_err(|error| crate::XlaError::Pjrt(error.to_string()))?;
        Ok(self.field("compile-options", bytes))
    }

    pub fn platform(
        self,
        name: &str,
        version: &str,
    ) -> Self {
        self.text("platform-name", name)
            .text("platform-version", version)
    }

    pub fn topology(self, serialized: Option<&[u8]>) -> Self {
        match serialized {
            Some(bytes) => self.field("topology", bytes),
            None => self.text("topology", "<unavailable>"),
        }
    }

    pub fn pjrt_api(self, major: i32, minor: i32) -> Self {
        self.integer("pjrt-api-major", i64::from(major))
            .integer("pjrt-api-minor", i64::from(minor))
    }

    pub fn severian_version(self, version: &str) -> Self {
        self.text("severian-version", version)
    }

    pub fn finish(self) -> CacheKey {
        CacheKey::new(self.high.finish(), self.low.finish())
    }

    fn write_len(&mut self, len: usize) {
        self.write(&(len as u64).to_le_bytes());
    }

    fn write(&mut self, bytes: &[u8]) {
        self.high.write(bytes);
        // Reverse bytes into the second stream so trivial prefixes don't
        // produce tightly correlated halves.
        for byte in bytes.iter().rev() {
            self.low.write_byte(*byte);
        }
    }
}

#[derive(Clone, Copy)]
struct Fnv64 {
    state: u64,
}

impl Fnv64 {
    const PRIME: u64 = 0x00000100000001B3;

    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_byte(byte);
        }
    }

    fn write_byte(&mut self, byte: u8) {
        self.state ^= u64::from(byte);
        self.state = self.state.wrapping_mul(Self::PRIME);
    }

    const fn finish(self) -> u64 {
        self.state
    }
}
