#![forbid(unsafe_code)]

/// Stable foreign-function ABI versions understood by the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AbiVersion {
    CV1,
}

impl AbiVersion {
    pub const fn manifest_name(self) -> &'static str {
        match self {
            Self::CV1 => "c-v1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallingConvention {
    C,
}

/// Types with a fixed representation in Severian's first C ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AbiType {
    Unit,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Usize,
    Isize,
    StringView,
    BytesView,
    Handle,
    OutHandle,
    OutError,
    OutUsize,
}

impl AbiType {
    pub const fn c_name(self) -> &'static str {
        match self {
            Self::Unit => "void",
            Self::Bool => "bool",
            Self::I8 => "int8_t",
            Self::I16 => "int16_t",
            Self::I32 => "int32_t",
            Self::I64 => "int64_t",
            Self::U8 => "uint8_t",
            Self::U16 => "uint16_t",
            Self::U32 => "uint32_t",
            Self::U64 => "uint64_t",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::Usize => "size_t",
            Self::Isize => "ptrdiff_t",
            Self::StringView => "sev_string_view_v1",
            Self::BytesView => "sev_bytes_view_v1",
            Self::Handle => "sev_handle_v1",
            Self::OutHandle => "sev_handle_v1 *",
            Self::OutError => "sev_error_v1 *",
            Self::OutUsize => "size_t *",
        }
    }

    pub const fn source_name(self) -> &'static str {
        match self {
            Self::Unit => "unit",
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Usize => "usize",
            Self::Isize => "isize",
            Self::StringView => "StringView",
            Self::BytesView => "BytesView",
            Self::Handle => "Handle",
            Self::OutHandle => "OutHandle",
            Self::OutError => "OutError",
            Self::OutUsize => "OutUsize",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ownership {
    Copy,
    Borrowed,
    Owned,
    Out,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbiParameter {
    pub name: String,
    pub ty: AbiType,
    pub ownership: Ownership,
    pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbiResult {
    pub ty: AbiType,
    pub ownership: Ownership,
    pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExternalFunction {
    pub package: String,
    pub function: String,
    pub symbol: String,
    pub shim_symbol: String,
    pub abi: AbiVersion,
    pub calling_convention: CallingConvention,
    pub parameters: Vec<AbiParameter>,
    pub result: AbiResult,
}

pub const C_V1_PREAMBLE: &str = r#"#ifndef SEVERIAN_C_V1_H
#define SEVERIAN_C_V1_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const uint8_t *data;
    size_t length;
} sev_string_view_v1;

typedef struct {
    const uint8_t *data;
    size_t length;
} sev_bytes_view_v1;

typedef struct {
    void *value;
} sev_handle_v1;

typedef struct {
    int32_t code;
    sev_string_view_v1 message;
} sev_error_v1;

#endif
"#;

pub fn c_v1_header<'a>(functions: impl IntoIterator<Item = &'a ExternalFunction>) -> String {
    let mut output = C_V1_PREAMBLE.replace("\n#endif\n", "\n");
    for function in functions {
        output.push_str(function.result.ty.c_name());
        output.push(' ');
        output.push_str(&function.symbol);
        output.push('(');
        if function.parameters.is_empty() {
            output.push_str("void");
        } else {
            for (index, parameter) in function.parameters.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                output.push_str(parameter.ty.c_name());
                output.push(' ');
                output.push_str(&parameter.name);
            }
        }
        output.push_str(");\n");
    }
    output.push_str("\n#endif\n");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_header_uses_only_stable_c_types() {
        let function = ExternalFunction {
            package: "network".into(),
            function: "connect_raw".into(),
            symbol: "sev_abi_v1_network_connect".into(),
            shim_symbol: "__sev_ffi_shim_sev_abi_v1_network_connect".into(),
            abi: AbiVersion::CV1,
            calling_convention: CallingConvention::C,
            parameters: vec![
                AbiParameter {
                    name: "host".into(),
                    ty: AbiType::StringView,
                    ownership: Ownership::Borrowed,
                    nullable: false,
                },
                AbiParameter {
                    name: "connection".into(),
                    ty: AbiType::OutHandle,
                    ownership: Ownership::Out,
                    nullable: false,
                },
            ],
            result: AbiResult {
                ty: AbiType::I32,
                ownership: Ownership::Copy,
                nullable: false,
            },
        };
        let header = c_v1_header([&function]);
        assert!(header.contains("sev_string_view_v1 host"));
        assert!(header.contains("sev_handle_v1 * connection"));
        assert!(!header.contains("sev_value"));
    }
}
