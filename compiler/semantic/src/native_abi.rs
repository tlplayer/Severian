use severian_abi::{
    AbiParameter, AbiResult, AbiType, AbiVersion, CallingConvention, ExternalFunction, Ownership,
};
use severian_ast::{Item, Type};
use severian_package::PackageInterface;

use crate::SemanticError;

/// Validates declarations owned by a package that opted into a declarative
/// native unit. Legacy compiler-runtime declarations are intentionally outside
/// this incremental c-v1 gate until their packages migrate.
pub fn validate_native_abi(
    interface: &PackageInterface,
) -> Result<Vec<ExternalFunction>, SemanticError> {
    if interface.native_units.is_empty() {
        return Ok(Vec::new());
    }
    let package = interface
        .native_units
        .first()
        .map(|unit| unit.package.clone())
        .unwrap_or_else(|| interface.name.clone());
    let mut symbols = std::collections::HashSet::new();
    let mut functions = Vec::new();
    for item in &interface.module.items {
        let Item::Function(function) = item else {
            continue;
        };
        let Some(symbol) = &function.native_symbol else {
            continue;
        };
        if !valid_c_symbol(symbol) || !symbol.starts_with("sev_abi_v1_") {
            return Err(error(
                function.name.span,
                format!(
                    "E0802: native symbol `{symbol}` is not a c-v1 provider symbol\nhelp: use a unique C identifier beginning with `sev_abi_v1_`"
                ),
            ));
        }
        if !symbols.insert(symbol.clone()) {
            return Err(error(
                function.name.span,
                format!("E0803: duplicate c-v1 native symbol `{symbol}`"),
            ));
        }
        if !function.generic_params.is_empty() {
            return Err(error(
                function.name.span,
                "E0801: generic functions are not C-ABI-safe\nhelp: declare a concrete c-v1 function and wrap it in generic Severian code",
            ));
        }
        let parameters = function
            .params
            .iter()
            .map(|parameter| {
                if parameter.default.is_some() {
                    return Err(error(
                        parameter.span,
                        "E0801: default parameters are not C-ABI-safe",
                    ));
                }
                let declared = parameter.ty.as_ref().ok_or_else(|| {
                    error(
                        parameter.span,
                        "E0801: an untyped parameter is not C-ABI-safe",
                    )
                })?;
                let ty = abi_type(declared, false)?;
                if ty == AbiType::Unit {
                    return Err(error(
                        declared.span(),
                        "E0801: `unit` is not a valid C ABI parameter",
                    ));
                }
                Ok(AbiParameter {
                    name: parameter.name.name.clone(),
                    ty,
                    ownership: ownership(ty),
                    nullable: false,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result_ty = match function.return_type.as_ref() {
            Some(ty) => abi_type(ty, true)?,
            None => AbiType::Unit,
        };
        if matches!(
            result_ty,
            AbiType::Handle
                | AbiType::OutHandle
                | AbiType::OutError
                | AbiType::OutUsize
                | AbiType::BytesView
        ) {
            return Err(error(
                function
                    .return_type
                    .as_ref()
                    .map(Type::span)
                    .unwrap_or(function.name.span),
                format!(
                    "E0801: `{}` is not a valid c-v1 return type",
                    result_ty.source_name()
                ),
            ));
        }
        functions.push(ExternalFunction {
            package: package.clone(),
            function: format!("{}.{}", interface.name, function.name.name),
            symbol: symbol.clone(),
            shim_symbol: format!("__sev_ffi_shim_{symbol}"),
            abi: AbiVersion::CV1,
            calling_convention: CallingConvention::C,
            parameters,
            result: AbiResult {
                ty: result_ty,
                ownership: ownership(result_ty),
                nullable: false,
            },
        });
    }
    Ok(functions)
}

fn abi_type(ty: &Type, returning: bool) -> Result<AbiType, SemanticError> {
    let Type::Named(path) = ty else {
        return Err(not_safe(ty, "composite type"));
    };
    let name = path
        .segments
        .last()
        .map(|segment| segment.name.as_str())
        .unwrap_or("");
    let abi = match name {
        "unit" => AbiType::Unit,
        "bool" => AbiType::Bool,
        "i8" => AbiType::I8,
        "i16" => AbiType::I16,
        "i32" => AbiType::I32,
        "i64" => AbiType::I64,
        "u8" => AbiType::U8,
        "u16" => AbiType::U16,
        "u32" => AbiType::U32,
        "u64" => AbiType::U64,
        "f32" => AbiType::F32,
        "f64" => AbiType::F64,
        "usize" => AbiType::Usize,
        "isize" => AbiType::Isize,
        // Inputs use the explicit FFI wrapper. Provider-owned views returned
        // from C are copied into ordinary Severian strings.
        "StringView" if !returning => AbiType::StringView,
        "string" if returning => AbiType::StringView,
        "BytesView" if !returning => AbiType::BytesView,
        "Handle" => AbiType::Handle,
        "OutHandle" if !returning => AbiType::OutHandle,
        "OutError" if !returning => AbiType::OutError,
        "OutUsize" if !returning => AbiType::OutUsize,
        _ => return Err(not_safe(ty, name)),
    };
    Ok(abi)
}

fn ownership(ty: AbiType) -> Ownership {
    match ty {
        AbiType::StringView | AbiType::BytesView | AbiType::Handle => Ownership::Borrowed,
        AbiType::OutHandle | AbiType::OutError | AbiType::OutUsize => Ownership::Out,
        _ => Ownership::Copy,
    }
}

fn not_safe(ty: &Type, name: &str) -> SemanticError {
    let help = if name == "Any" {
        "use an opaque `ffi.Handle[T]` or an explicit byte/string view"
    } else {
        "use a fixed-width scalar, an opaque Handle, or an explicit byte/string view"
    };
    error(
        ty.span(),
        format!("E0801: `{name}` is not C-ABI-safe\nhelp: {help}"),
    )
}

fn valid_c_symbol(symbol: &str) -> bool {
    let mut bytes = symbol.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

fn error(span: severian_ast::Span, message: impl Into<String>) -> SemanticError {
    SemanticError {
        span,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_package::{NativeLanguage, NativeUnit, TargetPattern};
    use std::path::PathBuf;

    fn interface(source: &str) -> PackageInterface {
        let tokens = severian_lexer::lex(source).unwrap();
        PackageInterface {
            name: "ffi".into(),
            export_package: Some("network".into()),
            module: severian_parser::parse(&tokens).unwrap(),
            compiler: Default::default(),
            native_units: vec![NativeUnit {
                package: "network".into(),
                name: "network-posix".into(),
                abi: AbiVersion::CV1,
                language: NativeLanguage::C,
                sources: vec![PathBuf::from("tcp.c")],
                include_directories: Vec::new(),
                libraries: Vec::new(),
                targets: vec![TargetPattern("linux".into())],
            }],
            native_assets: Vec::new(),
            source_path: PathBuf::from("ffi.sev"),
            source: source.into(),
        }
    }

    #[test]
    fn rejects_dynamic_values_at_c_v1_boundaries() {
        let source = "unsafe:\n    extern(\"sev_abi_v1_bad\") def bad(value: Any) -> Any\n";
        let error = validate_native_abi(&interface(source)).unwrap_err();
        assert!(error.message.contains("`Any` is not C-ABI-safe"));
        assert!(error.message.contains("ffi.Handle"));
    }

    #[test]
    fn records_real_out_parameter_signature() {
        let source = "import ffi\n\nunsafe:\n    extern(\"sev_abi_v1_network_connect\") def connect_raw(host: ffi.StringView, port: u16, connection: ffi.OutHandle, error: ffi.OutError) -> i32\n";
        let functions = validate_native_abi(&interface(source)).unwrap();
        assert_eq!(functions[0].parameters[0].ty, AbiType::StringView);
        assert_eq!(functions[0].parameters[2].ty, AbiType::OutHandle);
        assert_eq!(functions[0].result.ty, AbiType::I32);
    }
}
