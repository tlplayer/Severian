use severian_abi::{AbiFloatFormat, AbiType, ScalarType};
use severian_ffi::{BoundaryPlan, ForeignTypeRef, ParameterMode};
use std::collections::BTreeSet;

/// Generic source-storage to foreign-view bridge, selected by contracts rather
/// than a function name or the domain library that happens to call it.
pub fn bridge_symbol(plan: &BoundaryPlan) -> Option<String> {
    if !plan
        .parameters
        .iter()
        .any(|parameter| matches!(parameter.contract.ty, ForeignTypeRef::Sequence { .. }))
    {
        return None;
    }
    let identity = format!("{:?}", plan);
    let hash = identity.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    Some(format!("__sev_xxi_bytes_{hash:016x}"))
}

fn scalar(ty: &AbiType) -> Result<&'static str, String> {
    Ok(match ty {
        AbiType::Void => "void",
        AbiType::Scalar(ScalarType::Boolean) => "_Bool",
        AbiType::Scalar(ScalarType::Integer {
            bits: 8,
            signed: true,
        }) => "int8_t",
        AbiType::Scalar(ScalarType::Integer {
            bits: 8,
            signed: false,
        }) => "uint8_t",
        AbiType::Scalar(ScalarType::Integer {
            bits: 16,
            signed: true,
        }) => "int16_t",
        AbiType::Scalar(ScalarType::Integer {
            bits: 16,
            signed: false,
        }) => "uint16_t",
        AbiType::Scalar(ScalarType::Integer {
            bits: 32,
            signed: true,
        }) => "int32_t",
        AbiType::Scalar(ScalarType::Integer {
            bits: 32,
            signed: false,
        }) => "uint32_t",
        AbiType::Scalar(ScalarType::Integer {
            bits: 64,
            signed: true,
        }) => "int64_t",
        AbiType::Scalar(ScalarType::Integer {
            bits: 64,
            signed: false,
        }) => "uint64_t",
        AbiType::Scalar(ScalarType::Float {
            format: AbiFloatFormat::Ieee(32),
        }) => "float",
        AbiType::Scalar(ScalarType::Float {
            format: AbiFloatFormat::Ieee(64),
        }) => "double",
        _ => {
            return Err(format!(
                "XXI byte-view bridge does not support scalar representation {ty:?}"
            ))
        }
    })
}

pub fn render_bridges(plans: &[BoundaryPlan]) -> Result<String, String> {
    let mut output = String::new();
    let mut emitted = BTreeSet::new();
    for plan in plans {
        let Some(bridge) = bridge_symbol(plan) else {
            continue;
        };
        if !emitted.insert(bridge.clone()) {
            continue;
        }
        if plan.signature.variadic
            || !matches!(
                plan.signature.convention,
                severian_abi::CallingConvention::C | severian_abi::CallingConvention::System
            )
        {
            return Err("XXI byte-view bridges require a non-variadic target C ABI".into());
        }
        let symbol = plan.symbol.name.as_str();
        if symbol.is_empty()
            || symbol.as_bytes()[0].is_ascii_digit()
            || !symbol
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(
                "XXI byte-view bridges require a C identifier for the foreign symbol".into(),
            );
        }
        let result = scalar(&plan.result_type)?;
        let mut incoming = Vec::new();
        let mut outgoing = Vec::new();
        let mut arguments = Vec::new();
        let mut acquire = String::new();
        let mut release = String::new();
        for (index, parameter) in plan.parameters.iter().enumerate() {
            if matches!(parameter.contract.ty, ForeignTypeRef::Sequence { .. }) {
                let mutable = parameter.mode == ParameterMode::InOut;
                incoming.push(format!("sev_xxi_list a{index}"));
                outgoing.push(format!("sev_xxi_bytes{}", if mutable { " *" } else { "" }));
                acquire.push_str(&format!(
                    "    sev_xxi_bytes_loan loan{index} = sev_xxi_bytes_acquire(a{index}, {});\n",
                    u8::from(mutable)
                ));
                arguments.push(format!(
                    "{}loan{index}.view",
                    if mutable { "&" } else { "" }
                ));
                release = format!("    sev_xxi_bytes_release(&loan{index});\n{release}");
            } else {
                let ty = scalar(&parameter.abi_type)?;
                if ty == "void" {
                    return Err("void cannot be a foreign parameter".into());
                }
                incoming.push(format!("{ty} a{index}"));
                outgoing.push(ty.into());
                arguments.push(format!("a{index}"));
            }
        }
        let mut aliases = String::new();
        for (index, parameter) in plan.parameters.iter().enumerate() {
            if !matches!(parameter.contract.ty, ForeignTypeRef::Sequence { .. }) {
                continue;
            }
            for (other, previous) in plan.parameters[..index].iter().enumerate() {
                if matches!(previous.contract.ty, ForeignTypeRef::Sequence { .. })
                    && (parameter.mode == ParameterMode::InOut
                        || previous.mode == ParameterMode::InOut)
                {
                    aliases.push_str(&format!("    if (a{index}.storage == a{other}.storage) sev_xxi_contract_failure(\"overlapping exclusive foreign loans\");\n"));
                }
            }
        }
        acquire = format!("{aliases}{acquire}");
        output.push_str(&format!(
            "extern {result} {symbol}({});\n{result} {bridge}({}) {{\n{acquire}",
            outgoing.join(", "),
            incoming.join(", ")
        ));
        output.push_str(&format!(
            "    {}{symbol}({});\n{release}",
            if result == "void" {
                String::new()
            } else {
                format!("{result} result = ")
            },
            arguments.join(", ")
        ));
        if result != "void" {
            output.push_str("    return result;\n");
        }
        output.push_str("}\n");
    }
    Ok(output)
}
