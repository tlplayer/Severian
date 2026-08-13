use super::*;

pub(super) fn task_type_suffix(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "i64",
        ValueType::Float => "f64",
        ValueType::Bool => "bool",
        ValueType::Unit => "unit",
        ValueType::String
        | ValueType::List
        | ValueType::Tuple
        | ValueType::Map
        | ValueType::Set
        | ValueType::Tensor(_)
        | ValueType::TensorAny
        | ValueType::Channel
        | ValueType::Function
        | ValueType::Result
        | ValueType::Option
        | ValueType::Any => "ptr",
    }
}

pub(super) fn source_function_symbol(name: &str) -> String {
    if name == "main" {
        "main".into()
    } else {
        format!("__sev_fn_{}", mangle_symbol_component(name))
    }
}

pub(super) fn mangle_symbol_component(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn class_function_symbol(class: &str, method: &str) -> String {
    format!("__sev_method_{class}_{method}")
}

#[derive(Debug, Clone)]
pub(super) struct DynamicMethodDispatch {
    pub(super) symbol: String,
    pub(super) method: String,
    pub(super) params: Vec<ValueType>,
    pub(super) returns: ValueType,
    pub(super) classes: Vec<String>,
}

pub(super) fn dynamic_method_dispatches(program: &Program) -> Vec<DynamicMethodDispatch> {
    let mut groups = HashMap::<(String, Vec<ValueType>, ValueType), Vec<String>>::new();
    for class in &program.classes {
        for method in &class.methods {
            let params = method
                .params
                .iter()
                .map(|parameter| parameter.ty)
                .collect::<Vec<_>>();
            groups
                .entry((method.name.clone(), params, method.return_type))
                .or_default()
                .push(class.name.clone());
        }
    }
    let mut dispatches = groups
        .into_iter()
        .map(|((method, params, returns), mut classes)| {
            classes.sort();
            DynamicMethodDispatch {
                symbol: dynamic_method_dispatch_symbol(&method, &params, returns),
                method,
                params,
                returns,
                classes,
            }
        })
        .collect::<Vec<_>>();
    dispatches.sort_by(|left, right| left.symbol.cmp(&right.symbol));
    dispatches
}

pub(super) fn dynamic_method_dispatch_symbol(
    method: &str,
    params: &[ValueType],
    returns: ValueType,
) -> String {
    let mut signature = params
        .iter()
        .map(|ty| dynamic_type_suffix(*ty))
        .collect::<Vec<_>>()
        .join("_");
    if signature.is_empty() {
        signature.push_str("none");
    }
    format!(
        "__sev_dispatch_{}_{}_{}",
        mangle_symbol_component(method),
        signature,
        dynamic_type_suffix(returns)
    )
}

pub(super) fn dynamic_type_suffix(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "i64",
        ValueType::Float => "f64",
        ValueType::Bool => "bool",
        ValueType::Unit => "unit",
        ValueType::String => "string",
        ValueType::List => "list",
        ValueType::Tuple => "tuple",
        ValueType::Map => "map",
        ValueType::Set => "set",
        ValueType::Tensor(_) => "tensor",
        ValueType::TensorAny => "tensor_any",
        ValueType::Channel => "channel",
        ValueType::Function => "function",
        ValueType::Result => "result",
        ValueType::Option => "option",
        ValueType::Any => "any",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NativeCallSignature {
    pub(super) id: FunctionId,
    pub(super) symbol: String,
    pub(super) params: Vec<ValueType>,
    pub(super) returns: ValueType,
}

pub(super) fn native_call_signatures(program: &Program) -> HashMap<String, NativeCallSignature> {
    let mut signatures = HashMap::new();
    let mut scanned = program.clone();
    scanned.visit_expressions_mut(&mut |expression| {
        let Expression::Typed {
            ty,
            expression: kind,
            ..
        } = expression
        else {
            return;
        };
        let Expression::Call { target, args } = kind.as_ref() else {
            return;
        };
        let Some(symbol) = &target.native_symbol else {
            return;
        };
        signatures
            .entry(symbol.clone())
            .or_insert_with(|| NativeCallSignature {
                id: target.id,
                symbol: symbol.clone(),
                params: args
                    .iter()
                    .map(|argument| argument.ty().unwrap_or(ValueType::Any))
                    .collect(),
                returns: *ty,
            });
    });
    signatures
}

pub(super) fn is_predeclared_native_symbol(symbol: &str) -> bool {
    matches!(symbol, "__sev_regex_matches")
}

pub(super) fn c_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "int64_t",
        ValueType::Float => "double",
        ValueType::Bool => "bool",
        ValueType::Unit => "void",
        ValueType::String
        | ValueType::List
        | ValueType::Tuple
        | ValueType::Map
        | ValueType::Set
        | ValueType::Tensor(_)
        | ValueType::TensorAny
        | ValueType::Channel
        | ValueType::Function
        | ValueType::Result
        | ValueType::Option
        | ValueType::Any => "void *",
    }
}

pub(super) fn static_tensor_elements(tensor: severian_hir::TensorType) -> Option<i64> {
    let rank = usize::from(tensor.rank?);
    tensor.dimensions[..rank]
        .iter()
        .try_fold(1_i64, |total, dimension| {
            let severian_hir::TensorDimension::Static(value) = dimension else {
                return None;
            };
            total.checked_mul(i64::try_from(*value).ok()?)
        })
}

pub(super) fn tensor_element_bytes(element: severian_hir::TensorElementType) -> i64 {
    match element {
        severian_hir::TensorElementType::BF16 => 2,
        severian_hir::TensorElementType::F32 | severian_hir::TensorElementType::I32 => 4,
        severian_hir::TensorElementType::F64 | severian_hir::TensorElementType::I64 => 8,
    }
}

pub(super) fn mlir_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "i64",
        ValueType::Float => "f64",
        ValueType::Bool => "i1",
        ValueType::String => "!llvm.ptr",
        ValueType::Unit => "!llvm.void",
        ValueType::List
        | ValueType::Tuple
        | ValueType::Map
        | ValueType::Set
        | ValueType::Tensor(_)
        | ValueType::TensorAny
        | ValueType::Channel
        | ValueType::Function
        | ValueType::Any
        | ValueType::Result
        | ValueType::Option => "!llvm.ptr",
    }
}

pub(super) fn assignment_binary(op: AssignmentOp) -> BinaryOp {
    match op {
        AssignmentOp::Assign => unreachable!(),
        AssignmentOp::Add => BinaryOp::Add,
        AssignmentOp::Sub => BinaryOp::Sub,
        AssignmentOp::Mul => BinaryOp::Mul,
        AssignmentOp::Div => BinaryOp::Div,
        AssignmentOp::Mod => BinaryOp::Mod,
    }
}

pub(super) fn escape_string(value: &str) -> String {
    let mut escaped = String::new();
    for byte in value.as_bytes() {
        match byte {
            b' '..=b'!' | b'#'..=b'[' | b']'..=b'~' => escaped.push(*byte as char),
            _ => write!(escaped, "\\{byte:02X}").unwrap(),
        }
    }
    escaped
}

pub(super) fn native_format_template(template: &str, arg_types: &[ValueType]) -> String {
    let mut output = String::new();
    let mut remainder = template;
    let mut index = 0;

    while let Some(open) = remainder.find('{') {
        output.push_str(&remainder[..open].replace('%', "%%"));
        let field = &remainder[open + 1..];
        let close = field.find('}').expect("formatted fields are validated");
        output.push_str(match arg_types[index] {
            ValueType::Int => "%ld",
            ValueType::Float => "%.15g",
            ValueType::String => "%s",
            ValueType::Bool => "%d",
            _ => "%p",
        });
        index += 1;
        remainder = &field[close + 1..];
    }
    output.push_str(&remainder.replace('%', "%%"));
    output
}
