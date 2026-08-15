use severian_abi::AbiType;
use severian_hir::Program;
use std::fmt::Write;

pub(super) fn append_c_v1_shims(source: &mut String, program: &Program) {
    if program.metadata.external_functions.is_empty() {
        return;
    }
    source.push_str(concat!(
        "static void sev_ffi_replace_field(void *raw, const char *name, void *item) {\n",
        "  sev_object *object = raw;\n",
        "  if (!object || object->magic != SEV_OBJECT_MAGIC) abort();\n",
        "  for (int64_t index = 0; index < object->size; ++index) {\n",
        "    if (strcmp(object->names[index], name) == 0) { object->values[index] = item; return; }\n",
        "  }\n",
        "  abort();\n",
        "}\n",
        "static char *sev_ffi_copy_string(sev_string_view_v1 view) {\n",
        "  if (!view.data && view.length) abort();\n",
        "  char *copy = sev_allocate(view.length + 1);\n",
        "  if (view.length) memcpy(copy, view.data, view.length);\n",
        "  copy[view.length] = '\\0';\n",
        "  return copy;\n",
        "}\n\n",
    ));
    for function in program.metadata.external_functions.values() {
        append_provider_declaration(source, function);
        append_shim_signature(source, function);
        append_parameter_conversions(source, function);
        append_provider_call(source, function);
        append_out_conversions(source, function);
        append_result_conversion(source, function.result.ty);
        source.push_str("}\n\n");
    }
}

fn append_provider_declaration(source: &mut String, function: &severian_abi::ExternalFunction) {
    write!(
        source,
        "{} {}(",
        function.result.ty.c_name(),
        function.symbol
    )
    .unwrap();
    append_parameters(source, function, false);
    source.push_str(");\n");
}

fn append_shim_signature(source: &mut String, function: &severian_abi::ExternalFunction) {
    write!(
        source,
        "{} {}(",
        shim_c_type(function.result.ty),
        function.shim_symbol
    )
    .unwrap();
    append_parameters(source, function, true);
    source.push_str(") {\n");
}

fn append_parameters(source: &mut String, function: &severian_abi::ExternalFunction, shim: bool) {
    if function.parameters.is_empty() {
        source.push_str("void");
    }
    for (index, parameter) in function.parameters.iter().enumerate() {
        if index > 0 {
            source.push_str(", ");
        }
        let ty = if shim {
            shim_c_type(parameter.ty)
        } else {
            parameter.ty.c_name()
        };
        let name = if shim {
            format!("arg_{index}")
        } else {
            parameter.name.clone()
        };
        write!(source, "{ty} {name}").unwrap();
    }
}

fn append_parameter_conversions(source: &mut String, function: &severian_abi::ExternalFunction) {
    for (index, parameter) in function.parameters.iter().enumerate() {
        match parameter.ty {
            AbiType::StringView => writeln!(source, "  sev_string_view_v1 abi_{index} = {{ .data = (const uint8_t *)arg_{index}, .length = arg_{index} ? strlen(arg_{index}) : 0 }};").unwrap(),
            AbiType::BytesView => {
                writeln!(source, "  void *boxed_bytes_{index} = __sev_object_get(arg_{index}, \"data\");").unwrap();
                writeln!(source, "  sev_collection *collection_{index} = __sev_unbox_ptr(boxed_bytes_{index});").unwrap();
                writeln!(source, "  size_t length_{index} = collection_{index} && collection_{index}->size > 0 ? (size_t)collection_{index}->size : 0;").unwrap();
                writeln!(source, "  uint8_t *data_{index} = sev_allocate(length_{index} ? length_{index} : 1);").unwrap();
                writeln!(source, "  for (size_t item = 0; item < length_{index}; ++item) {{ int64_t byte = __sev_unbox_i64(collection_{index}->items[item]); if (byte < 0 || byte > 255) abort(); data_{index}[item] = (uint8_t)byte; }}").unwrap();
                writeln!(source, "  sev_bytes_view_v1 abi_{index} = {{ .data = data_{index}, .length = length_{index} }};").unwrap();
            }
            AbiType::Handle => {
                writeln!(source, "  void *boxed_handle_{index} = __sev_object_get(arg_{index}, \"opaque\");").unwrap();
                writeln!(source, "  sev_handle_v1 abi_{index} = {{ .value = (void *)(intptr_t)__sev_unbox_i64(boxed_handle_{index}) }};").unwrap();
            }
            AbiType::OutHandle => writeln!(source, "  sev_handle_v1 abi_{index} = {{ .value = NULL }};").unwrap(),
            AbiType::OutError => writeln!(source, "  sev_error_v1 abi_{index} = {{ .code = 0, .message = {{ .data = NULL, .length = 0 }} }};").unwrap(),
            AbiType::OutUsize => writeln!(source, "  size_t abi_{index} = 0;").unwrap(),
            _ => {}
        }
    }
}

fn append_provider_call(source: &mut String, function: &severian_abi::ExternalFunction) {
    let arguments = function
        .parameters
        .iter()
        .enumerate()
        .map(|(index, parameter)| match parameter.ty {
            AbiType::OutHandle | AbiType::OutError | AbiType::OutUsize => format!("&abi_{index}"),
            AbiType::StringView | AbiType::BytesView | AbiType::Handle => format!("abi_{index}"),
            _ => format!("({})arg_{index}", parameter.ty.c_name()),
        })
        .collect::<Vec<_>>()
        .join(", ");
    if function.result.ty == AbiType::Unit {
        writeln!(source, "  {}({arguments});", function.symbol).unwrap();
    } else {
        writeln!(
            source,
            "  {} result = {}({arguments});",
            function.result.ty.c_name(),
            function.symbol
        )
        .unwrap();
    }
}

fn append_out_conversions(source: &mut String, function: &severian_abi::ExternalFunction) {
    for (index, parameter) in function.parameters.iter().enumerate() {
        match parameter.ty {
            AbiType::BytesView => writeln!(source, "  free(data_{index});").unwrap(),
            AbiType::OutHandle => writeln!(source, "  sev_ffi_replace_field(arg_{index}, \"value\", __sev_box_i64((int64_t)(intptr_t)abi_{index}.value));").unwrap(),
            AbiType::OutError => {
                writeln!(source, "  sev_ffi_replace_field(arg_{index}, \"code\", __sev_box_i64(abi_{index}.code));").unwrap();
                writeln!(source, "  sev_ffi_replace_field(arg_{index}, \"message\", __sev_box_string(sev_ffi_copy_string(abi_{index}.message)));").unwrap();
            }
            AbiType::OutUsize => writeln!(source, "  sev_ffi_replace_field(arg_{index}, \"value\", __sev_box_i64((int64_t)abi_{index}));").unwrap(),
            _ => {}
        }
    }
}

fn append_result_conversion(source: &mut String, ty: AbiType) {
    match ty {
        AbiType::Unit => {}
        AbiType::StringView => source.push_str("  return sev_ffi_copy_string(result);\n"),
        AbiType::Handle => source.push_str("  return result.value;\n"),
        AbiType::Bool => source.push_str("  return result;\n"),
        AbiType::F32 | AbiType::F64 => source.push_str("  return (double)result;\n"),
        _ => source.push_str("  return (int64_t)result;\n"),
    }
}

fn shim_c_type(ty: AbiType) -> &'static str {
    match ty {
        AbiType::Unit => "void",
        AbiType::Bool => "bool",
        AbiType::F32 | AbiType::F64 => "double",
        AbiType::I8
        | AbiType::I16
        | AbiType::I32
        | AbiType::I64
        | AbiType::U8
        | AbiType::U16
        | AbiType::U32
        | AbiType::U64
        | AbiType::Usize
        | AbiType::Isize => "int64_t",
        AbiType::StringView
        | AbiType::BytesView
        | AbiType::Handle
        | AbiType::OutHandle
        | AbiType::OutError
        | AbiType::OutUsize => "void *",
    }
}
