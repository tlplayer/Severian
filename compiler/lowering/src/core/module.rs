use super::*;

pub fn lower(mir: &severian_mir::Program) -> Module {
    lower_hir(mir.lowering_hir())
}

pub(super) fn lower_hir(program: &Program) -> Module {
    let metadata = program.metadata.clone();
    let mut resolved_program = program.clone();
    resolve_contract_locations(&mut resolved_program, &metadata);
    resolve_external_symbols(&mut resolved_program);
    let program = &resolved_program;
    let mut strings = Vec::new();
    for class in &program.classes {
        strings.push(class.name.clone());
        strings.extend(class.fields.iter().cloned());
        for constraint in &class.field_constraints {
            collect_expression_strings(constraint, &mut strings);
        }
        for function in class.methods.iter().chain(&class.constructors) {
            collect_strings(&function.instructions, &mut strings);
        }
    }
    for global in &program.globals {
        collect_expression_strings(&global.value, &mut strings);
    }
    for function in &program.functions {
        collect_strings(&function.instructions, &mut strings);
    }
    for registry in program.metadata.trait_registries.values() {
        for implementation in &registry.implementations {
            strings.push(implementation.name.clone());
            for value in implementation.properties.values() {
                collect_trait_property_strings(value, &mut strings);
            }
        }
    }
    for file in program.metadata.sources.files() {
        let path = file.path.to_string_lossy().into_owned();
        if !strings.contains(&path) {
            strings.push(path);
        }
    }

    let mut output = String::from("module {\n");
    for (index, value) in strings.iter().enumerate() {
        writeln!(
            output,
            "  llvm.mlir.global internal constant @__sev_str_{index}(\"{}\\00\")",
            escape_string(value)
        )
        .unwrap();
    }
    output.push_str(concat!(
        "  llvm.mlir.global internal constant @__sev_fmt_int(\"%ld\\0A\\00\")\n",
        "  llvm.mlir.global internal constant @__sev_fmt_float(\"%.15g\\0A\\00\")\n",
        "  llvm.mlir.global internal constant @__sev_bool_true(\"true\\00\")\n",
        "  llvm.mlir.global internal constant @__sev_bool_false(\"false\\00\")\n",
        "  llvm.func @puts(!llvm.ptr) -> i32\n",
        "  llvm.func @printf(!llvm.ptr, ...) -> i32\n\n",
        "  llvm.func @snprintf(!llvm.ptr, i64, !llvm.ptr, ...) -> i32\n",
        "  llvm.func @malloc(i64) -> !llvm.ptr\n",
        "  llvm.func @abort()\n",
        "  llvm.func @__sev_runtime_set_site(!llvm.ptr, i64, i64, i64)\n",
        "  llvm.func @__sev_runtime_fail_assertion()\n",
        "  llvm.func @__sev_runtime_fail_division_zero()\n",
        "  llvm.func @strtod(!llvm.ptr, !llvm.ptr) -> f64\n\n",
        "  llvm.func @__sev_strlen(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_string_length(!llvm.ptr) -> i64\n\n",
        "  llvm.func @__sev_box_i64(i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_box_f64(f64) -> !llvm.ptr\n",
        "  llvm.func @__sev_box_bool(i1) -> !llvm.ptr\n",
        "  llvm.func @__sev_box_string(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_box_collection(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_unbox_i64(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_unbox_f64(!llvm.ptr) -> f64\n",
        "  llvm.func @__sev_unbox_bool(!llvm.ptr) -> i1\n",
        "  llvm.func @__sev_unbox_string(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_unbox_ptr(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_closure_new(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_closure_function(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_closure_environment(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_add(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_sub(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_mul(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_div(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_int(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_value_float(!llvm.ptr) -> f64\n",
        "  llvm.func @__sev_value_string(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_concat(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_concat(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_repeat(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_equal(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_string_char_at(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_slice(!llvm.ptr, i64, i64, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_characters(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_words(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_split(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_frequencies(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_equal(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_value_less(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_value_size(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_value_bytes(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_value_capacity(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_tensor_size(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_tensor_bytes(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_tensor_capacity(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_value_index(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_get(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_set(!llvm.ptr, !llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_value_slice(!llvm.ptr, i64, i64, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_new(i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_clone(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_push(!llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_collection_get(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_slice(!llvm.ptr, i64, i64, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_set(!llvm.ptr, i64, !llvm.ptr)\n",
        "  llvm.func @__sev_collection_size(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_collection_equal(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_collection_reversed(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_pop(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_pop_at(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_appendleft(!llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_collection_extend(!llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_collection_insert(!llvm.ptr, i64, !llvm.ptr)\n",
        "  llvm.func @__sev_collection_remove(!llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_collection_heap_push(!llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_collection_heap_pop(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_last(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_sorted(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_sorted_reverse(!llvm.ptr, i1) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_sorted_keys(!llvm.ptr, !llvm.ptr, i1) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_join(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_sum(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_minimum(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_maximum(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_to_set(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_enumerate(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_zip(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_any(!llvm.ptr) -> i1\n",
        "  llvm.func @__sev_collection_all(!llvm.ptr) -> i1\n",
        "  llvm.func @__sev_range(i64, i64, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_abs(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_min(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_max(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_divmod(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_set_difference(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_set_combine(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_set_to_list(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_set_contains(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_set_add(!llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_map_new() -> !llvm.ptr\n",
        "  llvm.func @__sev_map_insert(!llvm.ptr, !llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_map_contains(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_map_get(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_size(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_map_bytes(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_map_capacity(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_map_key_at(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_value_at(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_keys(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_values(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_get_default(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_pop_required(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_pop(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_set_default(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_frequencies(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_strip(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_lstrip(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_rstrip(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_lower(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_upper(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_capitalize(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_title(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_swapcase(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_collapse_space(!llvm.ptr, i1) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_split_lines(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_split_limit(!llvm.ptr, !llvm.ptr, i64, i1) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_replace(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_replace_limit(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_starts_with(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_string_ends_with(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_string_contains(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_string_find(!llvm.ptr, !llvm.ptr) -> i64\n",
        "  llvm.func @__sev_string_rfind(!llvm.ptr, !llvm.ptr) -> i64\n",
        "  llvm.func @__sev_string_count(!llvm.ptr, !llvm.ptr) -> i64\n",
        "  llvm.func @__sev_string_predicate(!llvm.ptr, i64) -> i1\n",
        "  llvm.func @__sev_string_remove_affix(!llvm.ptr, !llvm.ptr, i1, i1) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_translate(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_replace_many(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_remove(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_remove_matches(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_repeat(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_pad(!llvm.ptr, i64, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_take(!llvm.ptr, i64, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_segment(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_between(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_string_partition(!llvm.ptr, !llvm.ptr, i1) -> !llvm.ptr\n",
        "  llvm.func @__sev_print_value(!llvm.ptr)\n",
        "  llvm.func @__sev_print_collection(!llvm.ptr)\n\n",
        "  llvm.func @__sev_print_value_inline(!llvm.ptr)\n",
        "  llvm.func @__sev_print_space()\n",
        "  llvm.func @__sev_print_newline()\n",
        "  llvm.func @__sev_object_new(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_object_declare(!llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_object_set(!llvm.ptr, !llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_object_get(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_map_get(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_dynamic_object_get(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_object_is(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_dispatch_draw(!llvm.ptr)\n\n",
        "  llvm.func @__sev_variant_new(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_variant_is(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_variant_field(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_print_variant(!llvm.ptr)\n\n",
        "  llvm.func @__sev_builtin_read(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_builtin_http_get(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_builtin_int_parse(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_builtin_float_parse(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_builtin_file_write(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n\n",
        "  llvm.func @__sev_task_await_unit(!llvm.ptr)\n\n",
        "  llvm.func @llvm.sqrt.f64(f64) -> f64\n\n",
    ));
    for dispatch in dynamic_method_dispatches(program) {
        write!(output, "  llvm.func @{}(!llvm.ptr", dispatch.symbol).unwrap();
        for parameter in &dispatch.params {
            write!(output, ", {}", mlir_type(*parameter)).unwrap();
        }
        output.push(')');
        if dispatch.returns != ValueType::Unit {
            write!(output, " -> {}", mlir_type(dispatch.returns)).unwrap();
        }
        output.push('\n');
    }

    let mut fusion_runtime_symbols = HashSet::new();
    let mut scanned_program = program.clone();
    scanned_program.visit_expressions_mut(&mut |expression| {
        if let Expression::FusedPipeline { runtime_symbol, .. } = expression {
            fusion_runtime_symbols.insert(runtime_symbol.clone());
        }
    });
    for runtime_symbol in fusion_runtime_symbols {
        writeln!(
            output,
            "  llvm.func @{runtime_symbol}(!llvm.ptr, i64, i64) -> !llvm.ptr"
        )
        .unwrap();
    }

    let native_call_signatures = native_call_signatures(program);
    let mut native_symbols = program
        .functions
        .iter()
        .filter_map(|function| {
            function
                .native_symbol
                .as_ref()
                .map(|symbol| (function.id, symbol.clone()))
        })
        .collect::<HashMap<_, _>>();
    for signature in native_call_signatures.values() {
        native_symbols.insert(signature.id, signature.symbol.clone());
    }
    let mut declared_native_symbols = HashSet::new();
    for function in program
        .functions
        .iter()
        .filter(|function| function.native_symbol.is_some())
    {
        let symbol = function.native_symbol.as_ref().unwrap();
        if is_predeclared_native_symbol(symbol) || !declared_native_symbols.insert(symbol.as_str())
        {
            continue;
        }
        write!(output, "  llvm.func @{symbol}(").unwrap();
        for (index, parameter) in function.params.iter().enumerate() {
            if index > 0 {
                output.push_str(", ");
            }
            output.push_str(mlir_type(parameter.ty));
        }
        output.push(')');
        if function.return_type != ValueType::Unit {
            write!(output, " -> {}", mlir_type(function.return_type)).unwrap();
        }
        output.push('\n');
    }
    for function in program.functions.iter().filter(|function| {
        function.native_symbol.is_none()
            && function
                .decorators
                .iter()
                .any(|decorator| decorator.package == "tensor")
    }) {
        write!(
            output,
            "  llvm.func @{}(",
            source_function_symbol(&function.name)
        )
        .unwrap();
        for (index, parameter) in function.params.iter().enumerate() {
            if index > 0 {
                output.push_str(", ");
            }
            output.push_str(mlir_type(parameter.ty));
        }
        write!(output, ") -> {}\n", mlir_type(function.return_type)).unwrap();
    }
    let locally_declared_symbols = program
        .functions
        .iter()
        .filter_map(|function| function.native_symbol.as_deref())
        .collect::<HashSet<_>>();
    for signature in native_call_signatures.values() {
        if locally_declared_symbols.contains(signature.symbol.as_str())
            || is_predeclared_native_symbol(&signature.symbol)
        {
            continue;
        }
        write!(output, "  llvm.func @{}(", signature.symbol).unwrap();
        for (index, parameter) in signature.params.iter().enumerate() {
            if index > 0 {
                output.push_str(", ");
            }
            output.push_str(mlir_type(*parameter));
        }
        output.push(')');
        if signature.returns != ValueType::Unit {
            write!(output, " -> {}", mlir_type(signature.returns)).unwrap();
        }
        output.push('\n');
    }
    let has_tensor_relu = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_relu");
    let has_tensor_add = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_add");
    let has_tensor_matmul = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_matmul");
    let has_tensor_transpose = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_transpose");
    let has_tensor_scale = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_scale");
    let has_tensor_softmax_rows = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_softmax_rows");
    let has_tensor_layer_norm = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_layer_norm");
    let has_tensor_relu_backward = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_relu_backward");
    let has_tensor_softmax_backward = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_softmax_backward");
    let has_tensor_layer_norm_backward = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_layer_norm_backward");
    let has_tensor_autodiff = native_symbols
        .values()
        .any(|symbol| symbol == "__sev_tensor_backward_mse");
    let has_model_graph = native_symbols
        .values()
        .any(|symbol| symbol.starts_with("__sev_model_graph_"));
    output.push_str(&tensor::mlir_kernels(
        has_tensor_relu || has_model_graph,
        has_tensor_add || has_model_graph,
        has_tensor_matmul || has_model_graph,
        has_tensor_transpose || has_model_graph,
        has_tensor_scale || has_model_graph,
        has_tensor_softmax_rows || has_model_graph,
        has_tensor_layer_norm || has_model_graph,
        has_tensor_relu_backward || has_tensor_autodiff,
        has_tensor_softmax_backward || has_tensor_autodiff,
        has_tensor_layer_norm_backward || has_tensor_autodiff,
    ));

    let task_specs = task_specs(program);
    let uses_channels = uses_channels(program);
    let mut await_types = HashSet::new();
    for task in &task_specs {
        write!(output, "  llvm.func @__sev_task_spawn_{}(", task.symbol).unwrap();
        for (index, ty) in task.params.iter().enumerate() {
            if index > 0 {
                output.push_str(", ");
            }
            output.push_str(mlir_type(*ty));
        }
        output.push_str(") -> !llvm.ptr\n");
        await_types.insert(task.return_type);
    }
    for class in &program.classes {
        for method in &class.methods {
            write!(
                output,
                "  llvm.func @__sev_task_spawn_{}_{}(!llvm.ptr",
                class.name, method.name
            )
            .unwrap();
            for parameter in &method.params {
                write!(output, ", {}", mlir_type(parameter.ty)).unwrap();
            }
            output.push_str(") -> !llvm.ptr\n");
            await_types.insert(method.return_type);
        }
    }
    let mut declared_await_suffixes = HashSet::new();
    for ty in await_types {
        if ty != ValueType::Unit {
            let suffix = task_type_suffix(ty);
            if !declared_await_suffixes.insert(suffix) {
                continue;
            }
            writeln!(
                output,
                "  llvm.func @__sev_task_await_{}(!llvm.ptr) -> {}",
                suffix,
                mlir_type(ty)
            )
            .unwrap();
        }
    }
    if uses_channels {
        output.push_str(concat!(
            "  llvm.func @__sev_channel_create(i64) -> !llvm.ptr\n",
            "  llvm.func @__sev_channel_send_ptr_async(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
            "  llvm.func @__sev_channel_receive_ptr(!llvm.ptr) -> !llvm.ptr\n",
        ));
    }
    if !task_specs.is_empty() || uses_channels {
        output.push('\n');
    }

    let mut function_returns = program
        .functions
        .iter()
        .map(|function| (function.id, function.return_type))
        .collect::<HashMap<_, _>>();
    let mut function_params = program
        .functions
        .iter()
        .map(|function| {
            (
                function.id,
                function
                    .params
                    .iter()
                    .map(|parameter| parameter.ty)
                    .collect(),
            )
        })
        .collect::<HashMap<_, Vec<_>>>();
    for class in &program.classes {
        for method in &class.methods {
            function_returns.insert(method.id, method.return_type);
            function_params.insert(
                method.id,
                method.params.iter().map(|parameter| parameter.ty).collect(),
            );
        }
    }
    let function_return_classes = program
        .functions
        .iter()
        .filter_map(|function| {
            let signature = program.metadata.functions.get(&function.id)?;
            let definition = match program.metadata.types.get(signature.returns)? {
                TypeKind::Named { definition, .. } => definition,
                _ => return None,
            };
            Some((function.id, *definition))
        })
        .collect::<HashMap<_, _>>();
    let method_return_classes = program
        .classes
        .iter()
        .flat_map(|class| {
            class.methods.iter().filter_map(|method| {
                let definition = program
                    .metadata
                    .functions
                    .get(&method.id)
                    .and_then(|signature| program.metadata.types.get(signature.returns))
                    .and_then(|kind| match kind {
                        TypeKind::Named { definition, .. } => Some(*definition),
                        _ => None,
                    })?;
                Some(((class.id, method.name.clone()), definition))
            })
        })
        .collect::<HashMap<_, _>>();
    let closure_definitions = Rc::new(RefCell::new(String::new()));
    let next_closure = Rc::new(Cell::new(0));
    let function_closures = Rc::new(RefCell::new(HashMap::new()));
    let environment = LoweringEnvironment {
        globals: &program.globals,
        classes: &program.classes,
        strings: &strings,
        function_returns: &function_returns,
        function_params: &function_params,
        function_return_classes: &function_return_classes,
        method_return_classes: &method_return_classes,
        closure_definitions: &closure_definitions,
        next_closure: &next_closure,
        function_closures: &function_closures,
        native_symbols: &native_symbols,
        sources: &program.metadata.sources,
        trait_registries: &program.metadata.trait_registries,
    };
    for class in &program.classes {
        for constructor in &class.constructors {
            lower_class_function(
                class,
                constructor,
                &class_function_symbol(&class.name, &format!("ctor_{}", constructor.params.len())),
                &environment,
                &mut output,
            );
        }
        for method in &class.methods {
            lower_class_function(
                class,
                method,
                &class_function_symbol(&class.name, &method.name),
                &environment,
                &mut output,
            );
        }
    }
    for function in &program.functions {
        if function.native_symbol.is_some()
            || function
                .decorators
                .iter()
                .any(|decorator| decorator.package == "tensor")
        {
            continue;
        }
        lower_function(function, &environment, &mut output);
    }
    output.push_str(&closure_definitions.borrow());
    output.push_str("}\n");
    Module::new(output)
}

fn collect_trait_property_strings(
    value: &severian_hir::TraitPropertyValue,
    strings: &mut Vec<String>,
) {
    match value {
        severian_hir::TraitPropertyValue::String(value)
        | severian_hir::TraitPropertyValue::Symbol(value) => strings.push(value.clone()),
        severian_hir::TraitPropertyValue::Constructor { arguments, .. }
        | severian_hir::TraitPropertyValue::List(arguments)
        | severian_hir::TraitPropertyValue::Set(arguments)
        | severian_hir::TraitPropertyValue::Tuple(arguments) => {
            for argument in arguments {
                collect_trait_property_strings(argument, strings);
            }
        }
        severian_hir::TraitPropertyValue::Map(entries) => {
            for (key, value) in entries {
                collect_trait_property_strings(key, strings);
                collect_trait_property_strings(value, strings);
            }
        }
        _ => {}
    }
}

fn resolve_external_symbols(program: &mut Program) {
    let shims = program
        .metadata
        .external_functions
        .iter()
        .map(|(symbol, function)| (symbol.clone(), function.shim_symbol.clone()))
        .collect::<HashMap<_, _>>();
    let replace = |symbol: &mut Option<String>| {
        if let Some(shim) = symbol
            .as_ref()
            .and_then(|symbol| shims.get(symbol))
            .cloned()
        {
            *symbol = Some(shim);
        }
    };
    for function in &mut program.functions {
        replace(&mut function.native_symbol);
    }
    for class in &mut program.classes {
        for function in class.methods.iter_mut().chain(&mut class.constructors) {
            replace(&mut function.native_symbol);
        }
    }
    program.visit_expressions_mut(&mut |expression| match expression {
        Expression::Call { target, .. } | Expression::Function(target) => {
            replace(&mut target.native_symbol)
        }
        Expression::ChaosRule { function, .. } => replace(&mut function.native_symbol),
        _ => {}
    });
}

pub(super) fn resolve_contract_locations(
    program: &mut Program,
    metadata: &severian_hir::ProgramMetadata,
) {
    fn visit(instructions: &mut [Instruction], metadata: &severian_hir::ProgramMetadata) {
        for instruction in instructions {
            match instruction {
                Instruction::If {
                    condition,
                    then_instructions,
                    else_instructions,
                } => {
                    if let Some(location) = contract_source_location(condition, metadata) {
                        if let Some(Expression::Typed { expression, .. }) = else_instructions
                            .first_mut()
                            .and_then(|instruction| match instruction {
                                Instruction::Evaluate(expression) => Some(expression),
                                _ => None,
                            })
                        {
                            if let Expression::Call { target, args } = expression.as_mut() {
                                if target.name == "__sev_contract_fail" {
                                    if let Some(Expression::Typed { expression, .. }) =
                                        args.get_mut(1)
                                    {
                                        **expression = Expression::String(location);
                                    }
                                }
                            }
                        }
                    }
                    visit(then_instructions, metadata);
                    visit(else_instructions, metadata);
                }
                Instruction::While { instructions, .. }
                | Instruction::For { instructions, .. }
                | Instruction::With { instructions, .. } => visit(instructions, metadata),
                Instruction::Switch { arms, .. } | Instruction::ChannelSwitch { arms, .. } => {
                    for arm in arms {
                        visit(&mut arm.instructions, metadata);
                    }
                }
                _ => {}
            }
        }
    }
    for function in &mut program.functions {
        visit(&mut function.instructions, metadata);
    }
    for class in &mut program.classes {
        for function in class.methods.iter_mut().chain(&mut class.constructors) {
            visit(&mut function.instructions, metadata);
        }
    }
}

pub(super) fn contract_source_location(
    condition: &Expression,
    metadata: &severian_hir::ProgramMetadata,
) -> Option<String> {
    let span = metadata.sources.expression_span(condition.hir_id()?)?;
    let file = metadata.sources.file(span.file)?;
    let before = file.source.get(..span.range.start)?;
    let line = before.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let column = before
        .rsplit_once('\n')
        .map_or(before.len(), |(_, tail)| tail.len())
        + 1;
    Some(format!("{}:{line}:{column}", file.path.display()))
}
