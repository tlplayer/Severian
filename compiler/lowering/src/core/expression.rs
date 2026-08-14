use super::*;

impl LowerContext<'_> {
    pub(super) fn lower_expression(&mut self, expression: &Expression) -> (String, ValueType) {
        match expression {
            Expression::Typed {
                id, ty, expression, ..
            } => {
                let previous = self.active_hir_id.replace(*id);
                let result = if let Expression::Conditional {
                    condition,
                    then_expression,
                    else_expression,
                } = expression.kind()
                {
                    self.lower_conditional_expression(
                        condition,
                        then_expression,
                        else_expression,
                        Some(*ty),
                    )
                } else if let Expression::Slice {
                    object,
                    start,
                    end,
                    step,
                } = expression.kind()
                {
                    self.lower_slice_expression(object, start, end, step, Some(*ty))
                } else {
                    let (value, lowered_type) = self.lower_expression(expression);
                    if lowered_type == ValueType::Any && *ty != ValueType::Any {
                        let (value, coerced_type) = self.unbox_value((value, lowered_type), *ty);
                        (
                            value,
                            if coerced_type == ValueType::Any {
                                *ty
                            } else {
                                coerced_type
                            },
                        )
                    } else {
                        (value, lowered_type)
                    }
                };
                self.active_hir_id = previous;
                result
            }
            Expression::Integer(value) => {
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant({value} : i64) : i64"
                )
                .unwrap();
                (result, ValueType::Int)
            }
            Expression::Float(bits) => {
                let result = self.fresh_value();
                let value = f64::from_bits(*bits);
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant({value:.17e} : f64) : f64"
                )
                .unwrap();
                (result, ValueType::Float)
            }
            Expression::Boolean(value) => {
                let result = self.fresh_value();
                let value = i32::from(*value);
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant({value} : i1) : i1"
                )
                .unwrap();
                (result, ValueType::Bool)
            }
            Expression::String(value) => {
                let index = self
                    .strings
                    .iter()
                    .position(|candidate| candidate == value)
                    .unwrap();
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.addressof @__sev_str_{index} : !llvm.ptr"
                )
                .unwrap();
                (result, ValueType::String)
            }
            Expression::Variable(name) => {
                if let Some(value) = self.variables.get(&name.id).cloned() {
                    value
                } else if self.field_names.contains(&name.name) {
                    let object = self.field_object.clone().unwrap();
                    let field = self.string_address(&name.name);
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_object_get({object}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                    if let Some(class) = self.field_classes.get(&name.name).cloned() {
                        self.object_classes.insert(result.clone(), class);
                    }
                    let ty = self
                        .field_types
                        .get(&name.name)
                        .copied()
                        .unwrap_or(ValueType::Any);
                    if ty == ValueType::Any || matches!(ty, ValueType::Tensor(_)) {
                        (result, ty)
                    } else {
                        self.unbox_value((result, ValueType::Any), ty)
                    }
                } else {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    (result, ValueType::Any)
                }
            }
            Expression::Ownership {
                op: OwnershipOp::Clone,
                value,
            } => {
                let (value, ty) = self.lower_expression(value);
                if matches!(ty, ValueType::List | ValueType::Tuple | ValueType::Set) {
                    let result = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {result} = llvm.call @__sev_collection_clone({value}) : (!llvm.ptr) -> !llvm.ptr"
                    )
                    .unwrap();
                    (result, ty)
                } else {
                    (value, ty)
                }
            }
            Expression::Ownership { value, .. } => self.lower_expression(value),
            Expression::Function(target) => {
                let adapter = self.ensure_function_closure(target);
                let kind = self.fresh_value();
                writeln!(
                    self.output,
                    "    {kind} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                let environment = self.fresh_value();
                writeln!(self.output, "    {environment} = llvm.call @__sev_collection_new({kind}) : (i64) -> !llvm.ptr").unwrap();
                let function = self.fresh_value();
                writeln!(
                    self.output,
                    "    {function} = llvm.mlir.addressof @{adapter} : !llvm.ptr"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_closure_new({function}, {environment}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::Function)
            }
            Expression::Lambda { params, body } => {
                let name = self.emit_lambda_closure(params, body);
                (name, ValueType::Function)
            }
            Expression::Closure { params, body, .. } => {
                let name = self.emit_block_closure(params, body);
                (name, ValueType::Function)
            }
            Expression::List(values) => self.lower_collection_literal(values, ValueType::List, 0),
            Expression::Tuple(values) => self.lower_collection_literal(values, ValueType::Tuple, 1),
            Expression::Set(values) => self.lower_collection_literal(values, ValueType::Set, 2),
            Expression::Map(entries) => {
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_map_new() : () -> !llvm.ptr"
                )
                .unwrap();
                for (key, value) in entries {
                    let key = self.lower_expression(key);
                    let value = self.lower_expression(value);
                    let key = self.box_value(key);
                    let value = self.box_value(value);
                    writeln!(self.output, "    llvm.call @__sev_map_insert({result}, {key}, {value}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                }
                (result, ValueType::Map)
            }
            Expression::ListComprehension { element, clauses } => {
                self.lower_comprehension(Some(element), None, None, clauses, ValueType::List)
            }
            Expression::SetComprehension { element, clauses } => {
                self.lower_comprehension(Some(element), None, None, clauses, ValueType::Set)
            }
            Expression::MapComprehension {
                key,
                value,
                clauses,
            } => self.lower_comprehension(None, Some(key), Some(value), clauses, ValueType::Map),
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => self.lower_conditional_expression(
                condition,
                then_expression,
                else_expression,
                then_expression.ty().or_else(|| else_expression.ty()),
            ),
            Expression::FusedPipeline {
                input,
                runtime_symbol,
                operations,
                packing_bits,
            } => {
                let (input, _) = self.lower_expression(input);
                let packed = operations
                    .iter()
                    .enumerate()
                    .fold(0i64, |packed, (index, opcode)| {
                        packed | (i64::from(*opcode) << (index * usize::from(*packing_bits)))
                    });
                let packed_value = self.fresh_value();
                writeln!(
                    self.output,
                    "    {packed_value} = llvm.mlir.constant({packed} : i64) : i64"
                )
                .unwrap();
                let count = self.fresh_value();
                writeln!(
                    self.output,
                    "    {count} = llvm.mlir.constant({} : i64) : i64",
                    operations.len()
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @{runtime_symbol}({input}, {packed_value}, {count}) {{severian_fusion = \"automatic\", severian_parallel = \"auto\", severian_candidates = \"simd,simt,gpu\", severian_device_fallback = \"cpu\"}} : (!llvm.ptr, i64, i64) -> !llvm.ptr").unwrap();
                (result, ValueType::List)
            }
            Expression::PrintArgs(values) => {
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        writeln!(self.output, "    llvm.call @__sev_print_space() : () -> ()")
                            .unwrap();
                    }
                    let value = self.lower_expression(value);
                    let value = self.box_value(value);
                    writeln!(
                        self.output,
                        "    llvm.call @__sev_print_value_inline({value}) : (!llvm.ptr) -> ()"
                    )
                    .unwrap();
                }
                writeln!(
                    self.output,
                    "    llvm.call @__sev_print_newline() : () -> ()"
                )
                .unwrap();
                (String::new(), ValueType::Unit)
            }
            Expression::Construct { class, args, .. } => {
                let class_name = self.string_address(class);
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_object_new({class_name}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                let lowered_args = args
                    .iter()
                    .map(|arg| self.lower_expression(arg))
                    .collect::<Vec<_>>();
                let definition = self
                    .classes
                    .iter()
                    .find(|candidate| candidate.name == *class)
                    .cloned();
                if let Some(definition) = &definition {
                    for field in &definition.fields {
                        let field = self.string_address(field);
                        writeln!(self.output, "    llvm.call @__sev_object_declare({result}, {field}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
                    }
                }
                let constructor = definition.as_ref().and_then(|definition| {
                    definition
                        .constructors
                        .iter()
                        .find(|constructor| constructor.params.len() == args.len())
                });
                if let Some(constructor) = constructor {
                    let lowered_args = lowered_args
                        .into_iter()
                        .enumerate()
                        .map(|(index, argument)| {
                            let expected = constructor.params[index].ty;
                            if argument.1 == ValueType::Any && expected != ValueType::Any {
                                self.unbox_value(argument, expected)
                            } else if expected == ValueType::Any && argument.1 != ValueType::Any {
                                (self.box_value(argument), ValueType::Any)
                            } else {
                                argument
                            }
                        })
                        .collect::<Vec<_>>();
                    let values = lowered_args
                        .iter()
                        .map(|(value, _)| value.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let types = lowered_args
                        .iter()
                        .map(|(_, ty)| mlir_type(*ty))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let value_suffix = if values.is_empty() {
                        String::new()
                    } else {
                        format!(", {values}")
                    };
                    let type_suffix = if types.is_empty() {
                        String::new()
                    } else {
                        format!(", {types}")
                    };
                    let constructor_symbol =
                        class_function_symbol(class, &format!("ctor_{}", constructor.params.len()));
                    writeln!(self.output, "    llvm.call @{constructor_symbol}({result}{value_suffix}) : (!llvm.ptr{type_suffix}) -> ()").unwrap();
                } else if let Some(definition) = definition {
                    for (index, field) in definition.fields.iter().enumerate() {
                        let value = lowered_args.get(index).cloned().or_else(|| {
                            definition.field_defaults[index]
                                .as_ref()
                                .map(|default| self.lower_expression(default))
                        });
                        let Some(value) = value else { continue };
                        let field = self.string_address(field);
                        let value = self.box_value(value);
                        writeln!(self.output, "    llvm.call @__sev_object_set({result}, {field}, {value}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                    }
                }
                self.object_classes.insert(result.clone(), class.clone());
                if let Some(definition) = self
                    .classes
                    .iter()
                    .find(|candidate| candidate.name == *class)
                {
                    self.object_class_ids.insert(result.clone(), definition.id);
                }
                self.validate_object(&result, class);
                (result, ValueType::Any)
            }
            Expression::ConstructFields {
                class,
                fields,
                validate,
                ..
            } => {
                let explicit = fields
                    .iter()
                    .map(|(name, value)| (name.as_str(), value))
                    .collect::<HashMap<_, _>>();
                self.lower_field_construction(class, None, &explicit, false, *validate)
            }
            Expression::ObjectUpdate {
                object,
                class,
                fields,
                json_document,
                ..
            } => {
                let source = self.lower_expression(object).0;
                let explicit = fields
                    .iter()
                    .map(|(name, value)| (name.as_str(), value))
                    .collect::<HashMap<_, _>>();
                self.lower_field_construction(class, Some(&source), &explicit, *json_document, true)
            }
            Expression::ObjectDocument { object, fields } => {
                let source = self.lower_expression(object).0;
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_map_new() : () -> !llvm.ptr"
                )
                .unwrap();
                for field in fields {
                    let field_name = self.string_address(field);
                    let value = self.fresh_value();
                    writeln!(self.output, "    {value} = llvm.call @__sev_object_get({source}, {field_name}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                    let key = self.fresh_value();
                    writeln!(self.output, "    {key} = llvm.call @__sev_box_string({field_name}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    writeln!(self.output, "    llvm.call @__sev_map_insert({result}, {key}, {value}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                }
                (result, ValueType::Map)
            }
            Expression::Member { object, member } => {
                let (object, _) = self.lower_expression(object);
                let field = self.string_address(member);
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_object_get({object}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                let metadata = self.object_field_metadata(&object, member);
                if let Some((_, Some(class))) = &metadata {
                    self.object_classes.insert(result.clone(), class.clone());
                }
                let ty = metadata.map(|(ty, _)| ty).unwrap_or(ValueType::Any);
                if ty == ValueType::Any || matches!(ty, ValueType::Tensor(_)) {
                    (result, ty)
                } else {
                    self.unbox_value((result, ValueType::Any), ty)
                }
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if self.has_known_class_method(object, method) => {
                self.lower_known_class_method_call(object, method, args)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "get"
                && args.len() == 1
                && !self.has_known_class_method(object, method)
                && !self.has_abstract_class_method(object, method) =>
            {
                let (object, _) = self.lower_expression(object);
                let (field, _) = self.lower_expression(&args[0]);
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_dynamic_object_get({object}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                let metadata = match args[0].kind() {
                    Expression::String(name) => self.object_field_metadata(&object, name),
                    _ => None,
                };
                if let Some((_, Some(class))) = &metadata {
                    self.object_classes.insert(result.clone(), class.clone());
                }
                let ty = metadata.map(|(ty, _)| ty).unwrap_or(ValueType::Any);
                if ty == ValueType::Any || matches!(ty, ValueType::Tensor(_)) {
                    (result, ty)
                } else {
                    self.unbox_value((result, ValueType::Any), ty)
                }
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "set"
                && args.len() == 2
                && !self.has_known_class_method(object, method)
                && !self.has_abstract_class_method(object, method) =>
            {
                let (object, _) = self.lower_expression(object);
                let (field, _) = self.lower_expression(&args[0]);
                let value = self.lower_expression(&args[1]);
                let value = self.box_value(value);
                writeln!(self.output, "    llvm.call @__sev_object_set({object}, {field}, {value}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                if let Some(class) = self.object_classes.get(&object).cloned() {
                    self.validate_object(&object, &class);
                }
                (String::new(), ValueType::Unit)
            }
            Expression::ChaosRule { .. } => {
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                (result, ValueType::Any)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "len" | "size" | "bytes" | "bits" | "capacity"
            ) && args.is_empty()
                && !self.has_known_class_method(object, method)
                && !self.has_abstract_class_method(object, method) =>
            {
                let (value, ty) = self.lower_expression(object);
                if let ValueType::Tensor(tensor) = ty {
                    if let Some(elements) = static_tensor_elements(tensor) {
                        let amount = match method.as_str() {
                            "bytes" => elements * tensor_element_bytes(tensor.element),
                            "bits" => elements * tensor_element_bytes(tensor.element) * 8,
                            _ => elements,
                        };
                        let result = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {result} = llvm.mlir.constant({amount} : i64) : i64"
                        )
                        .unwrap();
                        return (result, ValueType::Int);
                    }
                }
                let runtime = match (method.as_str(), ty) {
                    ("len" | "size", ValueType::String) => "__sev_string_length",
                    ("len" | "size", ValueType::Map) => "__sev_map_size",
                    ("len" | "size", ValueType::Tensor(_) | ValueType::TensorAny) => {
                        "__sev_tensor_size"
                    }
                    ("len" | "size", _) => "__sev_collection_size",
                    ("capacity", ValueType::Map) => "__sev_map_capacity",
                    ("capacity", ValueType::Tensor(_) | ValueType::TensorAny) => {
                        "__sev_tensor_capacity"
                    }
                    ("capacity", _) => "__sev_value_capacity",
                    ("bytes" | "bits", ValueType::Map) => "__sev_map_bytes",
                    ("bytes" | "bits", ValueType::Tensor(_) | ValueType::TensorAny) => {
                        "__sev_tensor_bytes"
                    }
                    ("bytes" | "bits", _) => "__sev_value_bytes",
                    _ => unreachable!(),
                };
                let requires_box = matches!(
                    (method.as_str(), ty),
                    (
                        "bytes" | "bits" | "capacity",
                        ValueType::String | ValueType::List | ValueType::Tuple | ValueType::Set
                    )
                );
                let value = if requires_box {
                    self.box_value((value, ty))
                } else {
                    value
                };
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @{runtime}({value}) : (!llvm.ptr) -> i64"
                )
                .unwrap();
                if method == "bits" {
                    let eight = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {eight} = llvm.mlir.constant(8 : i64) : i64"
                    )
                    .unwrap();
                    let bits = self.fresh_value();
                    writeln!(self.output, "    {bits} = llvm.mul {result}, {eight} : i64").unwrap();
                    (bits, ValueType::Int)
                } else {
                    (result, ValueType::Int)
                }
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "append" => {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    let unboxed = self.fresh_value();
                    writeln!(self.output, "    {unboxed} = llvm.call @__sev_unbox_ptr({object}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    object = unboxed;
                }
                let value = self.lower_expression(&args[0]);
                let value = self.box_value(value);
                writeln!(self.output, "    llvm.call @__sev_collection_push({object}, {value}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
                (String::new(), ValueType::Unit)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "append_left" | "appendleft" | "extend" | "remove" | "heap_push" | "heapPush"
            ) && args.len() == 1
                && !(method == "remove" && object.ty() == Some(ValueType::String)) =>
            {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::List).0;
                }
                let argument = self.lower_expression(&args[0]);
                let function = match method.as_str() {
                    "append_left" | "appendleft" => "__sev_collection_appendleft",
                    "extend" => "__sev_collection_extend",
                    "remove" => "__sev_collection_remove",
                    "heap_push" | "heapPush" => "__sev_collection_heap_push",
                    _ => unreachable!(),
                };
                let argument = if method == "extend" {
                    self.unbox_value(argument, ValueType::List).0
                } else {
                    self.box_value(argument)
                };
                writeln!(self.output, "    llvm.call @{function}({object}, {argument}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
                (String::new(), ValueType::Unit)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "insert" && args.len() == 2 => {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::List).0;
                }
                let index = self.lower_expression(&args[0]);
                let index = self.unbox_value(index, ValueType::Int).0;
                let value = self.lower_expression(&args[1]);
                let value = self.box_value(value);
                writeln!(self.output, "    llvm.call @__sev_collection_insert({object}, {index}, {value}) : (!llvm.ptr, i64, !llvm.ptr) -> ()").unwrap();
                (String::new(), ValueType::Unit)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "pop"
                && (1..=2).contains(&args.len())
                && object.ty() == Some(ValueType::Map) =>
            {
                let (object, _) = self.lower_expression(object);
                let key = self.lower_expression(&args[0]);
                let key = self.box_value(key);
                let result = self.fresh_value();
                if let Some(fallback) = args.get(1) {
                    let fallback = self.lower_expression(fallback);
                    let fallback = self.box_value(fallback);
                    writeln!(self.output, "    {result} = llvm.call @__sev_map_pop({object}, {key}, {fallback}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                } else {
                    writeln!(self.output, "    {result} = llvm.call @__sev_map_pop_required({object}, {key}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                }
                (result, ValueType::Any)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "pop" && args.len() == 1 => {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::List).0;
                }
                let index = self.lower_expression(&args[0]);
                let index = self.unbox_value(index, ValueType::Int).0;
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_collection_pop_at({object}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                (result, ValueType::Any)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "reversed" && args.is_empty() => {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    let unboxed = self.fresh_value();
                    writeln!(self.output, "    {unboxed} = llvm.call @__sev_unbox_ptr({object}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    object = unboxed;
                }
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_collection_reversed({object}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::List)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "filter"
                && args.len() == 1
                && object.ty() == Some(ValueType::String) =>
            {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let characters = self.fresh_value();
                writeln!(self.output, "    {characters} = llvm.call @__sev_string_characters({object}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                let filtered =
                    self.lower_collection_transform_from_value(characters, &args[0], true);
                let separator = self.string_address("");
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_collection_join({filtered}, {separator}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(method.as_str(), "map" | "filter")
                && args.len() == 1
                && !self.has_known_class_method(object, method)
                && !self.has_abstract_class_method(object, method) =>
            {
                self.lower_collection_transform(object, &args[0], method == "filter")
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "reduce"
                && (1..=2).contains(&args.len())
                && !self.has_known_class_method(object, method)
                && !self.has_abstract_class_method(object, method) =>
            {
                self.lower_collection_reduce(object, &args[0], args.get(1))
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "sorted"
                && !args.is_empty()
                && !args
                    .first()
                    .is_some_and(|argument| matches!(argument.kind(), Expression::Boolean(_))) =>
            {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::List).0;
                }
                let keys =
                    self.lower_collection_transform_from_value(object.clone(), &args[0], false);
                let reverse = if let Some(reverse) = args.get(1) {
                    let reverse = self.lower_expression(reverse);
                    self.unbox_value(reverse, ValueType::Bool).0
                } else {
                    let reverse = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {reverse} = llvm.mlir.constant(0 : i1) : i1"
                    )
                    .unwrap();
                    reverse
                };
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_collection_sorted_keys({object}, {keys}, {reverse}) : (!llvm.ptr, !llvm.ptr, i1) -> !llvm.ptr").unwrap();
                (result, ValueType::List)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "sorted" && args.len() == 1 => {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::List).0;
                }
                let reverse = self.lower_expression(&args[0]);
                let reverse = self.unbox_value(reverse, ValueType::Bool).0;
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_collection_sorted_reverse({object}, {reverse}) : (!llvm.ptr, i1) -> !llvm.ptr").unwrap();
                (result, ValueType::List)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "pop"
                    | "pop_left"
                    | "popleft"
                    | "heap_pop"
                    | "heapPop"
                    | "last"
                    | "sorted"
                    | "sum"
                    | "minimum"
                    | "maximum"
            ) && args.is_empty() =>
            {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::List).0;
                }
                let result = self.fresh_value();
                let function = match method.as_str() {
                    "pop" => "__sev_collection_pop",
                    "pop_left" | "popleft" => "__sev_collection_pop_at",
                    "heap_pop" | "heapPop" => "__sev_collection_heap_pop",
                    "last" => "__sev_collection_last",
                    "sorted" => "__sev_collection_sorted",
                    "sum" => "__sev_collection_sum",
                    "minimum" => "__sev_collection_minimum",
                    "maximum" => "__sev_collection_maximum",
                    _ => unreachable!(),
                };
                if matches!(method.as_str(), "pop_left" | "popleft") {
                    let zero = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                    )
                    .unwrap();
                    writeln!(self.output, "    {result} = llvm.call @{function}({object}, {zero}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                } else {
                    writeln!(
                        self.output,
                        "    {result} = llvm.call @{function}({object}) : (!llvm.ptr) -> !llvm.ptr"
                    )
                    .unwrap();
                }
                if method == "sorted" {
                    (result, ValueType::List)
                } else {
                    (result, ValueType::Any)
                }
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "join" && args.len() == 1 && object.ty() == Some(ValueType::String) => {
                let (separator, separator_type) = self.lower_expression(object);
                let separator = self
                    .unbox_value((separator, separator_type), ValueType::String)
                    .0;
                let values = self.lower_expression(&args[0]);
                let values = self.unbox_value(values, ValueType::List).0;
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_collection_join({values}, {separator}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "join"
                && args.len() == 1
                && !self.has_known_class_method(object, method) =>
            {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::List).0;
                }
                let separator = self.lower_expression(&args[0]);
                let separator = self.unbox_value(separator, ValueType::String).0;
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_collection_join({object}, {separator}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(method.as_str(), "to_set" | "toSet" | "to_list" | "toList")
                && args.is_empty() =>
            {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::List).0;
                }
                let result = self.fresh_value();
                let function = if matches!(method.as_str(), "to_set" | "toSet") {
                    "__sev_collection_to_set"
                } else {
                    "__sev_set_to_list"
                };
                writeln!(
                    self.output,
                    "    {result} = llvm.call @{function}({object}) : (!llvm.ptr) -> !llvm.ptr"
                )
                .unwrap();
                if matches!(method.as_str(), "to_set" | "toSet") {
                    (result, ValueType::Set)
                } else {
                    (result, ValueType::List)
                }
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "difference" && args.len() == 1 => {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::List).0;
                }
                let (mut excluded, excluded_type) = self.lower_expression(&args[0]);
                if excluded_type == ValueType::Any {
                    excluded = self
                        .unbox_value((excluded, excluded_type), ValueType::List)
                        .0;
                }
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_set_difference({object}, {excluded}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::Set)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "union" | "intersection" | "symmetric_difference" | "symmetricDifference"
            ) && args.len() == 1 =>
            {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::Set).0;
                }
                let (mut other, other_type) = self.lower_expression(&args[0]);
                if other_type == ValueType::Any {
                    other = self.unbox_value((other, other_type), ValueType::Set).0;
                }
                let operation = match method.as_str() {
                    "union" => 0,
                    "intersection" => 1,
                    _ => 2,
                };
                let operation_value = self.fresh_value();
                writeln!(
                    self.output,
                    "    {operation_value} = llvm.mlir.constant({operation} : i64) : i64"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_set_combine({object}, {other}, {operation_value}) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr").unwrap();
                (result, ValueType::Set)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "characters" | "words" | "splitlines" | "split_lines" | "lines"
            ) && args.is_empty() =>
            {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let result = self.fresh_value();
                let function = match method.as_str() {
                    "characters" => "__sev_string_characters",
                    "words" => "__sev_string_words",
                    "splitlines" | "split_lines" | "lines" => "__sev_string_split_lines",
                    _ => unreachable!(),
                };
                writeln!(
                    self.output,
                    "    {result} = llvm.call @{function}({object}) : (!llvm.ptr) -> !llvm.ptr"
                )
                .unwrap();
                (result, ValueType::List)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "frequencies" && args.is_empty() => {
                let (mut object, object_type) = self.lower_expression(object);
                let function = if object_type == ValueType::String {
                    "__sev_string_frequencies"
                } else {
                    if object_type == ValueType::Any {
                        object = self.unbox_value((object, object_type), ValueType::List).0;
                    }
                    "__sev_collection_frequencies"
                };
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @{function}({object}) : (!llvm.ptr) -> !llvm.ptr"
                )
                .unwrap();
                (result, ValueType::Map)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "split" && args.is_empty() => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_words({object}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::List)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(method.as_str(), "split" | "rsplit") && (1..=2).contains(&args.len()) => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let separator = self.lower_expression(&args[0]);
                let separator = self.unbox_value(separator, ValueType::String).0;
                let limit = if let Some(limit) = args.get(1) {
                    let limit = self.lower_expression(limit);
                    self.unbox_value(limit, ValueType::Int).0
                } else {
                    let limit = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {limit} = llvm.mlir.constant(-1 : i64) : i64"
                    )
                    .unwrap();
                    limit
                };
                let reverse = i32::from(method == "rsplit");
                let reverse_value = self.fresh_value();
                writeln!(
                    self.output,
                    "    {reverse_value} = llvm.mlir.constant({reverse} : i1) : i1"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_split_limit({object}, {separator}, {limit}, {reverse_value}) : (!llvm.ptr, !llvm.ptr, i64, i1) -> !llvm.ptr").unwrap();
                (result, ValueType::List)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "strip"
                    | "lstrip"
                    | "rstrip"
                    | "lower"
                    | "upper"
                    | "capitalize"
                    | "title"
                    | "swapcase"
                    | "collapse_space"
                    | "normalize_space"
                    | "collapse_horizontal_space"
            ) && args.is_empty() =>
            {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let function = match method.as_str() {
                    "strip" => "__sev_string_strip",
                    "lstrip" => "__sev_string_lstrip",
                    "rstrip" => "__sev_string_rstrip",
                    "lower" => "__sev_string_lower",
                    "upper" => "__sev_string_upper",
                    "capitalize" => "__sev_string_capitalize",
                    "title" => "__sev_string_title",
                    "swapcase" => "__sev_string_swapcase",
                    _ => "__sev_string_collapse_space",
                };
                let result = self.fresh_value();
                if matches!(
                    method.as_str(),
                    "collapse_space" | "normalize_space" | "collapse_horizontal_space"
                ) {
                    let horizontal = if method == "collapse_horizontal_space" {
                        1
                    } else {
                        0
                    };
                    let horizontal_value = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {horizontal_value} = llvm.mlir.constant({horizontal} : i1) : i1"
                    )
                    .unwrap();
                    writeln!(self.output, "    {result} = llvm.call @{function}({object}, {horizontal_value}) : (!llvm.ptr, i1) -> !llvm.ptr").unwrap();
                } else {
                    writeln!(
                        self.output,
                        "    {result} = llvm.call @{function}({object}) : (!llvm.ptr) -> !llvm.ptr"
                    )
                    .unwrap();
                }
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "is_empty"
                    | "is_space"
                    | "is_alpha"
                    | "is_digit"
                    | "is_alnum"
                    | "is_ascii"
                    | "is_lower"
                    | "is_upper"
                    | "is_ascii_alnum"
                    | "is_word"
                    | "is_punctuation"
            ) && args.is_empty() =>
            {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let operation = match method.as_str() {
                    "is_empty" => 0,
                    "is_space" => 1,
                    "is_alpha" => 2,
                    "is_digit" => 3,
                    "is_alnum" => 4,
                    "is_ascii" => 5,
                    "is_lower" => 6,
                    "is_upper" => 7,
                    "is_ascii_alnum" => 8,
                    "is_word" => 9,
                    _ => 10,
                };
                let operation_value = self.fresh_value();
                writeln!(
                    self.output,
                    "    {operation_value} = llvm.mlir.constant({operation} : i64) : i64"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_predicate({object}, {operation_value}) : (!llvm.ptr, i64) -> i1").unwrap();
                (result, ValueType::Bool)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "starts_with"
                    | "startsWith"
                    | "ends_with"
                    | "endsWith"
                    | "contains"
                    | "find"
                    | "rfind"
                    | "index"
                    | "rindex"
                    | "count"
            ) && args.len() == 1 =>
            {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let needle = self.lower_expression(&args[0]);
                let needle = self.unbox_value(needle, ValueType::String).0;
                let function = match method.as_str() {
                    "starts_with" | "startsWith" => "__sev_string_starts_with",
                    "ends_with" | "endsWith" => "__sev_string_ends_with",
                    "contains" => "__sev_string_contains",
                    "find" => "__sev_string_find",
                    "rfind" => "__sev_string_rfind",
                    "index" => "__sev_string_find",
                    "rindex" => "__sev_string_rfind",
                    _ => "__sev_string_count",
                };
                let result = self.fresh_value();
                let ty = if matches!(
                    method.as_str(),
                    "starts_with" | "startsWith" | "ends_with" | "endsWith" | "contains"
                ) {
                    ValueType::Bool
                } else {
                    ValueType::Int
                };
                writeln!(self.output, "    {result} = llvm.call @{function}({object}, {needle}) : (!llvm.ptr, !llvm.ptr) -> {}", mlir_type(ty)).unwrap();
                if matches!(method.as_str(), "index" | "rindex") {
                    let found = self.fresh_value();
                    let zero = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "    {found} = llvm.icmp \"sge\" {result}, {zero} : i64"
                    )
                    .unwrap();
                    let ok = self.fresh_block();
                    let missing = self.fresh_block();
                    writeln!(
                        self.output,
                        "    llvm.cond_br {found}, ^bb{ok}, ^bb{missing}"
                    )
                    .unwrap();
                    writeln!(self.output, "  ^bb{missing}:").unwrap();
                    writeln!(
                        self.output,
                        "    llvm.call @abort() : () -> ()\n    llvm.unreachable"
                    )
                    .unwrap();
                    writeln!(self.output, "  ^bb{ok}:").unwrap();
                }
                (result, ty)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "trim_prefix" | "trim_suffix" | "remove_prefix" | "remove_suffix"
            ) && args.len() == 1 =>
            {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let affix = self.lower_expression(&args[0]);
                let affix = self.unbox_value(affix, ValueType::String).0;
                let suffix = i32::from(method.ends_with("suffix"));
                let repeated = i32::from(method.starts_with("trim_"));
                let suffix_value = self.fresh_value();
                let repeated_value = self.fresh_value();
                writeln!(
                    self.output,
                    "    {suffix_value} = llvm.mlir.constant({suffix} : i1) : i1"
                )
                .unwrap();
                writeln!(
                    self.output,
                    "    {repeated_value} = llvm.mlir.constant({repeated} : i1) : i1"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_remove_affix({object}, {affix}, {suffix_value}, {repeated_value}) : (!llvm.ptr, !llvm.ptr, i1, i1) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(method.as_str(), "translate" | "replace_many") && args.len() == 1 => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let mapping = self.lower_expression(&args[0]);
                let mapping = self.unbox_value(mapping, ValueType::Map).0;
                let function = if method == "translate" {
                    "__sev_string_translate"
                } else {
                    "__sev_string_replace_many"
                };
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @{function}({object}, {mapping}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "remove" || method == "remove_all" => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let removal = self.lower_expression(&args[0]);
                let result = self.fresh_value();
                if method == "remove" {
                    let removal_type = removal.1;
                    if removal_type == ValueType::List {
                        let matches = self.unbox_value(removal, ValueType::List).0;
                        writeln!(self.output, "    {result} = llvm.call @__sev_string_remove_matches({object}, {matches}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                    } else {
                        let characters = self.unbox_value(removal, ValueType::String).0;
                        writeln!(self.output, "    {result} = llvm.call @__sev_string_remove({object}, {characters}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                    }
                } else {
                    let needle = self.unbox_value(removal, ValueType::String).0;
                    let empty = self.string_address("");
                    writeln!(self.output, "    {result} = llvm.call @__sev_string_replace({object}, {needle}, {empty}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                }
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "repeat" && args.len() == 1 => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let count = self.lower_expression(&args[0]);
                let count = self.unbox_value(count, ValueType::Int).0;
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_repeat({object}, {count}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(method.as_str(), "pad_left" | "pad_right" | "center")
                && args.len() == 1 =>
            {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let width = self.lower_expression(&args[0]);
                let width = self.unbox_value(width, ValueType::Int).0;
                let alignment = match method.as_str() {
                    "pad_left" => 0,
                    "pad_right" => 1,
                    _ => 2,
                };
                let alignment_value = self.fresh_value();
                writeln!(
                    self.output,
                    "    {alignment_value} = llvm.mlir.constant({alignment} : i64) : i64"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_pad({object}, {width}, {alignment_value}) : (!llvm.ptr, i64, i64) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(method.as_str(), "first" | "take" | "last" | "drop")
                && args.len() == 1 =>
            {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let count = self.lower_expression(&args[0]);
                let count = self.unbox_value(count, ValueType::Int).0;
                let operation = match method.as_str() {
                    "first" | "take" => 0,
                    "last" => 1,
                    _ => 2,
                };
                let operation_value = self.fresh_value();
                writeln!(
                    self.output,
                    "    {operation_value} = llvm.mlir.constant({operation} : i64) : i64"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_take({object}, {count}, {operation_value}) : (!llvm.ptr, i64, i64) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "slice" && args.len() == 2 => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let start = self.lower_expression(&args[0]);
                let start = self.unbox_value(start, ValueType::Int).0;
                let end = self.lower_expression(&args[1]);
                let end = self.unbox_value(end, ValueType::Int).0;
                let step = self.fresh_value();
                writeln!(
                    self.output,
                    "    {step} = llvm.mlir.constant(1 : i64) : i64"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_slice({object}, {start}, {end}, {step}) : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "before" | "after" | "before_last" | "after_last"
            ) && args.len() == 1 =>
            {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let separator = self.lower_expression(&args[0]);
                let separator = self.unbox_value(separator, ValueType::String).0;
                let operation = match method.as_str() {
                    "before" => 0,
                    "after" => 1,
                    "before_last" => 2,
                    _ => 3,
                };
                let operation_value = self.fresh_value();
                writeln!(
                    self.output,
                    "    {operation_value} = llvm.mlir.constant({operation} : i64) : i64"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_segment({object}, {separator}, {operation_value}) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "between" && args.len() == 2 => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let opener = self.lower_expression(&args[0]);
                let opener = self.unbox_value(opener, ValueType::String).0;
                let closer = self.lower_expression(&args[1]);
                let closer = self.unbox_value(closer, ValueType::String).0;
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_between({object}, {opener}, {closer}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "replace" && (2..=3).contains(&args.len()) => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let old = self.lower_expression(&args[0]);
                let old = self.unbox_value(old, ValueType::String).0;
                let new = self.lower_expression(&args[1]);
                let new = self.unbox_value(new, ValueType::String).0;
                let result = self.fresh_value();
                if let Some(limit) = args.get(2) {
                    let limit = self.lower_expression(limit);
                    let limit = self.unbox_value(limit, ValueType::Int).0;
                    writeln!(self.output, "    {result} = llvm.call @__sev_string_replace_limit({object}, {old}, {new}, {limit}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr").unwrap();
                } else {
                    writeln!(self.output, "    {result} = llvm.call @__sev_string_replace({object}, {old}, {new}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                }
                (result, ValueType::String)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(method.as_str(), "partition" | "rpartition") && args.len() == 1 => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let separator = self.lower_expression(&args[0]);
                let separator = self.unbox_value(separator, ValueType::String).0;
                let reverse = i32::from(method == "rpartition");
                let reverse_value = self.fresh_value();
                writeln!(
                    self.output,
                    "    {reverse_value} = llvm.mlir.constant({reverse} : i1) : i1"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_partition({object}, {separator}, {reverse_value}) : (!llvm.ptr, !llvm.ptr, i1) -> !llvm.ptr").unwrap();
                (result, ValueType::Tuple)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(method.as_str(), "keys" | "values")
                && args.is_empty()
                && !self.has_known_class_method(object, method)
                && !self.has_abstract_class_method(object, method) =>
            {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::Map).0;
                }
                let result = self.fresh_value();
                let function = if method == "keys" {
                    "__sev_map_keys"
                } else {
                    "__sev_map_values"
                };
                writeln!(
                    self.output,
                    "    {result} = llvm.call @{function}({object}) : (!llvm.ptr) -> !llvm.ptr"
                )
                .unwrap();
                (result, ValueType::List)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "items"
                && args.is_empty()
                && !self.has_known_class_method(object, method)
                && !self.has_abstract_class_method(object, method) =>
            {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    object = self.unbox_value((object, object_type), ValueType::Map).0;
                }
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_map_items({object}) : (!llvm.ptr) -> !llvm.ptr"
                )
                .unwrap();
                (result, ValueType::List)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(method.as_str(), "get" | "set_default" | "setDefault")
                && args.len() == 2
                && !self.has_known_class_method(object, method)
                && !self.has_abstract_class_method(object, method) =>
            {
                let (object, _) = self.lower_expression(object);
                let key = self.lower_expression(&args[0]);
                let key = self.box_value(key);
                let fallback = self.lower_expression(&args[1]);
                let fallback = self.box_value(fallback);
                let function = if method == "get" {
                    "__sev_map_get_default"
                } else {
                    "__sev_map_set_default"
                };
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @{function}({object}, {key}, {fallback}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::Any)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "length" && args.is_empty() => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_string_length({object}) : (!llvm.ptr) -> i64"
                )
                .unwrap();
                (result, ValueType::Int)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "wrapping_add" | "saturating_add" | "overflowing_add"
            ) && args.len() == 1 =>
            {
                let (left, left_type) = self.lower_expression(object);
                let left = self.unbox_value((left, left_type), ValueType::Int).0;
                let (right, right_type) = self.lower_expression(&args[0]);
                let right = self.unbox_value((right, right_type), ValueType::Int).0;
                let sum = self.fresh_value();
                writeln!(self.output, "    {sum} = llvm.add {left}, {right} : i64").unwrap();
                let limit = self.fresh_value();
                writeln!(
                    self.output,
                    "    {limit} = llvm.mlir.constant(256 : i64) : i64"
                )
                .unwrap();
                let wrapped = self.fresh_value();
                writeln!(
                    self.output,
                    "    {wrapped} = llvm.urem {sum}, {limit} : i64"
                )
                .unwrap();
                if method == "wrapping_add" {
                    return (wrapped, ValueType::Int);
                }
                let max = self.fresh_value();
                writeln!(
                    self.output,
                    "    {max} = llvm.mlir.constant(255 : i64) : i64"
                )
                .unwrap();
                let overflowed = self.fresh_value();
                writeln!(
                    self.output,
                    "    {overflowed} = llvm.icmp \"sgt\" {sum}, {max} : i64"
                )
                .unwrap();
                if method == "saturating_add" {
                    let capped = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {capped} = llvm.select {overflowed}, {max}, {sum} : i1, i64"
                    )
                    .unwrap();
                    return (capped, ValueType::Int);
                }
                let kind = self.fresh_value();
                writeln!(
                    self.output,
                    "    {kind} = llvm.mlir.constant(1 : i64) : i64"
                )
                .unwrap();
                let tuple = self.fresh_value();
                writeln!(
                    self.output,
                    "    {tuple} = llvm.call @__sev_collection_new({kind}) : (i64) -> !llvm.ptr"
                )
                .unwrap();
                let wrapped = self.box_value((wrapped, ValueType::Int));
                let overflowed = self.box_value((overflowed, ValueType::Bool));
                writeln!(self.output, "    llvm.call @__sev_collection_push({tuple}, {wrapped}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
                writeln!(self.output, "    llvm.call @__sev_collection_push({tuple}, {overflowed}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
                (tuple, ValueType::Tuple)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "lessThan" && args.len() == 1 => {
                let left = self.lower_expression(object);
                let right = self.lower_expression(&args[0]);
                let left = self.box_value(left);
                let right = self.box_value(right);
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_value_less({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
                (result, ValueType::Bool)
            }
            Expression::MethodCall {
                object: _,
                method,
                args,
            } if method == "zero" && args.is_empty() => {
                let zero = self.fresh_value();
                writeln!(
                    self.output,
                    "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                let result = self.box_value((zero, ValueType::Int));
                (result, ValueType::Any)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "add" && args.len() == 1 => {
                let left = self.lower_expression(object);
                let right = self.lower_expression(&args[0]);
                if left.1 == ValueType::List {
                    let right = self.box_value(right);
                    writeln!(self.output, "    llvm.call @__sev_collection_push({}, {right}) : (!llvm.ptr, !llvm.ptr) -> ()", left.0).unwrap();
                    return (String::new(), ValueType::Unit);
                }
                let left = self.box_value(left);
                let right = self.box_value(right);
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_value_add({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::Any)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } => {
                let (object, _) = self.lower_expression(object);
                let lowered_args = args
                    .iter()
                    .map(|arg| self.lower_expression(arg))
                    .collect::<Vec<_>>();
                let inferred_methods = self
                    .classes
                    .iter()
                    .filter_map(|class| {
                        class
                            .methods
                            .iter()
                            .find(|definition| {
                                definition.name == *method && definition.params.len() == args.len()
                            })
                            .map(|definition| (class, definition))
                    })
                    .collect::<Vec<_>>();
                let canonical_class = self.object_class_ids.get(&object).copied();
                let class = canonical_class
                    .and_then(|id| self.classes.iter().find(|candidate| candidate.id == id))
                    .map(|class| class.name.clone())
                    .or_else(|| self.object_classes.get(&object).cloned())
                    .or_else(|| {
                        (inferred_methods.len() == 1).then(|| inferred_methods[0].0.name.clone())
                    });
                if let Some(receiver) = self.receiver_types.get(&object) {
                    if !receiver.concrete && receiver.methods.iter().any(|name| name == method) {
                        let first = inferred_methods[0].1;
                        let params = first
                            .params
                            .iter()
                            .map(|parameter| parameter.ty)
                            .collect::<Vec<_>>();
                        let lowered = lowered_args
                            .iter()
                            .cloned()
                            .enumerate()
                            .map(|(index, argument)| {
                                let expected = params[index];
                                if argument.1 == ValueType::Any && expected != ValueType::Any {
                                    self.unbox_value(argument, expected)
                                } else if expected == ValueType::Any && argument.1 != ValueType::Any
                                {
                                    (self.box_value(argument), ValueType::Any)
                                } else {
                                    argument
                                }
                            })
                            .collect::<Vec<_>>();
                        let values = lowered
                            .iter()
                            .map(|(value, _)| value.as_str())
                            .collect::<Vec<_>>();
                        let mut operands = object.clone();
                        if !values.is_empty() {
                            operands.push_str(", ");
                            operands.push_str(&values.join(", "));
                        }
                        let mut types = String::from("!llvm.ptr");
                        for parameter in &params {
                            write!(types, ", {}", mlir_type(*parameter)).unwrap();
                        }
                        let symbol =
                            dynamic_method_dispatch_symbol(method, &params, first.return_type);
                        if first.return_type == ValueType::Unit {
                            writeln!(
                                self.output,
                                "    llvm.call @{symbol}({operands}) : ({types}) -> ()"
                            )
                            .unwrap();
                            return (String::new(), ValueType::Unit);
                        }
                        let result = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {result} = llvm.call @{symbol}({operands}) : ({types}) -> {}",
                            mlir_type(first.return_type)
                        )
                        .unwrap();
                        return (result, first.return_type);
                    }
                }
                if class.is_none() && inferred_methods.len() > 1 {
                    let first = inferred_methods[0].1;
                    let params = first
                        .params
                        .iter()
                        .map(|parameter| parameter.ty)
                        .collect::<Vec<_>>();
                    let uniform = inferred_methods.iter().all(|(_, definition)| {
                        definition.return_type == first.return_type
                            && definition
                                .params
                                .iter()
                                .map(|parameter| parameter.ty)
                                .eq(params.iter().copied())
                    });
                    if uniform {
                        let lowered_args = lowered_args
                            .into_iter()
                            .enumerate()
                            .map(|(index, argument)| {
                                let expected = params[index];
                                if argument.1 == ValueType::Any && expected != ValueType::Any {
                                    self.unbox_value(argument, expected)
                                } else if expected == ValueType::Any && argument.1 != ValueType::Any
                                {
                                    (self.box_value(argument), ValueType::Any)
                                } else {
                                    argument
                                }
                            })
                            .collect::<Vec<_>>();
                        let values = lowered_args
                            .iter()
                            .map(|(value, _)| value.as_str())
                            .collect::<Vec<_>>();
                        let mut operands = object.clone();
                        if !values.is_empty() {
                            operands.push_str(", ");
                            operands.push_str(&values.join(", "));
                        }
                        let mut types = String::from("!llvm.ptr");
                        for parameter in &params {
                            write!(types, ", {}", mlir_type(*parameter)).unwrap();
                        }
                        let symbol =
                            dynamic_method_dispatch_symbol(method, &params, first.return_type);
                        if first.return_type == ValueType::Unit {
                            writeln!(
                                self.output,
                                "    llvm.call @{symbol}({operands}) : ({types}) -> ()"
                            )
                            .unwrap();
                            return (String::new(), ValueType::Unit);
                        }
                        let result = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {result} = llvm.call @{symbol}({operands}) : ({types}) -> {}",
                            mlir_type(first.return_type)
                        )
                        .unwrap();
                        return (result, first.return_type);
                    }
                }
                let Some(class) = class else {
                    if method == "draw" {
                        writeln!(
                            self.output,
                            "    llvm.call @__sev_dispatch_draw({object}) : (!llvm.ptr) -> ()"
                        )
                        .unwrap();
                        return (String::new(), ValueType::Unit);
                    }
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                };
                let symbol = class_function_symbol(&class, method);
                let method_definition = self
                    .classes
                    .iter()
                    .find(|candidate| candidate.name == class)
                    .and_then(|definition| {
                        definition
                            .methods
                            .iter()
                            .find(|candidate| candidate.name == *method)
                    })
                    .cloned();
                let lowered_args = lowered_args
                    .into_iter()
                    .enumerate()
                    .map(|(index, argument)| {
                        let expected = method_definition
                            .as_ref()
                            .and_then(|definition| definition.params.get(index))
                            .map(|parameter| parameter.ty)
                            .unwrap_or(argument.1);
                        if argument.1 == ValueType::Any && expected != ValueType::Any {
                            self.unbox_value(argument, expected)
                        } else if expected == ValueType::Any && argument.1 != ValueType::Any {
                            (self.box_value(argument), ValueType::Any)
                        } else {
                            argument
                        }
                    })
                    .collect::<Vec<_>>();
                let values = lowered_args
                    .iter()
                    .map(|(value, _)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = lowered_args
                    .iter()
                    .map(|(_, ty)| mlir_type(*ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                let value_suffix = if values.is_empty() {
                    String::new()
                } else {
                    format!(", {values}")
                };
                let type_suffix = if types.is_empty() {
                    String::new()
                } else {
                    format!(", {types}")
                };
                let return_type = method_definition
                    .as_ref()
                    .map(|definition| definition.return_type)
                    .unwrap_or(ValueType::Any);
                if return_type == ValueType::Unit {
                    writeln!(self.output, "    llvm.call @{symbol}({object}{value_suffix}) : (!llvm.ptr{type_suffix}) -> ()").unwrap();
                    (String::new(), ValueType::Unit)
                } else {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @{symbol}({object}{value_suffix}) : (!llvm.ptr{type_suffix}) -> {}", mlir_type(return_type)).unwrap();
                    if return_type == ValueType::Any {
                        if let Some(class_id) = canonical_class {
                            let returned_id = self
                                .method_return_classes
                                .get(&(class_id, method.clone()))
                                .copied()
                                .unwrap_or(class_id);
                            self.object_class_ids.insert(result.clone(), returned_id);
                            if let Some(returned) = self
                                .classes
                                .iter()
                                .find(|candidate| candidate.id == returned_id)
                            {
                                self.object_classes
                                    .insert(result.clone(), returned.name.clone());
                            }
                        } else {
                            self.object_classes.insert(result.clone(), class);
                        }
                    }
                    (result, return_type)
                }
            }
            Expression::Task { value, placement } => {
                if let Expression::Send { value, channel } = value.kind() {
                    let lowered = self.lower_expression(value);
                    let value_type = lowered.1;
                    let value = self.box_value(lowered);
                    let (channel, _) = self.lower_expression(channel);
                    self.channel_types.insert(channel.clone(), value_type);
                    let result = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {result} = llvm.call @__sev_channel_send_ptr_async({value}, {channel}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr"
                    )
                    .unwrap();
                    self.task_results.insert(result.clone(), ValueType::Unit);
                    return (result, ValueType::Any);
                }
                if let Expression::MethodCall {
                    object,
                    method,
                    args,
                } = value.kind()
                {
                    let (object, _) = self.lower_expression(object);
                    let Some(class) = self.object_classes.get(&object).cloned() else {
                        let result = self.fresh_value();
                        writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                        return (result, ValueType::Any);
                    };
                    let definition = self
                        .classes
                        .iter()
                        .find(|candidate| candidate.name == class)
                        .and_then(|definition| {
                            definition
                                .methods
                                .iter()
                                .find(|candidate| candidate.name == *method)
                        })
                        .cloned();
                    let Some(definition) = definition else {
                        let result = self.fresh_value();
                        writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                        return (result, ValueType::Any);
                    };
                    let lowered_args = args
                        .iter()
                        .map(|argument| self.lower_expression(argument))
                        .collect::<Vec<_>>()
                        .into_iter()
                        .enumerate()
                        .map(|(index, argument)| {
                            let expected = definition.params[index].ty;
                            if argument.1 == ValueType::Any && expected != ValueType::Any {
                                self.unbox_value(argument, expected)
                            } else if expected == ValueType::Any && argument.1 != ValueType::Any {
                                (self.box_value(argument), ValueType::Any)
                            } else {
                                argument
                            }
                        })
                        .collect::<Vec<_>>();
                    let values = lowered_args
                        .iter()
                        .map(|(value, _)| value.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let types = lowered_args
                        .iter()
                        .map(|(_, ty)| mlir_type(*ty))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let value_suffix = if values.is_empty() {
                        String::new()
                    } else {
                        format!(", {values}")
                    };
                    let type_suffix = if types.is_empty() {
                        String::new()
                    } else {
                        format!(", {types}")
                    };
                    let result = self.fresh_value();
                    let mut attributes = Vec::new();
                    match placement {
                        TaskPlacement::Default => {}
                        TaskPlacement::Local => {
                            attributes.push("severian_distribution = \"local\"")
                        }
                        TaskPlacement::Gpu => {
                            attributes.push("severian_parallel = \"gpu\"");
                            attributes.push("severian_device_fallback = \"cpu\"");
                        }
                        TaskPlacement::Simd => {
                            attributes.push("severian_parallel = \"simd\"");
                            attributes.push("severian_device_fallback = \"cpu\"");
                        }
                        TaskPlacement::Simt => {
                            attributes.push("severian_parallel = \"simt\"");
                            attributes.push("severian_device_fallback = \"cpu\"");
                        }
                    }
                    let placement_attribute = if attributes.is_empty() {
                        String::new()
                    } else {
                        format!(" {{{}}}", attributes.join(", "))
                    };
                    writeln!(self.output, "    {result} = llvm.call @__sev_task_spawn_{class}_{method}({object}{value_suffix}){placement_attribute} : (!llvm.ptr{type_suffix}) -> !llvm.ptr").unwrap();
                    self.task_results
                        .insert(result.clone(), definition.return_type);
                    return (result, ValueType::Any);
                }
                let Expression::Call { target, args } = value.kind() else {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                };
                let linked_function = &target.name;
                let args = args
                    .iter()
                    .map(|argument| self.lower_expression(argument))
                    .collect::<Vec<_>>();
                let args = self.coerce_resolved_call_arguments(target, args);
                let values = args
                    .iter()
                    .map(|(value, _)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = args
                    .iter()
                    .map(|(_, ty)| mlir_type(*ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                let return_type = target
                    .signature
                    .as_ref()
                    .map(|signature| signature.returns)
                    .or_else(|| self.function_returns.get(&target.id).copied())
                    .unwrap_or(ValueType::Any);
                let result = self.fresh_value();
                let mut attributes = Vec::new();
                match placement {
                    TaskPlacement::Default => {}
                    TaskPlacement::Local => attributes.push("severian_distribution = \"local\""),
                    TaskPlacement::Gpu => {
                        attributes.push("severian_parallel = \"gpu\"");
                        attributes.push("severian_device_fallback = \"cpu\"");
                    }
                    TaskPlacement::Simd => {
                        attributes.push("severian_parallel = \"simd\"");
                        attributes.push("severian_device_fallback = \"cpu\"");
                    }
                    TaskPlacement::Simt => {
                        attributes.push("severian_parallel = \"simt\"");
                        attributes.push("severian_device_fallback = \"cpu\"");
                    }
                }
                let placement_attribute = if attributes.is_empty() {
                    String::new()
                } else {
                    format!(" {{{}}}", attributes.join(", "))
                };
                let task_symbol = mangle_symbol_component(linked_function);
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_task_spawn_{task_symbol}({values}){placement_attribute} : ({types}) -> !llvm.ptr"
                )
                .unwrap();
                self.task_results.insert(result.clone(), return_type);
                (result, ValueType::Any)
            }
            Expression::Await(value) => {
                if let Expression::Tuple(tasks) = value.kind() {
                    for task in tasks {
                        self.lower_expression(&Expression::Await(Box::new(task.clone())));
                    }
                    return (String::new(), ValueType::Unit);
                }
                let (task, awaited_type) = self.lower_expression(value);
                let return_type = self.task_results.remove(&task);
                if awaited_type == ValueType::Channel {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_channel_receive_ptr({task}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                }
                if let Some(channel_type) = self.channel_types.get(&task).copied() {
                    let result = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {result} = llvm.call @__sev_channel_receive_ptr({task}) : (!llvm.ptr) -> !llvm.ptr"
                    )
                    .unwrap();
                    return self.unbox_value((result, ValueType::Any), channel_type);
                }
                let Some(return_type) = return_type else {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                };
                if return_type == ValueType::Unit {
                    writeln!(
                        self.output,
                        "    llvm.call @__sev_task_await_unit({task}) : (!llvm.ptr) -> ()"
                    )
                    .unwrap();
                    return (String::new(), ValueType::Unit);
                }
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_task_await_{}({task}) : (!llvm.ptr) -> {}",
                    task_type_suffix(return_type),
                    mlir_type(return_type)
                )
                .unwrap();
                (result, return_type)
            }
            Expression::Channel(capacity) => {
                let (capacity, _) = self.lower_expression(capacity);
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_channel_create({capacity}) : (i64) -> !llvm.ptr"
                )
                .unwrap();
                (result, ValueType::Channel)
            }
            Expression::Send { value, channel } => {
                let lowered = self.lower_expression(value);
                let value_type = lowered.1;
                let value = self.box_value(lowered);
                let (channel, _) = self.lower_expression(channel);
                self.channel_types.insert(channel.clone(), value_type);
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_channel_send_ptr_async({value}, {channel}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr"
                )
                .unwrap();
                self.task_results.insert(result.clone(), ValueType::Unit);
                (result, ValueType::Any)
            }
            Expression::Variant {
                type_id,
                name,
                fields,
                ..
            } => {
                let tag = self.string_address(name);
                let field = if fields.len() > 1 {
                    let kind = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {kind} = llvm.mlir.constant(1 : i64) : i64"
                    )
                    .unwrap();
                    let tuple = self.fresh_value();
                    writeln!(self.output, "    {tuple} = llvm.call @__sev_collection_new({kind}) : (i64) -> !llvm.ptr").unwrap();
                    for field in fields {
                        let field = self.lower_expression(field);
                        let field = self.box_value(field);
                        writeln!(self.output, "    llvm.call @__sev_collection_push({tuple}, {field}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
                    }
                    tuple
                } else if let Some(field) = fields.first() {
                    let field = self.lower_expression(field);
                    self.box_value(field)
                } else {
                    let empty = self.fresh_value();
                    writeln!(self.output, "    {empty} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    empty
                };
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_variant_new({tag}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                let ty = match type_id {
                    Some(id) if *id == TypeDefinitionId::from_name("Result") => ValueType::Result,
                    Some(id) if *id == TypeDefinitionId::from_name("Option") => ValueType::Option,
                    _ => ValueType::Any,
                };
                (result, ty)
            }
            Expression::Index { object, index } => {
                let (object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    let index = self.lower_expression(index);
                    let key = self.box_value(index);
                    self.emit_runtime_site();
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_value_get({object}, {key}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                }
                let index = self.lower_expression(index);
                let result = self.fresh_value();
                self.emit_runtime_site();
                if object_type == ValueType::Map {
                    let key = self.box_value(index);
                    writeln!(self.output, "    {result} = llvm.call @__sev_map_get({object}, {key}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                } else if object_type == ValueType::String {
                    let index = self.unbox_value(index, ValueType::Int).0;
                    writeln!(self.output, "    {result} = llvm.call @__sev_string_char_at({object}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                    return (result, ValueType::String);
                } else {
                    let index = self.unbox_value(index, ValueType::Int).0;
                    writeln!(self.output, "    {result} = llvm.call @__sev_collection_get({object}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                }
                (result, ValueType::Any)
            }
            Expression::Slice {
                object,
                start,
                end,
                step,
            } => self.lower_slice_expression(object, start, end, step, None),
            Expression::Format {
                template,
                args,
                arg_types,
            } => {
                let native_template = native_format_template(template, arg_types);
                let index = self
                    .strings
                    .iter()
                    .position(|candidate| candidate == &native_template)
                    .unwrap();
                let format = self.fresh_value();
                writeln!(
                    self.output,
                    "    {format} = llvm.mlir.addressof @__sev_str_{index} : !llvm.ptr"
                )
                .unwrap();
                let mut lowered_args = Vec::new();
                for argument in args {
                    let (mut value, ty) = self.lower_expression(argument);
                    let native_type = if ty == ValueType::Bool {
                        let promoted = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {promoted} = llvm.zext {value} : i1 to i32"
                        )
                        .unwrap();
                        value = promoted;
                        "i32"
                    } else {
                        mlir_type(ty)
                    };
                    lowered_args.push((value, native_type));
                }
                let values = lowered_args
                    .iter()
                    .map(|(value, _)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = lowered_args
                    .iter()
                    .map(|(_, ty)| *ty)
                    .collect::<Vec<_>>()
                    .join(", ");
                let suffix_values = if values.is_empty() {
                    String::new()
                } else {
                    format!(", {values}")
                };
                let suffix_types = if types.is_empty() {
                    String::new()
                } else {
                    format!(", {types}")
                };
                let empty = self.fresh_value();
                writeln!(self.output, "    {empty} = llvm.mlir.zero : !llvm.ptr").unwrap();
                let zero = self.fresh_value();
                writeln!(
                    self.output,
                    "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                let required = self.fresh_value();
                writeln!(
                    self.output,
                    "    {required} = llvm.call @snprintf({empty}, {zero}, {format}{suffix_values}) vararg(!llvm.func<i32 (!llvm.ptr, i64, !llvm.ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr{suffix_types}) -> i32"
                )
                .unwrap();
                let required_wide = self.fresh_value();
                writeln!(
                    self.output,
                    "    {required_wide} = llvm.zext {required} : i32 to i64"
                )
                .unwrap();
                let one = self.fresh_value();
                writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
                let capacity = self.fresh_value();
                writeln!(
                    self.output,
                    "    {capacity} = llvm.add {required_wide}, {one} : i64"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @malloc({capacity}) : (i64) -> !llvm.ptr"
                )
                .unwrap();
                let status = self.fresh_value();
                writeln!(
                    self.output,
                    "    {status} = llvm.call @snprintf({result}, {capacity}, {format}{suffix_values}) vararg(!llvm.func<i32 (!llvm.ptr, i64, !llvm.ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr{suffix_types}) -> i32"
                )
                .unwrap();
                (result, ValueType::String)
            }
            Expression::Unary { op, expression } => {
                let (value, ty) = self.lower_expression(expression);
                match op {
                    UnaryOp::Negate if ty == ValueType::Float => {
                        let zero = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {zero} = llvm.mlir.constant(0.0 : f64) : f64"
                        )
                        .unwrap();
                        self.lower_binary_values((zero, ty), BinaryOp::Sub, (value, ty))
                    }
                    UnaryOp::Negate if ty == ValueType::Any => {
                        let zero = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                        )
                        .unwrap();
                        let zero = self.box_value((zero, ValueType::Int));
                        self.lower_binary_values((zero, ValueType::Any), BinaryOp::Sub, (value, ty))
                    }
                    UnaryOp::Negate => {
                        let zero = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                        )
                        .unwrap();
                        self.lower_binary_values((zero, ty), BinaryOp::Sub, (value, ty))
                    }
                    UnaryOp::Not => {
                        let (value, _) = self.unbox_value((value, ty), ValueType::Bool);
                        let one = self.fresh_value();
                        writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1")
                            .unwrap();
                        let result = self.fresh_value();
                        writeln!(self.output, "    {result} = llvm.xor {value}, {one} : i1")
                            .unwrap();
                        (result, ValueType::Bool)
                    }
                }
            }
            Expression::Call { target, args } => {
                let function = &target.name;
                let linked_function = function.as_str();
                let args = args
                    .iter()
                    .map(|argument| self.lower_expression(argument))
                    .collect::<Vec<_>>();
                let compiler_intrinsic = matches!(
                    function.as_str(),
                    "int"
                        | "float"
                        | "string"
                        | "len"
                        | "size"
                        | "bytes"
                        | "bits"
                        | "capacity"
                        | "range"
                        | "abs"
                        | "min"
                        | "max"
                        | "divmod"
                        | "enumerate"
                        | "zip"
                        | "any"
                        | "all"
                );
                let args = if compiler_intrinsic {
                    args
                } else {
                    self.coerce_resolved_call_arguments(target, args)
                };
                if function == "int" {
                    let argument = args.first().cloned().unwrap();
                    if argument.1 == ValueType::Int {
                        return argument;
                    }
                    self.emit_runtime_site();
                    let value = self.box_value(argument);
                    let result = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {result} = llvm.call @__sev_value_int({value}) : (!llvm.ptr) -> i64"
                    )
                    .unwrap();
                    return (result, ValueType::Int);
                }
                if function == "float" {
                    let (value, ty) = args.first().cloned().unwrap();
                    return match ty {
                        ValueType::Float => (value, ValueType::Float),
                        ValueType::Int => {
                            let result = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {result} = llvm.sitofp {value} : i64 to f64"
                            )
                            .unwrap();
                            (result, ValueType::Float)
                        }
                        ValueType::String => {
                            self.emit_runtime_site();
                            let value = self.box_value((value, ty));
                            let result = self.fresh_value();
                            writeln!(self.output, "    {result} = llvm.call @__sev_value_float({value}) : (!llvm.ptr) -> f64").unwrap();
                            (result, ValueType::Float)
                        }
                        ValueType::Any => {
                            self.emit_runtime_site();
                            let result = self.fresh_value();
                            writeln!(self.output, "    {result} = llvm.call @__sev_value_float({value}) : (!llvm.ptr) -> f64").unwrap();
                            (result, ValueType::Float)
                        }
                        _ => {
                            self.emit_runtime_site();
                            let value = self.box_value((value, ty));
                            let result = self.fresh_value();
                            writeln!(self.output, "    {result} = llvm.call @__sev_value_float({value}) : (!llvm.ptr) -> f64").unwrap();
                            (result, ValueType::Float)
                        }
                    };
                }
                if function == "string" {
                    let argument = args.first().cloned().unwrap();
                    if argument.1 == ValueType::String {
                        return argument;
                    }
                    self.emit_runtime_site();
                    let value = self.box_value(argument);
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_value_string({value}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    return (result, ValueType::String);
                }
                if function == "len" || function == "size" {
                    let (value, ty) = args.first().cloned().unwrap();
                    if let ValueType::Tensor(tensor) = ty {
                        if let Some(elements) = static_tensor_elements(tensor) {
                            let result = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {result} = llvm.mlir.constant({elements} : i64) : i64"
                            )
                            .unwrap();
                            return (result, ValueType::Int);
                        }
                    }
                    if matches!(ty, ValueType::Tensor(_) | ValueType::TensorAny) {
                        let result = self.fresh_value();
                        writeln!(self.output, "    {result} = llvm.call @__sev_tensor_size({value}) : (!llvm.ptr) -> i64").unwrap();
                        return (result, ValueType::Int);
                    }
                    if ty == ValueType::String {
                        let result = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {result} = llvm.call @__sev_string_length({value}) : (!llvm.ptr) -> i64"
                        )
                        .unwrap();
                        return (result, ValueType::Int);
                    }
                    if ty == ValueType::Any {
                        let result = self.fresh_value();
                        writeln!(self.output, "    {result} = llvm.call @__sev_value_size({value}) : (!llvm.ptr) -> i64").unwrap();
                        return (result, ValueType::Int);
                    }
                    if ty == ValueType::Map {
                        let result = self.fresh_value();
                        writeln!(self.output, "    {result} = llvm.call @__sev_map_size({value}) : (!llvm.ptr) -> i64").unwrap();
                        return (result, ValueType::Int);
                    }
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_collection_size({value}) : (!llvm.ptr) -> i64").unwrap();
                    return (result, ValueType::Int);
                }
                if matches!(function.as_str(), "bytes" | "bits" | "capacity") {
                    let (value, ty) = args.first().cloned().unwrap();
                    let result = self.fresh_value();
                    if let ValueType::Tensor(tensor) = ty {
                        if let Some(elements) = static_tensor_elements(tensor) {
                            let amount = if function == "capacity" {
                                elements
                            } else {
                                let bytes = elements * tensor_element_bytes(tensor.element);
                                if function == "bits" {
                                    bytes * 8
                                } else {
                                    bytes
                                }
                            };
                            writeln!(
                                self.output,
                                "    {result} = llvm.mlir.constant({amount} : i64) : i64"
                            )
                            .unwrap();
                            return (result, ValueType::Int);
                        }
                    }
                    if function == "capacity" {
                        let runtime = match ty {
                            ValueType::Map => "__sev_map_capacity",
                            ValueType::Tensor(_) | ValueType::TensorAny => "__sev_tensor_capacity",
                            _ => "__sev_value_capacity",
                        };
                        let value = if matches!(
                            ty,
                            ValueType::Any
                                | ValueType::Map
                                | ValueType::Tensor(_)
                                | ValueType::TensorAny
                        ) {
                            value
                        } else {
                            self.box_value((value, ty))
                        };
                        writeln!(
                            self.output,
                            "    {result} = llvm.call @{runtime}({value}) : (!llvm.ptr) -> i64"
                        )
                        .unwrap();
                    } else {
                        let runtime = match ty {
                            ValueType::Map => "__sev_map_bytes",
                            ValueType::Tensor(_) | ValueType::TensorAny => "__sev_tensor_bytes",
                            _ => "__sev_value_bytes",
                        };
                        let value = if matches!(
                            ty,
                            ValueType::Any
                                | ValueType::Map
                                | ValueType::Tensor(_)
                                | ValueType::TensorAny
                        ) {
                            value
                        } else {
                            self.box_value((value, ty))
                        };
                        writeln!(
                            self.output,
                            "    {result} = llvm.call @{runtime}({value}) : (!llvm.ptr) -> i64"
                        )
                        .unwrap();
                        if function == "bits" {
                            let eight = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {eight} = llvm.mlir.constant(8 : i64) : i64"
                            )
                            .unwrap();
                            let bits = self.fresh_value();
                            writeln!(self.output, "    {bits} = llvm.mul {result}, {eight} : i64")
                                .unwrap();
                            return (bits, ValueType::Int);
                        }
                    }
                    return (result, ValueType::Int);
                }
                if function == "range" {
                    let mut integer_args = Vec::with_capacity(args.len());
                    for argument in args.iter().cloned() {
                        integer_args.push(self.unbox_value(argument, ValueType::Int));
                    }
                    let zero = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                    )
                    .unwrap();
                    let one = self.fresh_value();
                    writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
                    let (start, end, step) = match integer_args.as_slice() {
                        [end] => (zero, end.0.clone(), one),
                        [start, end] => (start.0.clone(), end.0.clone(), one),
                        [start, end, step] => (start.0.clone(), end.0.clone(), step.0.clone()),
                        _ => unreachable!("semantic analysis validates range arity"),
                    };
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_range({start}, {end}, {step}) : (i64, i64, i64) -> !llvm.ptr").unwrap();
                    return (result, ValueType::List);
                }
                if matches!(function.as_str(), "abs" | "min" | "max" | "divmod") {
                    let boxed = args
                        .iter()
                        .cloned()
                        .map(|value| self.box_value(value))
                        .collect::<Vec<_>>();
                    let result = self.fresh_value();
                    let runtime = format!("__sev_{function}");
                    if boxed.len() == 1 {
                        writeln!(
                            self.output,
                            "    {result} = llvm.call @{runtime}({}) : (!llvm.ptr) -> !llvm.ptr",
                            boxed[0]
                        )
                        .unwrap();
                    } else {
                        writeln!(self.output, "    {result} = llvm.call @{runtime}({}, {}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr", boxed[0], boxed[1]).unwrap();
                    }
                    if function == "divmod" {
                        return (result, ValueType::Tuple);
                    }
                    return (result, ValueType::Any);
                }
                if function == "enumerate" {
                    let (mut value, ty) = args.first().cloned().unwrap();
                    if ty == ValueType::Any {
                        value = self.unbox_value((value, ty), ValueType::List).0;
                    }
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_collection_enumerate({value}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    return (result, ValueType::List);
                }
                if function == "zip" {
                    let values = args
                        .iter()
                        .cloned()
                        .map(|(value, ty)| {
                            if ty == ValueType::Any {
                                self.unbox_value((value, ty), ValueType::List).0
                            } else {
                                value
                            }
                        })
                        .collect::<Vec<_>>();
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_collection_zip({}, {}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr", values[0], values[1]).unwrap();
                    return (result, ValueType::List);
                }
                if function == "any" || function == "all" {
                    let (mut value, ty) = args.first().cloned().unwrap();
                    if ty == ValueType::Any {
                        value = self.unbox_value((value, ty), ValueType::List).0;
                    }
                    let result = self.fresh_value();
                    let runtime = if function == "any" {
                        "__sev_collection_any"
                    } else {
                        "__sev_collection_all"
                    };
                    writeln!(
                        self.output,
                        "    {result} = llvm.call @{runtime}({value}) : (!llvm.ptr) -> i1"
                    )
                    .unwrap();
                    return (result, ValueType::Bool);
                }
                if target.native_symbol.is_none() && function.ends_with(".zero") && args.is_empty()
                {
                    let zero = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                    )
                    .unwrap();
                    return (self.box_value((zero, ValueType::Int)), ValueType::Any);
                }
                if target.native_symbol.is_none()
                    && function.ends_with(".add")
                    && args.len() == 2
                    && args[0].1 != ValueType::Set
                {
                    let left = self.box_value(args[0].clone());
                    let right = self.box_value(args[1].clone());
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_value_add({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                }
                if function == "probability.probability" {
                    let result = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {result} = llvm.mlir.constant(0.5 : f64) : f64"
                    )
                    .unwrap();
                    return (result, ValueType::Float);
                }
                if function == "regex.matches" {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_regex_matches({}, {}) : (!llvm.ptr, !llvm.ptr) -> i1", args[0].0, args[1].0).unwrap();
                    return (result, ValueType::Bool);
                }
                if !self.function_returns.contains_key(&target.id) {
                    let builtin = match function.as_str() {
                        "read" => Some(("__sev_builtin_read", ValueType::Result)),
                        "http.get" => Some(("__sev_builtin_http_get", ValueType::Result)),
                        "int.parse" => Some(("__sev_builtin_int_parse", ValueType::Result)),
                        "float.parse" => Some(("__sev_builtin_float_parse", ValueType::Result)),
                        "file.write" => Some(("__sev_builtin_file_write", ValueType::Result)),
                        _ => None,
                    };
                    if let Some((symbol, return_type)) = builtin {
                        let values = args
                            .iter()
                            .map(|(value, _)| value.as_str())
                            .collect::<Vec<_>>()
                            .join(", ");
                        let types = args
                            .iter()
                            .map(|(_, ty)| mlir_type(*ty))
                            .collect::<Vec<_>>()
                            .join(", ");
                        let result = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {result} = llvm.call @{symbol}({values}) : ({types}) -> !llvm.ptr"
                        )
                        .unwrap();
                        return (result, return_type);
                    }
                }
                let values = args
                    .iter()
                    .map(|(value, _)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = args
                    .iter()
                    .map(|(_, ty)| mlir_type(*ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                let return_type = match function.as_str() {
                    "sqrt" | "float" => ValueType::Float,
                    "int" | "len" | "size" | "bytes" | "bits" | "capacity" => ValueType::Int,
                    _ => target
                        .signature
                        .as_ref()
                        .map(|signature| signature.returns)
                        .or_else(|| self.function_returns.get(&target.id).copied())
                        .unwrap_or(ValueType::Any),
                };
                let symbol = if let Some(symbol) = &target.native_symbol {
                    symbol.clone()
                } else if function == "sqrt" {
                    "llvm.sqrt.f64".to_owned()
                } else {
                    self.native_symbols
                        .get(&target.id)
                        .cloned()
                        .unwrap_or_else(|| {
                            if self.function_returns.contains_key(&target.id) {
                                source_function_symbol(linked_function)
                            } else {
                                linked_function.to_owned()
                            }
                        })
                };
                if linked_function == "main" && return_type == ValueType::Unit {
                    let ignored = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {ignored} = llvm.call @{symbol}({values}) : ({types}) -> i32"
                    )
                    .unwrap();
                    return (String::new(), ValueType::Unit);
                }
                if return_type == ValueType::Unit {
                    writeln!(
                        self.output,
                        "    llvm.call @{symbol}({values}) : ({types}) -> ()"
                    )
                    .unwrap();
                    return (String::new(), ValueType::Unit);
                }
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @{symbol}({values}) : ({types}) -> {}",
                    mlir_type(return_type)
                )
                .unwrap();
                if let Some(class) = self.function_return_classes.get(&target.id) {
                    self.object_class_ids.insert(result.clone(), *class);
                    if let Some(definition) =
                        self.classes.iter().find(|candidate| candidate.id == *class)
                    {
                        self.object_classes
                            .insert(result.clone(), definition.name.clone());
                    }
                }
                (result, return_type)
            }
            Expression::CallValue {
                callee,
                args,
                return_type,
            } => {
                let (callee, _) = self.lower_expression(callee);
                let args = args
                    .iter()
                    .map(|argument| self.lower_expression(argument))
                    .collect::<Vec<_>>();
                let values = args
                    .into_iter()
                    .map(|value| self.box_value(value))
                    .collect::<Vec<_>>()
                    .join(", ");
                let function = self.fresh_value();
                writeln!(self.output, "    {function} = llvm.call @__sev_closure_function({callee}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                let environment = self.fresh_value();
                writeln!(self.output, "    {environment} = llvm.call @__sev_closure_environment({callee}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                let value_suffix = if values.is_empty() {
                    String::new()
                } else {
                    format!(", {values}")
                };
                let type_suffix = if values.is_empty() {
                    String::new()
                } else {
                    format!(
                        ", {}",
                        std::iter::repeat_n("!llvm.ptr", values.split(", ").count())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                if *return_type == ValueType::Unit {
                    writeln!(
                        self.output,
                        "    %ignored_closure_result_{} = llvm.call {function}({environment}{value_suffix}) : !llvm.ptr, (!llvm.ptr{type_suffix}) -> !llvm.ptr",
                        self.next_value
                    )
                    .unwrap();
                    self.next_value += 1;
                    return (String::new(), ValueType::Unit);
                }
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call {function}({environment}{value_suffix}) : !llvm.ptr, (!llvm.ptr{type_suffix}) -> !llvm.ptr"
                )
                .unwrap();
                if *return_type == ValueType::Any {
                    (result, ValueType::Any)
                } else {
                    self.unbox_value((result, ValueType::Any), *return_type)
                }
            }
            Expression::Binary { left, op, right } => {
                if matches!(op, BinaryOp::And | BinaryOp::Or) {
                    return self.lower_short_circuit_chain(left, *op, right);
                }
                let divisor_site = right.hir_id();
                let left = self.lower_expression(left);
                let right = self.lower_expression(right);
                if matches!(op, BinaryOp::Div | BinaryOp::Mod) {
                    self.emit_runtime_site_for(divisor_site);
                }
                if *op == BinaryOp::Power {
                    return self.lower_power_values(left, right);
                }
                self.lower_binary_values(left, *op, right)
            }
        }
    }

    fn lower_field_construction(
        &mut self,
        class: &str,
        source: Option<&str>,
        explicit: &HashMap<&str, &Expression>,
        json_document: bool,
        validate: bool,
    ) -> (String, ValueType) {
        let class_name = self.string_address(class);
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @__sev_object_new({class_name}) : (!llvm.ptr) -> !llvm.ptr"
        )
        .unwrap();
        let definition = self
            .classes
            .iter()
            .find(|candidate| candidate.name == class)
            .cloned();
        let source_definition = source
            .and_then(|value| self.object_classes.get(value))
            .and_then(|source_class| {
                self.classes
                    .iter()
                    .find(|candidate| candidate.name == *source_class)
            })
            .cloned();
        let source_fields = source_definition
            .as_ref()
            .map(|definition| definition.fields.iter().cloned().collect::<HashSet<_>>());

        if let Some(definition) = &definition {
            for field in &definition.fields {
                let field_name = self.string_address(field);
                writeln!(self.output, "    llvm.call @__sev_object_declare({result}, {field_name}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
            }
            for (index, field) in definition.fields.iter().enumerate() {
                let value = if let Some(value) = explicit.get(field.as_str()) {
                    let lowered = self.lower_expression(value);
                    Some(self.box_value(lowered))
                } else if let Some(source) = source.filter(|_| {
                    json_document
                        || source_fields
                            .as_ref()
                            .map_or(true, |fields| fields.contains(field))
                }) {
                    let field_name = self.string_address(field);
                    let mut copied = self.fresh_value();
                    let getter = if json_document {
                        "__sev_json_object_get"
                    } else {
                        "__sev_object_get"
                    };
                    writeln!(self.output, "    {copied} = llvm.call @{getter}({source}, {field_name}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                    if !json_document {
                        let source_field_class = source_definition.as_ref().and_then(|source| {
                            source
                                .fields
                                .iter()
                                .position(|candidate| candidate == field)
                                .and_then(|index| source.field_classes[index].as_ref())
                        });
                        let target_field_class = definition.field_classes[index].as_ref();
                        if let (Some(source_class), Some(target_class)) =
                            (source_field_class, target_field_class)
                        {
                            if source_class != target_class {
                                self.object_classes
                                    .insert(copied.clone(), source_class.clone());
                                if let Some(source_definition) = self
                                    .classes
                                    .iter()
                                    .find(|candidate| candidate.name == *source_class)
                                {
                                    self.object_class_ids
                                        .insert(copied.clone(), source_definition.id);
                                }
                                let nested = HashMap::new();
                                copied = self
                                    .lower_field_construction(
                                        target_class,
                                        Some(&copied),
                                        &nested,
                                        false,
                                        true,
                                    )
                                    .0;
                            }
                        }
                    }
                    Some(copied)
                } else {
                    definition.field_defaults[index].as_ref().map(|default| {
                        let lowered = self.lower_expression(default);
                        self.box_value(lowered)
                    })
                };
                let Some(value) = value else { continue };
                let field_name = self.string_address(field);
                writeln!(self.output, "    llvm.call @__sev_object_set({result}, {field_name}, {value}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
            }
        }
        self.object_classes.insert(result.clone(), class.to_owned());
        if let Some(definition) = &definition {
            self.object_class_ids.insert(result.clone(), definition.id);
        }
        if validate {
            self.validate_object(&result, class);
        }
        (result, ValueType::Any)
    }

    fn lower_known_class_method_call(
        &mut self,
        object: &Expression,
        method: &str,
        args: &[Expression],
    ) -> (String, ValueType) {
        let (object, _) = self.lower_expression(object);
        let canonical_class = self.object_class_ids.get(&object).copied();
        let class = canonical_class
            .and_then(|id| self.classes.iter().find(|candidate| candidate.id == id))
            .map(|class| class.name.clone())
            .or_else(|| self.object_classes.get(&object).cloned())
            .expect("known class method receiver retains its class identity");
        let definition = self
            .classes
            .iter()
            .find(|candidate| candidate.name == class)
            .and_then(|class| {
                class
                    .methods
                    .iter()
                    .find(|candidate| candidate.name == method)
            })
            .cloned()
            .expect("known class method retains its definition");
        let mut lowered_args = args
            .iter()
            .map(|argument| self.lower_expression(argument))
            .collect::<Vec<_>>();
        lowered_args.extend(definition.params.iter().skip(args.len()).map(|parameter| {
            self.lower_expression(
                parameter
                    .default
                    .as_ref()
                    .expect("omitted class method arguments have defaults"),
            )
        }));
        let lowered_args = lowered_args
            .into_iter()
            .enumerate()
            .map(|(index, argument)| {
                let expected = definition.params[index].ty;
                if argument.1 == ValueType::Any && expected != ValueType::Any {
                    self.unbox_value(argument, expected)
                } else if expected == ValueType::Any && argument.1 != ValueType::Any {
                    (self.box_value(argument), ValueType::Any)
                } else {
                    argument
                }
            })
            .collect::<Vec<_>>();
        let values = lowered_args
            .iter()
            .map(|(value, _)| value.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let types = lowered_args
            .iter()
            .map(|(_, ty)| mlir_type(*ty))
            .collect::<Vec<_>>()
            .join(", ");
        let value_suffix = if values.is_empty() {
            String::new()
        } else {
            format!(", {values}")
        };
        let type_suffix = if types.is_empty() {
            String::new()
        } else {
            format!(", {types}")
        };
        let symbol = class_function_symbol(&class, method);
        if definition.return_type == ValueType::Unit {
            writeln!(
                self.output,
                "    llvm.call @{symbol}({object}{value_suffix}) : (!llvm.ptr{type_suffix}) -> ()"
            )
            .unwrap();
            return (String::new(), ValueType::Unit);
        }
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @{symbol}({object}{value_suffix}) : (!llvm.ptr{type_suffix}) -> {}",
            mlir_type(definition.return_type)
        )
        .unwrap();
        if definition.return_type == ValueType::Any {
            if let Some(class_id) = canonical_class {
                let returned_id = self
                    .method_return_classes
                    .get(&(class_id, method.to_owned()))
                    .copied()
                    .unwrap_or(class_id);
                self.object_class_ids.insert(result.clone(), returned_id);
                if let Some(returned) = self
                    .classes
                    .iter()
                    .find(|candidate| candidate.id == returned_id)
                {
                    self.object_classes
                        .insert(result.clone(), returned.name.clone());
                }
            } else {
                self.object_classes.insert(result.clone(), class);
            }
        }
        (result, definition.return_type)
    }

    pub(super) fn validate_object(&mut self, object: &str, class: &str) {
        let Some(definition) = self
            .classes
            .iter()
            .find(|candidate| candidate.name == class)
            .cloned()
        else {
            return;
        };
        if definition.field_constraints.is_empty() {
            return;
        }

        let previous_object = self.field_object.replace(object.to_owned());
        let previous_names = std::mem::replace(
            &mut self.field_names,
            definition.fields.iter().cloned().collect(),
        );
        let previous_types = std::mem::replace(
            &mut self.field_types,
            definition
                .fields
                .iter()
                .cloned()
                .zip(definition.field_types.iter().copied())
                .collect(),
        );
        let previous_classes = std::mem::replace(
            &mut self.field_classes,
            definition
                .fields
                .iter()
                .cloned()
                .zip(definition.field_classes.iter().cloned())
                .filter_map(|(field, class)| class.map(|class| (field, class)))
                .collect(),
        );
        for constraint in &definition.field_constraints {
            let lowered = self.lower_expression(constraint);
            let (condition, _) = self.unbox_value(lowered, ValueType::Bool);
            let passed = self.fresh_block();
            let failed = self.fresh_block();
            writeln!(
                self.output,
                "    llvm.cond_br {condition}, ^bb{passed}, ^bb{failed}"
            )
            .unwrap();
            writeln!(self.output, "  ^bb{failed}:").unwrap();
            writeln!(self.output, "    llvm.call @abort() : () -> ()").unwrap();
            writeln!(self.output, "    llvm.unreachable").unwrap();
            writeln!(self.output, "  ^bb{passed}:").unwrap();
            self.terminated = false;
        }
        self.field_object = previous_object;
        self.field_names = previous_names;
        self.field_types = previous_types;
        self.field_classes = previous_classes;
    }
}
