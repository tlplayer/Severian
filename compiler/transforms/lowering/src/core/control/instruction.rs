use super::*;

impl LowerContext<'_> {
    pub(in crate::core) fn lower_instructions(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            if self.terminated {
                break;
            }
            match instruction {
                Instruction::Let { name, value } => {
                    let lowered = self.lower_expression(value);
                    self.variables.insert(name.id, lowered);
                }
                Instruction::TryLet {
                    name,
                    value,
                    payload_type,
                    receiver,
                } => {
                    let (result, _) = self.lower_expression(value);
                    let ok_tag = self.string_address("ok");
                    let succeeded = self.fresh_value();
                    writeln!(self.output, "    {succeeded} = llvm.call @__sev_variant_is({result}, {ok_tag}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
                    let success_block = self.fresh_block();
                    let failure_block = self.fresh_block();
                    writeln!(
                        self.output,
                        "    llvm.cond_br {succeeded}, ^bb{success_block}, ^bb{failure_block}"
                    )
                    .unwrap();
                    writeln!(self.output, "  ^bb{failure_block}:").unwrap();
                    if self.is_main {
                        let failure = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {failure} = llvm.mlir.constant(1 : i32) : i32"
                        )
                        .unwrap();
                        writeln!(self.output, "    llvm.return {failure} : i32").unwrap();
                    } else if self.declared_return == ValueType::Result {
                        writeln!(self.output, "    llvm.return {result} : !llvm.ptr").unwrap();
                    } else {
                        writeln!(self.output, "    llvm.call @abort() : () -> ()").unwrap();
                        writeln!(self.output, "    llvm.unreachable").unwrap();
                    }
                    writeln!(self.output, "  ^bb{success_block}:").unwrap();
                    let raw_payload = self.fresh_value();
                    writeln!(self.output, "    {raw_payload} = llvm.call @__sev_variant_field({result}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    let mut payload = if *payload_type == ValueType::Unit {
                        (raw_payload, ValueType::Any)
                    } else {
                        self.unbox_value((raw_payload, ValueType::Any), *payload_type)
                    };
                    if payload.1 == ValueType::Any
                        && !matches!(*payload_type, ValueType::Any | ValueType::Unit)
                    {
                        payload.1 = *payload_type;
                    }
                    if let Some(receiver) = receiver {
                        self.object_classes
                            .insert(payload.0.clone(), receiver.name.clone());
                        self.receiver_types
                            .insert(payload.0.clone(), receiver.clone());
                    }
                    self.variables.insert(name.id, payload);
                    self.terminated = false;
                }
                Instruction::Assign { target, op, value } => {
                    if let Expression::Variable(name) = target.kind() {
                        let right = self.lower_expression(value);
                        if self.field_names.contains(&name.name)
                            && !self.variables.contains_key(&name.id)
                        {
                            let object = self.field_object.clone().unwrap();
                            let field = self.string_address(&name.name);
                            let value = if *op == AssignmentOp::Assign {
                                right
                            } else {
                                let current = self.fresh_value();
                                writeln!(self.output, "    {current} = llvm.call @__sev_object_get({object}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                                let current = self.unbox_value((current, ValueType::Any), right.1);
                                self.lower_binary_values(current, assignment_binary(*op), right)
                            };
                            let boxed = self.box_value(value);
                            writeln!(self.output, "    llvm.call @__sev_object_set({object}, {field}, {boxed}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                            if let Some(class) = self.object_classes.get(&object).cloned() {
                                self.validate_object(&object, &class);
                            }
                            continue;
                        }
                        let lowered = if *op == AssignmentOp::Assign {
                            match self.variables.get(&name.id).map(|(_, ty)| *ty) {
                                Some(expected)
                                    if expected != ValueType::Any && right.1 == ValueType::Any =>
                                {
                                    self.unbox_value(right, expected)
                                }
                                Some(ValueType::Any) if right.1 != ValueType::Any => {
                                    (self.box_value(right), ValueType::Any)
                                }
                                _ => right,
                            }
                        } else {
                            let left = self
                                .variables
                                .get(&name.id)
                                .cloned()
                                .unwrap_or(right.clone());
                            self.lower_binary_values(left, assignment_binary(*op), right)
                        };
                        self.variables.insert(name.id, lowered);
                    } else if let Expression::Index { object, index } = target.kind() {
                        let (object, object_type) = self.lower_expression(object);
                        let index = self.lower_expression(index);
                        let right = self.lower_expression(value);
                        if object_type == ValueType::Any {
                            let key = self.box_value(index);
                            let boxed = if *op == AssignmentOp::Assign {
                                self.box_value(right)
                            } else {
                                let current = self.fresh_value();
                                writeln!(self.output, "    {current} = llvm.call @__sev_value_get({object}, {key}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                                let current = self.unbox_value((current, ValueType::Any), right.1);
                                let updated = self.lower_binary_values(
                                    current,
                                    assignment_binary(*op),
                                    right,
                                );
                                self.box_value(updated)
                            };
                            writeln!(self.output, "    llvm.call @__sev_value_set({object}, {key}, {boxed}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                        } else if object_type == ValueType::Map {
                            let key = self.box_value(index);
                            let boxed = if *op == AssignmentOp::Assign {
                                self.box_value(right)
                            } else {
                                let current = self.fresh_value();
                                writeln!(self.output, "    {current} = llvm.call @__sev_map_get({object}, {key}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                                let current = self.unbox_value((current, ValueType::Any), right.1);
                                let updated = self.lower_binary_values(
                                    current,
                                    assignment_binary(*op),
                                    right,
                                );
                                self.box_value(updated)
                            };
                            writeln!(self.output, "    llvm.call @__sev_map_insert({object}, {key}, {boxed}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                        } else {
                            let index = self.unbox_value(index, ValueType::Int).0;
                            let boxed = if *op == AssignmentOp::Assign {
                                self.box_value(right)
                            } else {
                                let current = self.fresh_value();
                                writeln!(self.output, "    {current} = llvm.call @__sev_collection_get({object}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                                let current = self.unbox_value((current, ValueType::Any), right.1);
                                let updated = self.lower_binary_values(
                                    current,
                                    assignment_binary(*op),
                                    right,
                                );
                                self.box_value(updated)
                            };
                            writeln!(self.output, "    llvm.call @__sev_collection_set({object}, {index}, {boxed}) : (!llvm.ptr, i64, !llvm.ptr) -> ()").unwrap();
                        }
                    } else if let Expression::Member { object, member } = target.kind() {
                        let (object, _) = self.lower_expression(object);
                        let field = self.string_address(member);
                        let right = self.lower_expression(value);
                        let expected = self
                            .object_field_metadata(&object, member)
                            .map(|(ty, _)| ty)
                            .unwrap_or(right.1);
                        let value = if *op == AssignmentOp::Assign {
                            if right.1 == ValueType::Any && expected != ValueType::Any {
                                self.unbox_value(right, expected)
                            } else {
                                right
                            }
                        } else {
                            let current = self.fresh_value();
                            writeln!(self.output, "    {current} = llvm.call @__sev_object_get({object}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                            let current = self.unbox_value((current, ValueType::Any), expected);
                            self.lower_binary_values(current, assignment_binary(*op), right)
                        };
                        let boxed = self.box_value(value);
                        writeln!(self.output, "    llvm.call @__sev_object_set({object}, {field}, {boxed}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                        if let Some(class) = self.object_classes.get(&object).cloned() {
                            self.validate_object(&object, &class);
                        }
                    }
                }
                Instruction::Print(value) => {
                    let (value, ty) = self.lower_expression(value);
                    match ty {
                        ValueType::String => {
                            let status = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {status} = llvm.call @puts({value}) : (!llvm.ptr) -> i32"
                            )
                            .unwrap();
                        }
                        ValueType::Int => {
                            self.lower_formatted_print("@__sev_fmt_int", &value, ValueType::Int)
                        }
                        ValueType::Float => {
                            self.lower_formatted_print("@__sev_fmt_float", &value, ValueType::Float)
                        }
                        ValueType::Bool => {
                            let true_value = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {true_value} = llvm.mlir.addressof @__sev_bool_true : !llvm.ptr"
                            )
                            .unwrap();
                            let false_value = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {false_value} = llvm.mlir.addressof @__sev_bool_false : !llvm.ptr"
                            )
                            .unwrap();
                            let text = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {text} = llvm.select {value}, {true_value}, {false_value} : i1, !llvm.ptr"
                            )
                            .unwrap();
                            let status = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {status} = llvm.call @puts({text}) : (!llvm.ptr) -> i32"
                            )
                            .unwrap();
                        }
                        ValueType::Any => {
                            writeln!(
                                self.output,
                                "    llvm.call @__sev_print_value({value}) : (!llvm.ptr) -> ()"
                            )
                            .unwrap();
                        }
                        ValueType::List | ValueType::Tuple | ValueType::Set => {
                            writeln!(self.output, "    llvm.call @__sev_print_collection({value}) : (!llvm.ptr) -> ()").unwrap();
                        }
                        ValueType::Result | ValueType::Option => {
                            writeln!(
                                self.output,
                                "    llvm.call @__sev_print_variant({value}) : (!llvm.ptr) -> ()"
                            )
                            .unwrap();
                        }
                        _ => {}
                    }
                }
                Instruction::Evaluate(expression) => {
                    let (value, _) = self.lower_expression(expression);
                    if matches!(expression.kind(), Expression::Task { .. }) {
                        if let Some(return_type) = self.task_results.remove(&value) {
                            if return_type == ValueType::Unit {
                                writeln!(self.output, "    llvm.call @__sev_task_await_unit({value}) : (!llvm.ptr) -> ()").unwrap();
                            } else {
                                let ignored = self.fresh_value();
                                writeln!(self.output, "    {ignored} = llvm.call @__sev_task_await_{}({value}) : (!llvm.ptr) -> {}", task_type_suffix(return_type), mlir_type(return_type)).unwrap();
                            }
                        }
                    }
                }
                Instruction::Assert(expression) => {
                    let runtime_site = expression.hir_id();
                    let lowered = self.lower_expression(expression);
                    let (condition, _) = self.unbox_value(lowered, ValueType::Bool);
                    let passed = self.fresh_block();
                    let failed = self.fresh_block();
                    writeln!(
                        self.output,
                        "    llvm.cond_br {condition}, ^bb{passed}, ^bb{failed}"
                    )
                    .unwrap();
                    writeln!(self.output, "  ^bb{failed}:").unwrap();
                    self.emit_runtime_site_for(runtime_site);
                    writeln!(
                        self.output,
                        "    llvm.call @__sev_runtime_fail_assertion() : () -> ()"
                    )
                    .unwrap();
                    writeln!(self.output, "    llvm.unreachable").unwrap();
                    writeln!(self.output, "  ^bb{passed}:").unwrap();
                    self.terminated = false;
                }
                Instruction::Return(value) => {
                    if self.is_main {
                        let success = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {success} = llvm.mlir.constant(0 : i32) : i32"
                        )
                        .unwrap();
                        writeln!(self.output, "    llvm.return {success} : i32").unwrap();
                    } else if let Some(value) = value {
                        let mut lowered = self.lower_expression(value);
                        if matches!(
                            self.declared_return,
                            ValueType::Any | ValueType::Result | ValueType::Option
                        ) && !matches!(
                            lowered.1,
                            ValueType::Any | ValueType::Result | ValueType::Option
                        ) {
                            lowered = (self.box_value(lowered), ValueType::Any);
                        }
                        let (value, ty) = self.unbox_value(lowered, self.declared_return);
                        writeln!(self.output, "    llvm.return {value} : {}", mlir_type(ty))
                            .unwrap();
                    } else if self.closure_callback {
                        let empty = self.fresh_value();
                        writeln!(self.output, "    {empty} = llvm.mlir.zero : !llvm.ptr").unwrap();
                        writeln!(self.output, "    llvm.return {empty} : !llvm.ptr").unwrap();
                    } else {
                        self.output.push_str("    llvm.return\n");
                    }
                    self.terminated = true;
                }
                Instruction::If {
                    condition,
                    then_instructions,
                    else_instructions,
                } => {
                    let incoming = self.variables.clone();
                    let condition = self.lower_expression(condition);
                    let (condition, _) = self.unbox_value(condition, ValueType::Bool);
                    let then_block = self.fresh_block();
                    let else_block = self.fresh_block();
                    let continue_block = self.fresh_block();
                    writeln!(
                        self.output,
                        "    llvm.cond_br {condition}, ^bb{then_block}, ^bb{else_block}"
                    )
                    .unwrap();
                    writeln!(self.output, "  ^bb{then_block}:").unwrap();
                    self.terminated = false;
                    self.lower_instructions(then_instructions);
                    let then_terminated = self.terminated;
                    let then_variables = self.variables.clone();
                    if !then_terminated {
                        let mut carried = incoming.keys().cloned().collect::<Vec<_>>();
                        carried.sort();
                        let values = carried
                            .iter()
                            .map(|name| {
                                then_variables
                                    .get(name)
                                    .unwrap_or_else(|| &incoming[name])
                                    .0
                                    .as_str()
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        let types = carried
                            .iter()
                            .map(|name| {
                                mlir_type(
                                    then_variables
                                        .get(name)
                                        .unwrap_or_else(|| &incoming[name])
                                        .1,
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        if values.is_empty() {
                            writeln!(self.output, "    llvm.br ^bb{continue_block}").unwrap();
                        } else {
                            writeln!(
                                self.output,
                                "    llvm.br ^bb{continue_block}({values} : {types})"
                            )
                            .unwrap();
                        }
                    }
                    writeln!(self.output, "  ^bb{else_block}:").unwrap();
                    self.variables = incoming.clone();
                    self.terminated = false;
                    self.lower_instructions(else_instructions);
                    let else_terminated = self.terminated;
                    let else_variables = self.variables.clone();
                    if !else_terminated {
                        let mut carried = incoming.keys().cloned().collect::<Vec<_>>();
                        carried.sort();
                        let values = carried
                            .iter()
                            .map(|name| {
                                else_variables
                                    .get(name)
                                    .unwrap_or_else(|| &incoming[name])
                                    .0
                                    .as_str()
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        let types = carried
                            .iter()
                            .map(|name| {
                                mlir_type(
                                    else_variables
                                        .get(name)
                                        .unwrap_or_else(|| &incoming[name])
                                        .1,
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        if values.is_empty() {
                            writeln!(self.output, "    llvm.br ^bb{continue_block}").unwrap();
                        } else {
                            writeln!(
                                self.output,
                                "    llvm.br ^bb{continue_block}({values} : {types})"
                            )
                            .unwrap();
                        }
                    }
                    if !then_terminated || !else_terminated {
                        let mut carried = incoming.into_iter().collect::<Vec<_>>();
                        carried.sort_by(|left, right| left.0.cmp(&right.0));
                        let arguments = carried
                            .iter()
                            .map(|(_, (_, ty))| {
                                let value = self.fresh_value();
                                (value, *ty)
                            })
                            .collect::<Vec<_>>();
                        let signature = arguments
                            .iter()
                            .map(|(value, ty)| format!("{value}: {}", mlir_type(*ty)))
                            .collect::<Vec<_>>()
                            .join(", ");
                        if signature.is_empty() {
                            writeln!(self.output, "  ^bb{continue_block}:").unwrap();
                        } else {
                            writeln!(self.output, "  ^bb{continue_block}({signature}):").unwrap();
                        }
                        self.variables.clear();
                        for ((name, (original, _)), (value, ty)) in
                            carried.into_iter().zip(arguments)
                        {
                            self.carry_value_metadata(&original, &value);
                            self.variables.insert(name, (value, ty));
                        }
                        self.terminated = false;
                    } else {
                        self.terminated = true;
                    }
                }
                Instruction::While {
                    setup,
                    condition,
                    instructions,
                    ..
                } => {
                    if let Some(setup) = setup {
                        self.lower_instructions(std::slice::from_ref(setup));
                    }
                    self.lower_while(condition, instructions);
                }
                Instruction::For {
                    setup,
                    pattern,
                    iterable,
                    instructions,
                } => {
                    if let Some(setup) = setup {
                        self.lower_instructions(std::slice::from_ref(setup));
                    }
                    self.lower_for(pattern, iterable, instructions)
                }
                Instruction::Switch { value, arms } => self.lower_switch(value, arms),
                Instruction::ChannelSwitch {
                    channels,
                    setup,
                    arms,
                    ..
                } => self.lower_channel_switch(channels, setup.as_deref(), arms),
                Instruction::With {
                    placement,
                    instructions,
                    ..
                } => {
                    let previous = self.placement;
                    self.placement = *placement;
                    if matches!(placement, TaskPlacement::Gpu | TaskPlacement::Simd) {
                        let marker = self.fresh_value();
                        let name = if *placement == TaskPlacement::Gpu {
                            "gpu"
                        } else {
                            "simd"
                        };
                        writeln!(self.output, "    {marker} = llvm.mlir.constant(0 : i1) {{severian_parallel = \"{name}\"}} : i1").unwrap();
                    }
                    self.lower_instructions(instructions);
                    self.placement = previous;
                }
                Instruction::Break => self.lower_loop_jump(false),
                Instruction::Continue => self.lower_loop_jump(true),
            }
        }
    }

    pub(in crate::core) fn lower_loop_jump(&mut self, continuing: bool) {
        let target = self
            .loop_targets
            .last()
            .cloned()
            .expect("semantic analysis rejects loop control outside a loop");
        let mut values = Vec::new();
        let mut types = Vec::new();
        if continuing {
            if let Some(index) = target.index {
                let one = self.fresh_value();
                writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
                let next = self.fresh_value();
                writeln!(self.output, "    {next} = llvm.add {index}, {one} : i64").unwrap();
                values.push(next);
                types.push("i64".to_owned());
            }
        }
        for (name, ty) in &target.carried {
            let (value, _) = self.variables.get(name).unwrap();
            values.push(value.clone());
            types.push(mlir_type(*ty).to_owned());
        }
        let block = if continuing {
            target.continue_block
        } else {
            target.break_block
        };
        if values.is_empty() {
            writeln!(self.output, "    llvm.br ^bb{block}").unwrap();
        } else {
            writeln!(
                self.output,
                "    llvm.br ^bb{block}({} : {})",
                values.join(", "),
                types.join(", ")
            )
            .unwrap();
        }
        self.terminated = true;
    }
}
