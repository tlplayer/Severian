use super::*;

impl LowerContext<'_> {
    fn lower_switch_literal_match(
        &mut self,
        value: &str,
        value_type: ValueType,
        pattern: &MatchPattern,
    ) -> Option<String> {
        let expected = match pattern {
            MatchPattern::Integer(expected) => {
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant({expected} : i64) : i64"
                )
                .unwrap();
                (result, ValueType::Int)
            }
            MatchPattern::Float(bits) => {
                let result = self.fresh_value();
                let expected = f64::from_bits(*bits);
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant({expected:.17e} : f64) : f64"
                )
                .unwrap();
                (result, ValueType::Float)
            }
            MatchPattern::Boolean(expected) => {
                let result = self.fresh_value();
                let expected = i32::from(*expected);
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant({expected} : i1) : i1"
                )
                .unwrap();
                (result, ValueType::Bool)
            }
            MatchPattern::String(expected) => (self.string_address(expected), ValueType::String),
            _ => return None,
        };

        let matches = self.fresh_value();
        if value_type == ValueType::Any {
            let expected = self.box_value(expected);
            writeln!(
                self.output,
                "    {matches} = llvm.call @__sev_value_equal({value}, {expected}) : (!llvm.ptr, !llvm.ptr) -> i1"
            )
            .unwrap();
        } else if value_type != expected.1 {
            writeln!(
                self.output,
                "    {matches} = llvm.mlir.constant(0 : i1) : i1"
            )
            .unwrap();
        } else if value_type == ValueType::String {
            writeln!(
                self.output,
                "    {matches} = llvm.call @__sev_string_equal({value}, {}) : (!llvm.ptr, !llvm.ptr) -> i1",
                expected.0
            )
            .unwrap();
        } else if value_type == ValueType::Float {
            writeln!(
                self.output,
                "    {matches} = llvm.fcmp \"oeq\" {value}, {} : f64",
                expected.0
            )
            .unwrap();
        } else {
            writeln!(
                self.output,
                "    {matches} = llvm.icmp \"eq\" {value}, {} : {}",
                expected.0,
                mlir_type(value_type)
            )
            .unwrap();
        }
        Some(matches)
    }

    pub(in crate::core) fn lower_switch(&mut self, value: &Expression, arms: &[SwitchArm]) {
        let (value, value_type) = self.lower_expression(value);
        let incoming = self.variables.clone();
        let mut carried = incoming.keys().cloned().collect::<Vec<_>>();
        carried.sort();
        let exit = self.fresh_block();
        for arm in arms {
            let body = self.fresh_block();
            let next = self.fresh_block();
            let mut bound = Vec::new();
            match &arm.pattern {
                MatchPattern::Constructor { name, fields } => {
                    let tag = self.string_address(name);
                    let matches = self.fresh_value();
                    if let Some(class) = self.classes.iter().find(|class| class.name == *name) {
                        writeln!(self.output, "    {matches} = llvm.call @__sev_object_is({value}, {tag}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
                        let mut combined = matches;
                        for (pattern, field_name) in fields.iter().zip(&class.fields) {
                            let field_name = self.string_address(field_name);
                            let field = self.fresh_value();
                            writeln!(self.output, "    {field} = llvm.call @__sev_object_get({value}, {field_name}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                            match pattern {
                                MatchPattern::Bind(name) => bound.push((
                                    name.id,
                                    self.variables.insert(name.id, (field, ValueType::Any)),
                                )),
                                MatchPattern::Integer(expected) => {
                                    let actual = self.fresh_value();
                                    writeln!(self.output, "    {actual} = llvm.call @__sev_unbox_i64({field}) : (!llvm.ptr) -> i64").unwrap();
                                    let expected_value = self.fresh_value();
                                    writeln!(self.output, "    {expected_value} = llvm.mlir.constant({expected} : i64) : i64").unwrap();
                                    let field_matches = self.fresh_value();
                                    writeln!(self.output, "    {field_matches} = llvm.icmp \"eq\" {actual}, {expected_value} : i64").unwrap();
                                    let both = self.fresh_value();
                                    writeln!(
                                        self.output,
                                        "    {both} = llvm.and {combined}, {field_matches} : i1"
                                    )
                                    .unwrap();
                                    combined = both;
                                }
                                MatchPattern::Wildcard => {}
                                _ => {}
                            }
                        }
                        writeln!(
                            self.output,
                            "    llvm.cond_br {combined}, ^bb{body}, ^bb{next}"
                        )
                        .unwrap();
                    } else {
                        writeln!(self.output, "    {matches} = llvm.call @__sev_variant_is({value}, {tag}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
                        let successful_result = name == "ok";
                        if !fields.is_empty() {
                            let payload = self.fresh_value();
                            writeln!(self.output, "    {payload} = llvm.call @__sev_variant_field({value}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                            for (index, pattern) in fields.iter().enumerate() {
                                let MatchPattern::Bind(name) = pattern else {
                                    continue;
                                };
                                let field = if fields.len() == 1 {
                                    payload.clone()
                                } else {
                                    let index_value = self.fresh_value();
                                    writeln!(
                                        self.output,
                                        "    {index_value} = llvm.mlir.constant({index} : i64) : i64"
                                    )
                                    .unwrap();
                                    let field = self.fresh_value();
                                    writeln!(self.output, "    {field} = llvm.call @__sev_collection_get({payload}, {index_value}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                                    field
                                };
                                if successful_result {
                                    if let Some(receiver) = arm.receivers.get(&name.id) {
                                        self.object_classes
                                            .insert(field.clone(), receiver.name.clone());
                                        self.receiver_types.insert(field.clone(), receiver.clone());
                                    }
                                }
                                bound.push((
                                    name.id,
                                    self.variables.insert(name.id, (field, ValueType::Any)),
                                ));
                            }
                        }
                        writeln!(
                            self.output,
                            "    llvm.cond_br {matches}, ^bb{body}, ^bb{next}"
                        )
                        .unwrap();
                    }
                }
                MatchPattern::Bind(name) => {
                    bound.push((
                        name.id,
                        self.variables.insert(name.id, (value.clone(), value_type)),
                    ));
                    writeln!(self.output, "    llvm.br ^bb{body}").unwrap();
                }
                MatchPattern::Wildcard => {
                    writeln!(self.output, "    llvm.br ^bb{body}").unwrap();
                }
                pattern => {
                    let matches = self
                        .lower_switch_literal_match(&value, value_type, pattern)
                        .expect("literal switch pattern must lower to a comparison");
                    writeln!(
                        self.output,
                        "    llvm.cond_br {matches}, ^bb{body}, ^bb{next}"
                    )
                    .unwrap();
                }
            }
            writeln!(self.output, "  ^bb{body}:").unwrap();
            if let Some(guard) = &arm.guard {
                let guarded = self.fresh_block();
                let (guard, _) = self.lower_expression(guard);
                writeln!(
                    self.output,
                    "    llvm.cond_br {guard}, ^bb{guarded}, ^bb{next}"
                )
                .unwrap();
                writeln!(self.output, "  ^bb{guarded}:").unwrap();
            }
            self.terminated = false;
            self.lower_instructions(&arm.instructions);
            if !self.terminated {
                let values = carried
                    .iter()
                    .map(|name| {
                        self.variables
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
                            self.variables
                                .get(name)
                                .unwrap_or_else(|| &incoming[name])
                                .1,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                if values.is_empty() {
                    writeln!(self.output, "    llvm.br ^bb{exit}").unwrap();
                } else {
                    writeln!(self.output, "    llvm.br ^bb{exit}({values} : {types})").unwrap();
                }
            }
            for (name, previous) in bound {
                if let Some(previous) = previous {
                    self.variables.insert(name, previous);
                } else {
                    self.variables.remove(&name);
                }
            }
            writeln!(self.output, "  ^bb{next}:").unwrap();
            self.variables = incoming.clone();
            self.terminated = false;
        }
        let values = carried
            .iter()
            .map(|name| incoming[name].0.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let types = carried
            .iter()
            .map(|name| mlir_type(incoming[name].1))
            .collect::<Vec<_>>()
            .join(", ");
        if values.is_empty() {
            writeln!(self.output, "    llvm.br ^bb{exit}").unwrap();
        } else {
            writeln!(self.output, "    llvm.br ^bb{exit}({values} : {types})").unwrap();
        }
        let arguments = carried
            .iter()
            .map(|name| (self.fresh_value(), incoming[name].1))
            .collect::<Vec<_>>();
        let signature = arguments
            .iter()
            .map(|(value, ty)| format!("{value}: {}", mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        if signature.is_empty() {
            writeln!(self.output, "  ^bb{exit}:").unwrap();
        } else {
            writeln!(self.output, "  ^bb{exit}({signature}):").unwrap();
        }
        self.variables.clear();
        for (name, (value, ty)) in carried.into_iter().zip(arguments) {
            self.carry_value_metadata(&incoming[&name].0, &value);
            self.variables.insert(name, (value, ty));
        }
        self.terminated = false;
    }

    pub(in crate::core) fn lower_channel_switch(
        &mut self,
        channels: &[Expression],
        setup: Option<&Instruction>,
        arms: &[SwitchArm],
    ) {
        if let Some(setup) = setup {
            self.lower_instructions(std::slice::from_ref(setup));
        }
        for channel in channels {
            let Expression::Variable(channel_name) = channel.kind() else {
                continue;
            };
            let Some(arm) = arms.iter().find(|arm| {
                matches!(
                    arm.source.as_ref(),
                    Some(source) if matches!(source.kind(), Expression::Variable(name) if name == channel_name)
                )
            }) else {
                continue;
            };
            let (channel, channel_type) = self.lower_expression(channel);
            let channel_type = self
                .channel_types
                .get(&channel)
                .copied()
                .unwrap_or(channel_type);
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @__sev_channel_receive_ptr({channel}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
            let result = self.unbox_value((result, ValueType::Any), channel_type);
            let mut bound = None;
            if let MatchPattern::Bind(name) = &arm.pattern {
                bound = Some((name.id, self.variables.insert(name.id, result)));
            }
            self.lower_instructions(&arm.instructions);
            if let Some((name, previous)) = bound {
                if let Some(previous) = previous {
                    self.variables.insert(name, previous);
                } else {
                    self.variables.remove(&name);
                }
            }
        }
    }

    pub(in crate::core) fn lower_while(&mut self, condition: &Expression, instructions: &[Instruction]) {
        let mut carried = self
            .variables
            .iter()
            .map(|(name, (value, ty))| (name.clone(), value.clone(), *ty))
            .collect::<Vec<_>>();
        carried.sort_by(|left, right| left.0.cmp(&right.0));

        let header = self.fresh_block();
        let body = self.fresh_block();
        let exit = self.fresh_block();
        let initial_values = carried
            .iter()
            .map(|(_, value, _)| value.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let initial_types = carried
            .iter()
            .map(|(_, _, ty)| mlir_type(*ty))
            .collect::<Vec<_>>()
            .join(", ");
        if initial_values.is_empty() {
            writeln!(self.output, "    llvm.br ^bb{header}").unwrap();
        } else {
            writeln!(
                self.output,
                "    llvm.br ^bb{header}({initial_values} : {initial_types})"
            )
            .unwrap();
        }

        let header_values = carried
            .iter()
            .map(|(name, _, ty)| (name.clone(), self.fresh_value(), *ty))
            .collect::<Vec<_>>();
        let header_arguments = header_values
            .iter()
            .map(|(_, value, ty)| format!("{value}: {}", mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        if header_arguments.is_empty() {
            writeln!(self.output, "  ^bb{header}:").unwrap();
        } else {
            writeln!(self.output, "  ^bb{header}({header_arguments}):").unwrap();
        }
        for (name, value, ty) in &header_values {
            self.variables.insert(name.clone(), (value.clone(), *ty));
            if let Some((_, original, _)) =
                carried.iter().find(|(candidate, _, _)| candidate == name)
            {
                self.carry_value_metadata(original, value);
            }
        }
        let condition = self.lower_expression(condition);
        let (condition, _) = self.unbox_value(condition, ValueType::Bool);
        let exit_values = header_values
            .iter()
            .map(|(_, value, _)| value.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let exit_suffix = if exit_values.is_empty() {
            String::new()
        } else {
            format!("({exit_values} : {initial_types})")
        };
        writeln!(
            self.output,
            "    llvm.cond_br {condition}, ^bb{body}, ^bb{exit}{exit_suffix}"
        )
        .unwrap();

        writeln!(self.output, "  ^bb{body}:").unwrap();
        self.loop_targets.push(LoopTarget {
            break_block: exit,
            continue_block: header,
            carried: carried
                .iter()
                .map(|(name, _, ty)| (name.clone(), *ty))
                .collect(),
            index: None,
        });
        self.terminated = false;
        self.lower_instructions(instructions);
        self.loop_targets.pop();
        if !self.terminated {
            let next_values = carried
                .iter()
                .map(|(name, _, _)| {
                    let (value, _) = self.variables.get(name).unwrap();
                    value.as_str()
                })
                .collect::<Vec<_>>()
                .join(", ");
            if next_values.is_empty() {
                writeln!(self.output, "    llvm.br ^bb{header}").unwrap();
            } else {
                writeln!(
                    self.output,
                    "    llvm.br ^bb{header}({next_values} : {initial_types})"
                )
                .unwrap();
            }
        }

        let exit_arguments = header_values
            .iter()
            .map(|(_, _, ty)| (self.fresh_value(), *ty))
            .collect::<Vec<_>>();
        let exit_signature = exit_arguments
            .iter()
            .map(|(value, ty)| format!("{value}: {}", mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        if exit_signature.is_empty() {
            writeln!(self.output, "  ^bb{exit}:").unwrap();
        } else {
            writeln!(self.output, "  ^bb{exit}({exit_signature}):").unwrap();
        }
        for ((name, _, _), (value, ty)) in header_values.iter().zip(exit_arguments) {
            if let Some((_, original, _)) =
                carried.iter().find(|(candidate, _, _)| candidate == name)
            {
                self.carry_value_metadata(original, &value);
            }
            self.variables.insert(name.clone(), (value, ty));
        }
        let carried_names = carried
            .iter()
            .map(|(name, _, _)| *name)
            .collect::<HashSet<_>>();
        self.variables
            .retain(|name, _| carried_names.contains(name));
        self.terminated = false;
    }

    pub(in crate::core) fn lower_for(
        &mut self,
        pattern: &severian_hir::MatchPattern,
        iterable: &Expression,
        instructions: &[Instruction],
    ) {
        let binding_names = match pattern {
            MatchPattern::Bind(name) => vec![name.clone()],
            MatchPattern::Constructor { name, fields } if name == "tuple" => fields
                .iter()
                .filter_map(|field| match field {
                    MatchPattern::Bind(name) => Some(name.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        let mut collection = None;
        let mut yields_indices = false;
        let mut map_collection = false;
        let mut string_collection = false;
        let (start, end) = match iterable.kind() {
            Expression::Call { target, args }
                if target.name == "range" && (1..=2).contains(&args.len()) =>
            {
                if args.len() == 1 {
                    let start = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {start} = llvm.mlir.constant(0 : i64) : i64"
                    )
                    .unwrap();
                    let end = self.lower_expression(&args[0]);
                    let end = self.unbox_value(end, ValueType::Int).0;
                    (start, end)
                } else {
                    let start = self.lower_expression(&args[0]);
                    let end = self.lower_expression(&args[1]);
                    (
                        self.unbox_value(start, ValueType::Int).0,
                        self.unbox_value(end, ValueType::Int).0,
                    )
                }
            }
            Expression::Call { target, args } if target.name == "indices" && args.len() == 1 => {
                let (mut value, value_type) = self.lower_expression(&args[0]);
                if value_type == ValueType::Any {
                    value = self.unbox_value((value, value_type), ValueType::List).0;
                }
                let start = self.fresh_value();
                writeln!(
                    self.output,
                    "    {start} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                let end = self.fresh_value();
                writeln!(
                    self.output,
                    "    {end} = llvm.call @__sev_collection_size({value}) : (!llvm.ptr) -> i64"
                )
                .unwrap();
                collection = Some(value);
                yields_indices = true;
                (start, end)
            }
            _ => {
                let (mut value, mut iterable_type) = self.lower_expression(iterable);
                if iterable_type == ValueType::Result {
                    let result = value;
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
                    value = self.fresh_value();
                    writeln!(self.output, "    {value} = llvm.call @__sev_variant_field({result}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    iterable_type = ValueType::Any;
                }
                if iterable_type == ValueType::Any {
                    value = self.unbox_value((value, iterable_type), ValueType::List).0;
                }
                let start = self.fresh_value();
                writeln!(
                    self.output,
                    "    {start} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                let end = self.fresh_value();
                if iterable_type == ValueType::Map {
                    writeln!(
                        self.output,
                        "    {end} = llvm.call @__sev_map_size({value}) : (!llvm.ptr) -> i64"
                    )
                    .unwrap();
                    map_collection = true;
                } else if iterable_type == ValueType::String {
                    writeln!(
                        self.output,
                        "    {end} = llvm.call @__sev_string_length({value}) : (!llvm.ptr) -> i64"
                    )
                    .unwrap();
                    string_collection = true;
                } else {
                    writeln!(self.output, "    {end} = llvm.call @__sev_collection_size({value}) : (!llvm.ptr) -> i64").unwrap();
                }
                collection = Some(value);
                (start, end)
            }
        };

        let previous_bindings = binding_names
            .iter()
            .map(|name| (name.id, self.variables.remove(&name.id)))
            .collect::<Vec<_>>();
        let mut carried = self
            .variables
            .iter()
            .map(|(name, (value, ty))| (name.clone(), value.clone(), *ty))
            .collect::<Vec<_>>();
        carried.sort_by(|left, right| left.0.cmp(&right.0));
        let header = self.fresh_block();
        let body = self.fresh_block();
        let exit = self.fresh_block();
        let initial_values = std::iter::once(start.as_str())
            .chain(carried.iter().map(|(_, value, _)| value.as_str()))
            .collect::<Vec<_>>()
            .join(", ");
        let initial_types = std::iter::once("i64")
            .chain(carried.iter().map(|(_, _, ty)| mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            self.output,
            "    llvm.br ^bb{header}({initial_values} : {initial_types})"
        )
        .unwrap();
        let index = self.fresh_value();
        let header_values = carried
            .iter()
            .map(|(name, _, ty)| (name.clone(), self.fresh_value(), *ty))
            .collect::<Vec<_>>();
        let header_arguments = header_values
            .iter()
            .map(|(_, value, ty)| format!("{value}: {}", mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        let header_suffix = if header_arguments.is_empty() {
            String::new()
        } else {
            format!(", {header_arguments}")
        };
        writeln!(self.output, "  ^bb{header}({index}: i64{header_suffix}):").unwrap();
        for (name, value, ty) in &header_values {
            self.variables.insert(name.clone(), (value.clone(), *ty));
            if let Some((_, original, _)) =
                carried.iter().find(|(candidate, _, _)| candidate == name)
            {
                self.carry_value_metadata(original, value);
            }
        }
        let condition = self.fresh_value();
        writeln!(
            self.output,
            "    {condition} = llvm.icmp \"slt\" {index}, {end} : i64"
        )
        .unwrap();
        let exit_value_names = header_values
            .iter()
            .map(|(_, value, _)| value.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let exit_types = header_values
            .iter()
            .map(|(_, _, ty)| mlir_type(*ty))
            .collect::<Vec<_>>()
            .join(", ");
        let exit_suffix = if exit_value_names.is_empty() {
            String::new()
        } else {
            format!("({exit_value_names} : {exit_types})")
        };
        writeln!(
            self.output,
            "    llvm.cond_br {condition}, ^bb{body}, ^bb{exit}{exit_suffix}"
        )
        .unwrap();

        writeln!(self.output, "  ^bb{body}:").unwrap();
        let binding = if let Some(collection) = &collection {
            if yields_indices {
                (index.clone(), ValueType::Int)
            } else if map_collection {
                let value = self.fresh_value();
                writeln!(self.output, "    {value} = llvm.call @__sev_map_value_at({collection}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                if let MatchPattern::Constructor { fields, .. } = pattern {
                    for (position, field) in fields.iter().enumerate() {
                        if let MatchPattern::Bind(name) = field {
                            let item = self.fresh_value();
                            let function = if position == 0 {
                                "__sev_map_key_at"
                            } else {
                                "__sev_map_value_at"
                            };
                            writeln!(self.output, "    {item} = llvm.call @{function}({collection}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                            self.variables.insert(name.id, (item, ValueType::Any));
                        }
                    }
                }
                (value, ValueType::Any)
            } else if string_collection {
                let item = self.fresh_value();
                writeln!(self.output, "    {item} = llvm.call @__sev_string_char_at({collection}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                (item, ValueType::String)
            } else {
                let item = self.fresh_value();
                writeln!(self.output, "    {item} = llvm.call @__sev_collection_get({collection}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                if let MatchPattern::Constructor { name, fields } = pattern {
                    if name == "tuple" {
                        let tuple = self
                            .unbox_value((item.clone(), ValueType::Any), ValueType::Tuple)
                            .0;
                        for (position, field) in fields.iter().enumerate() {
                            if let MatchPattern::Bind(name) = field {
                                let position_value = self.fresh_value();
                                writeln!(self.output, "    {position_value} = llvm.mlir.constant({position} : i64) : i64").unwrap();
                                let field_value = self.fresh_value();
                                writeln!(self.output, "    {field_value} = llvm.call @__sev_collection_get({tuple}, {position_value}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                                self.variables
                                    .insert(name.id, (field_value, ValueType::Any));
                            }
                        }
                    }
                }
                (item, ValueType::Any)
            }
        } else {
            (index.clone(), ValueType::Int)
        };
        if let MatchPattern::Bind(name) = pattern {
            self.variables.insert(name.id, binding);
        }
        self.loop_targets.push(LoopTarget {
            break_block: exit,
            continue_block: header,
            carried: carried
                .iter()
                .map(|(name, _, ty)| (name.clone(), *ty))
                .collect(),
            index: Some(index.clone()),
        });
        self.terminated = false;
        self.lower_instructions(instructions);
        self.loop_targets.pop();
        if !self.terminated {
            let one = self.fresh_value();
            writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
            let next = self.fresh_value();
            writeln!(self.output, "    {next} = llvm.add {index}, {one} : i64").unwrap();
            let carried_values = carried
                .iter()
                .map(|(name, _, _)| {
                    let (value, _) = self.variables.get(name).unwrap();
                    value.as_str()
                })
                .collect::<Vec<_>>()
                .join(", ");
            let carried_types = carried
                .iter()
                .map(|(_, _, ty)| mlir_type(*ty))
                .collect::<Vec<_>>()
                .join(", ");
            let next_values = if carried_values.is_empty() {
                next.clone()
            } else {
                format!("{next}, {carried_values}")
            };
            let next_types = if carried_types.is_empty() {
                "i64".to_owned()
            } else {
                format!("i64, {carried_types}")
            };
            writeln!(
                self.output,
                "    llvm.br ^bb{header}({next_values} : {next_types})"
            )
            .unwrap();
        }

        let exit_arguments = header_values
            .iter()
            .map(|(_, _, ty)| (self.fresh_value(), *ty))
            .collect::<Vec<_>>();
        let exit_signature = exit_arguments
            .iter()
            .map(|(value, ty)| format!("{value}: {}", mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        if exit_signature.is_empty() {
            writeln!(self.output, "  ^bb{exit}:").unwrap();
        } else {
            writeln!(self.output, "  ^bb{exit}({exit_signature}):").unwrap();
        }
        for ((variable, _, _), (value, ty)) in header_values.iter().zip(&exit_arguments) {
            self.variables
                .insert(variable.clone(), (value.clone(), *ty));
            if let Some((_, original, _)) = carried
                .iter()
                .find(|(candidate, _, _)| candidate == variable)
            {
                self.carry_value_metadata(original, value);
            }
        }
        let carried_names = carried
            .iter()
            .map(|(name, _, _)| *name)
            .collect::<HashSet<_>>();
        self.variables
            .retain(|name, _| carried_names.contains(name));
        for (name, previous_binding) in previous_bindings {
            if let Some(previous_binding) = previous_binding {
                self.variables.insert(name, previous_binding);
            } else {
                self.variables.remove(&name);
            }
        }
        self.terminated = false;
    }

    pub(in crate::core) fn fresh_block(&mut self) -> usize {
        let block = self.next_block;
        self.next_block += 1;
        block
    }
}
