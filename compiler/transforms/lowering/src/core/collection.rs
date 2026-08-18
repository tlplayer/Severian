use super::*;

impl LowerContext<'_> {
    pub(super) fn lower_comprehension(
        &mut self,
        element: Option<&Expression>,
        key: Option<&Expression>,
        value: Option<&Expression>,
        clauses: &[ComprehensionClause],
        result_type: ValueType,
    ) -> (String, ValueType) {
        let result = self.fresh_value();
        if result_type == ValueType::Map {
            writeln!(
                self.output,
                "    {result} = llvm.call @__sev_map_new() : () -> !llvm.ptr"
            )
            .unwrap();
        } else {
            let kind = self.fresh_value();
            let kind_value = i32::from(result_type == ValueType::Set) * 2;
            writeln!(
                self.output,
                "    {kind} = llvm.mlir.constant({kind_value} : i64) : i64"
            )
            .unwrap();
            writeln!(
                self.output,
                "    {result} = llvm.call @__sev_collection_new({kind}) : (i64) -> !llvm.ptr"
            )
            .unwrap();
        }
        self.lower_comprehension_level(element, key, value, clauses, 0, &result, result_type);
        (result, result_type)
    }

    pub(super) fn lower_inline_callable(
        &mut self,
        callable: &Expression,
        args: Vec<(String, ValueType)>,
    ) -> (String, ValueType) {
        let (callee, _) = self.lower_expression(callable);
        let boxed = args
            .into_iter()
            .map(|value| self.box_value(value))
            .collect::<Vec<_>>();
        let values = boxed.join(", ");
        let type_suffix = std::iter::repeat_n("!llvm.ptr", boxed.len())
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
        let type_suffix = if type_suffix.is_empty() {
            String::new()
        } else {
            format!(", {type_suffix}")
        };
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call {function}({environment}{value_suffix}) : !llvm.ptr, (!llvm.ptr{type_suffix}) -> !llvm.ptr"
        )
        .unwrap();
        (result, ValueType::Any)
    }

    pub(super) fn lower_collection_transform(
        &mut self,
        object: &Expression,
        callable: &Expression,
        filter: bool,
    ) -> (String, ValueType) {
        let (mut object, object_type) = self.lower_expression(object);
        if object_type == ValueType::Any {
            object = self.unbox_value((object, object_type), ValueType::List).0;
        }
        let result = self.lower_collection_transform_from_value(object, callable, filter);
        (result, ValueType::List)
    }

    pub(super) fn lower_collection_transform_from_value(
        &mut self,
        object: String,
        callable: &Expression,
        filter: bool,
    ) -> String {
        let kind = self.fresh_value();
        writeln!(
            self.output,
            "    {kind} = llvm.mlir.constant(0 : i64) : i64"
        )
        .unwrap();
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @__sev_collection_new({kind}) : (i64) -> !llvm.ptr"
        )
        .unwrap();
        let size = self.fresh_value();
        writeln!(
            self.output,
            "    {size} = llvm.call @__sev_collection_size({object}) : (!llvm.ptr) -> i64"
        )
        .unwrap();
        let zero = self.fresh_value();
        writeln!(
            self.output,
            "    {zero} = llvm.mlir.constant(0 : i64) : i64"
        )
        .unwrap();
        let header = self.fresh_block();
        let body = self.fresh_block();
        let append = self.fresh_block();
        let step = self.fresh_block();
        let exit = self.fresh_block();
        writeln!(self.output, "    llvm.br ^bb{header}({zero} : i64)").unwrap();
        let index = self.fresh_value();
        writeln!(self.output, "  ^bb{header}({index}: i64):").unwrap();
        let more = self.fresh_value();
        writeln!(
            self.output,
            "    {more} = llvm.icmp \"slt\" {index}, {size} : i64"
        )
        .unwrap();
        writeln!(self.output, "    llvm.cond_br {more}, ^bb{body}, ^bb{exit}").unwrap();
        writeln!(self.output, "  ^bb{body}:").unwrap();
        let item = self.fresh_value();
        writeln!(self.output, "    {item} = llvm.call @__sev_collection_get({object}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
        let transformed =
            self.lower_inline_callable(callable, vec![(item.clone(), ValueType::Any)]);
        if filter {
            let (condition, _) = self.unbox_value(transformed, ValueType::Bool);
            writeln!(
                self.output,
                "    llvm.cond_br {condition}, ^bb{append}, ^bb{step}"
            )
            .unwrap();
            writeln!(self.output, "  ^bb{append}:").unwrap();
            writeln!(self.output, "    llvm.call @__sev_collection_push({result}, {item}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
        } else {
            let transformed = self.box_value(transformed);
            writeln!(self.output, "    llvm.call @__sev_collection_push({result}, {transformed}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
        }
        writeln!(self.output, "    llvm.br ^bb{step}").unwrap();
        writeln!(self.output, "  ^bb{step}:").unwrap();
        let one = self.fresh_value();
        writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
        let next = self.fresh_value();
        writeln!(self.output, "    {next} = llvm.add {index}, {one} : i64").unwrap();
        writeln!(self.output, "    llvm.br ^bb{header}({next} : i64)").unwrap();
        writeln!(self.output, "  ^bb{exit}:").unwrap();
        result
    }

    pub(super) fn lower_collection_reduce(
        &mut self,
        object: &Expression,
        callable: &Expression,
        initial: Option<&Expression>,
    ) -> (String, ValueType) {
        let (mut object, object_type) = self.lower_expression(object);
        if object_type == ValueType::Any {
            object = self.unbox_value((object, object_type), ValueType::List).0;
        }
        let (start, accumulator) = if let Some(initial) = initial {
            let start = self.fresh_value();
            writeln!(
                self.output,
                "    {start} = llvm.mlir.constant(0 : i64) : i64"
            )
            .unwrap();
            let initial = self.lower_expression(initial);
            (start, self.box_value(initial))
        } else {
            let zero = self.fresh_value();
            writeln!(
                self.output,
                "    {zero} = llvm.mlir.constant(0 : i64) : i64"
            )
            .unwrap();
            let accumulator = self.fresh_value();
            writeln!(self.output, "    {accumulator} = llvm.call @__sev_collection_get({object}, {zero}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
            let start = self.fresh_value();
            writeln!(
                self.output,
                "    {start} = llvm.mlir.constant(1 : i64) : i64"
            )
            .unwrap();
            (start, accumulator)
        };
        let size = self.fresh_value();
        writeln!(
            self.output,
            "    {size} = llvm.call @__sev_collection_size({object}) : (!llvm.ptr) -> i64"
        )
        .unwrap();
        let header = self.fresh_block();
        let body = self.fresh_block();
        let exit = self.fresh_block();
        writeln!(
            self.output,
            "    llvm.br ^bb{header}({start}, {accumulator} : i64, !llvm.ptr)"
        )
        .unwrap();
        let index = self.fresh_value();
        let current = self.fresh_value();
        writeln!(
            self.output,
            "  ^bb{header}({index}: i64, {current}: !llvm.ptr):"
        )
        .unwrap();
        let more = self.fresh_value();
        writeln!(
            self.output,
            "    {more} = llvm.icmp \"slt\" {index}, {size} : i64"
        )
        .unwrap();
        writeln!(
            self.output,
            "    llvm.cond_br {more}, ^bb{body}, ^bb{exit}({current} : !llvm.ptr)"
        )
        .unwrap();
        writeln!(self.output, "  ^bb{body}:").unwrap();
        let item = self.fresh_value();
        writeln!(self.output, "    {item} = llvm.call @__sev_collection_get({object}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
        let updated = self.lower_inline_callable(
            callable,
            vec![(current, ValueType::Any), (item, ValueType::Any)],
        );
        let updated = self.box_value(updated);
        let one = self.fresh_value();
        writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
        let next = self.fresh_value();
        writeln!(self.output, "    {next} = llvm.add {index}, {one} : i64").unwrap();
        writeln!(
            self.output,
            "    llvm.br ^bb{header}({next}, {updated} : i64, !llvm.ptr)"
        )
        .unwrap();
        let result = self.fresh_value();
        writeln!(self.output, "  ^bb{exit}({result}: !llvm.ptr):").unwrap();
        (result, ValueType::Any)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_comprehension_level(
        &mut self,
        element: Option<&Expression>,
        key: Option<&Expression>,
        value: Option<&Expression>,
        clauses: &[ComprehensionClause],
        depth: usize,
        result: &str,
        result_type: ValueType,
    ) {
        let clause = &clauses[depth];
        let (mut iterable, iterable_type) = self.lower_expression(&clause.iterable);
        if iterable_type == ValueType::Any {
            iterable = self
                .unbox_value((iterable, iterable_type), ValueType::List)
                .0;
        }
        let size = self.fresh_value();
        writeln!(
            self.output,
            "    {size} = llvm.call @__sev_collection_size({iterable}) : (!llvm.ptr) -> i64"
        )
        .unwrap();
        let zero = self.fresh_value();
        writeln!(
            self.output,
            "    {zero} = llvm.mlir.constant(0 : i64) : i64"
        )
        .unwrap();
        let header = self.fresh_block();
        let body = self.fresh_block();
        let accepted = self.fresh_block();
        let step_block = self.fresh_block();
        let exit = self.fresh_block();
        writeln!(self.output, "    llvm.br ^bb{header}({zero} : i64)").unwrap();
        let index = self.fresh_value();
        writeln!(self.output, "  ^bb{header}({index}: i64):").unwrap();
        let more = self.fresh_value();
        writeln!(
            self.output,
            "    {more} = llvm.icmp \"slt\" {index}, {size} : i64"
        )
        .unwrap();
        writeln!(self.output, "    llvm.cond_br {more}, ^bb{body}, ^bb{exit}").unwrap();
        writeln!(self.output, "  ^bb{body}:").unwrap();
        let item = self.fresh_value();
        writeln!(self.output, "    {item} = llvm.call @__sev_collection_get({iterable}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
        let previous = self.bind_comprehension_pattern(&clause.pattern, &item);
        if let Some(condition) = &clause.condition {
            let condition = self.lower_expression(condition);
            let (condition, _) = self.unbox_value(condition, ValueType::Bool);
            writeln!(
                self.output,
                "    llvm.cond_br {condition}, ^bb{accepted}, ^bb{step_block}"
            )
            .unwrap();
            writeln!(self.output, "  ^bb{accepted}:").unwrap();
        }
        if depth + 1 < clauses.len() {
            self.lower_comprehension_level(
                element,
                key,
                value,
                clauses,
                depth + 1,
                result,
                result_type,
            );
        } else if result_type == ValueType::Map {
            let key = self.lower_expression(key.unwrap());
            let key = self.box_value(key);
            let value = self.lower_expression(value.unwrap());
            let value = self.box_value(value);
            writeln!(self.output, "    llvm.call @__sev_map_insert({result}, {key}, {value}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
        } else {
            let value = self.lower_expression(element.unwrap());
            let value = self.box_value(value);
            let function = if result_type == ValueType::Set {
                "__sev_set_add"
            } else {
                "__sev_collection_push"
            };
            writeln!(
                self.output,
                "    llvm.call @{function}({result}, {value}) : (!llvm.ptr, !llvm.ptr) -> ()"
            )
            .unwrap();
        }
        writeln!(self.output, "    llvm.br ^bb{step_block}").unwrap();
        writeln!(self.output, "  ^bb{step_block}:").unwrap();
        let one = self.fresh_value();
        writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
        let next = self.fresh_value();
        writeln!(self.output, "    {next} = llvm.add {index}, {one} : i64").unwrap();
        writeln!(self.output, "    llvm.br ^bb{header}({next} : i64)").unwrap();
        writeln!(self.output, "  ^bb{exit}:").unwrap();
        for (name, binding) in previous {
            if let Some(binding) = binding {
                self.variables.insert(name, binding);
            } else {
                self.variables.remove(&name);
            }
        }
    }

    pub(super) fn bind_comprehension_pattern(
        &mut self,
        pattern: &MatchPattern,
        item: &str,
    ) -> Vec<(BindingId, Option<(String, ValueType)>)> {
        match pattern {
            MatchPattern::Bind(name) => vec![(
                name.id,
                self.variables
                    .insert(name.id, (item.to_owned(), ValueType::Any)),
            )],
            MatchPattern::Constructor { name, fields } if name == "tuple" => {
                let tuple = self
                    .unbox_value((item.to_owned(), ValueType::Any), ValueType::Tuple)
                    .0;
                fields
                    .iter()
                    .enumerate()
                    .filter_map(|(position, field)| {
                        let MatchPattern::Bind(name) = field else {
                            return None;
                        };
                        let position_value = self.fresh_value();
                        writeln!(self.output, "    {position_value} = llvm.mlir.constant({position} : i64) : i64").unwrap();
                        let value = self.fresh_value();
                        writeln!(self.output, "    {value} = llvm.call @__sev_collection_get({tuple}, {position_value}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                        Some((
                            name.id,
                            self.variables
                                .insert(name.id, (value, ValueType::Any)),
                        ))
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }
}
