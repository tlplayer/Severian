use super::*;

impl LowerContext<'_> {
    pub(super) fn lower_short_circuit_chain(
        &mut self,
        left: &Expression,
        op: BinaryOp,
        right: &Expression,
    ) -> (String, ValueType) {
        let mut operands = Vec::new();
        collect_short_circuit_operands(left, op, &mut operands);
        collect_short_circuit_operands(right, op, &mut operands);
        let mut value = self.lower_expression(operands[0]);
        for operand in &operands[1..] {
            let (left, _) = self.unbox_value(value, ValueType::Bool);
            let right_block = self.fresh_block();
            let continue_block = self.fresh_block();
            let constant = self.fresh_value();
            let constant_value = i32::from(op == BinaryOp::Or);
            writeln!(
                self.output,
                "    {constant} = llvm.mlir.constant({constant_value} : i1) : i1"
            )
            .unwrap();
            if op == BinaryOp::And {
                writeln!(self.output, "    llvm.cond_br {left}, ^bb{right_block}, ^bb{continue_block}({constant} : i1)").unwrap();
            } else {
                writeln!(self.output, "    llvm.cond_br {left}, ^bb{continue_block}({constant} : i1), ^bb{right_block}").unwrap();
            }
            writeln!(self.output, "  ^bb{right_block}:").unwrap();
            let right = self.lower_expression(operand);
            let (right, _) = self.unbox_value(right, ValueType::Bool);
            writeln!(self.output, "    llvm.br ^bb{continue_block}({right} : i1)").unwrap();
            let result = self.fresh_value();
            writeln!(self.output, "  ^bb{continue_block}({result}: i1):").unwrap();
            value = (result, ValueType::Bool);
        }
        value
    }

    pub(super) fn lower_slice_expression(
        &mut self,
        object: &Expression,
        start: &Option<Box<Expression>>,
        end: &Option<Box<Expression>>,
        step: &Option<Box<Expression>>,
        expected_type: Option<ValueType>,
    ) -> (String, ValueType) {
        let (mut object, mut object_type) = self.lower_expression(object);
        let dynamic = object_type == ValueType::Any
            && !matches!(expected_type, Some(ValueType::String | ValueType::List));
        if object_type == ValueType::Any && !dynamic {
            object_type = expected_type.unwrap();
            object = self.unbox_value((object, ValueType::Any), object_type).0;
        }
        let mut bounds = Vec::new();
        for bound in [start, end, step] {
            if let Some(bound) = bound {
                let lowered = self.lower_expression(bound);
                bounds.push(self.unbox_value(lowered, ValueType::Int).0);
            } else {
                let missing = self.fresh_value();
                writeln!(
                    self.output,
                    "    {missing} = llvm.mlir.constant(-9223372036854775808 : i64) : i64"
                )
                .unwrap();
                bounds.push(missing);
            }
        }
        let result = self.fresh_value();
        if dynamic {
            writeln!(self.output, "    {result} = llvm.call @__sev_value_slice({object}, {}, {}, {}) : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr", bounds[0], bounds[1], bounds[2]).unwrap();
            return (result, ValueType::Any);
        }
        let function = if object_type == ValueType::String {
            "__sev_string_slice"
        } else {
            "__sev_collection_slice"
        };
        writeln!(self.output, "    {result} = llvm.call @{function}({object}, {}, {}, {}) : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr", bounds[0], bounds[1], bounds[2]).unwrap();
        (result, object_type)
    }

    pub(super) fn lower_conditional_expression(
        &mut self,
        condition: &Expression,
        then_expression: &Expression,
        else_expression: &Expression,
        expected_type: Option<ValueType>,
    ) -> (String, ValueType) {
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
        let mut then_value = self.lower_expression(then_expression);
        let result_type = expected_type.unwrap_or(then_value.1);
        then_value = self.coerce_conditional_value(then_value, result_type);
        writeln!(
            self.output,
            "    llvm.br ^bb{continue_block}({} : {})",
            then_value.0,
            mlir_type(result_type)
        )
        .unwrap();

        writeln!(self.output, "  ^bb{else_block}:").unwrap();
        let else_value = self.lower_expression(else_expression);
        let else_value = self.coerce_conditional_value(else_value, result_type);
        writeln!(
            self.output,
            "    llvm.br ^bb{continue_block}({} : {})",
            else_value.0,
            mlir_type(result_type)
        )
        .unwrap();

        let result = self.fresh_value();
        writeln!(
            self.output,
            "  ^bb{continue_block}({result}: {}):",
            mlir_type(result_type)
        )
        .unwrap();
        (result, result_type)
    }

    pub(super) fn coerce_conditional_value(
        &mut self,
        value: (String, ValueType),
        expected: ValueType,
    ) -> (String, ValueType) {
        if value.1 == expected {
            value
        } else if expected == ValueType::Any {
            (self.box_value(value), ValueType::Any)
        } else if value.1 == ValueType::Any {
            self.unbox_value(value, expected)
        } else {
            value
        }
    }

    pub(super) fn lower_collection_literal(
        &mut self,
        values: &[Expression],
        ty: ValueType,
        kind: i64,
    ) -> (String, ValueType) {
        let kind_value = self.fresh_value();
        writeln!(
            self.output,
            "    {kind_value} = llvm.mlir.constant({kind} : i64) : i64"
        )
        .unwrap();
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @__sev_collection_new({kind_value}) : (i64) -> !llvm.ptr"
        )
        .unwrap();
        for value in values {
            let value = self.lower_expression(value);
            let value = self.box_value(value);
            writeln!(
                self.output,
                "    llvm.call @__sev_collection_push({result}, {value}) : (!llvm.ptr, !llvm.ptr) -> ()"
            )
            .unwrap();
        }
        (result, ty)
    }

    pub(super) fn string_address(&mut self, value: &str) -> String {
        let index = self
            .strings
            .iter()
            .position(|candidate| candidate == value)
            .unwrap_or_else(|| {
                panic!("native metadata string `{value}` was not collected before lowering")
            });
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.mlir.addressof @__sev_str_{index} : !llvm.ptr"
        )
        .unwrap();
        result
    }

    pub(super) fn has_known_class_method(&self, object: &Expression, method: &str) -> bool {
        let Expression::Variable(name) = object.kind() else {
            return false;
        };
        if let Some(class_id) = self
            .variables
            .get(&name.id)
            .and_then(|(value, _)| self.object_class_ids.get(value))
        {
            return self
                .classes
                .iter()
                .find(|class| class.id == *class_id)
                .is_some_and(|class| {
                    class
                        .methods
                        .iter()
                        .any(|candidate| candidate.name == method)
                });
        }
        let class_name = self
            .variables
            .get(&name.id)
            .and_then(|(value, _)| self.object_classes.get(value))
            .or_else(|| self.field_classes.get(&name.name));
        let Some(class_name) = class_name else {
            return false;
        };
        self.classes
            .iter()
            .find(|class| class.name == *class_name)
            .is_some_and(|class| {
                class
                    .methods
                    .iter()
                    .any(|candidate| candidate.name == method)
            })
    }

    pub(super) fn has_abstract_class_method(&self, object: &Expression, method: &str) -> bool {
        let Expression::Variable(name) = object.kind() else {
            return false;
        };
        let Some(receiver) = self
            .variables
            .get(&name.id)
            .and_then(|(value, _)| self.receiver_types.get(value))
        else {
            return false;
        };
        !receiver.concrete && receiver.methods.iter().any(|name| name == method)
    }

    pub(super) fn object_field_metadata(
        &self,
        object: &str,
        field: &str,
    ) -> Option<(ValueType, Option<String>)> {
        self.object_classes
            .get(object)
            .and_then(|class| {
                self.classes
                    .iter()
                    .find(|candidate| candidate.name == *class)
            })
            .and_then(|class| {
                class
                    .fields
                    .iter()
                    .position(|candidate| candidate == field)
                    .map(|index| (class.field_types[index], class.field_classes[index].clone()))
            })
    }

    pub(super) fn box_value(&mut self, (value, ty): (String, ValueType)) -> String {
        let function = match ty {
            ValueType::Int => "__sev_box_i64",
            ValueType::Float => "__sev_box_f64",
            ValueType::Bool => "__sev_box_bool",
            ValueType::String => "__sev_box_string",
            ValueType::List | ValueType::Tuple | ValueType::Set | ValueType::Map => {
                "__sev_box_collection"
            }
            ValueType::Any => return value,
            _ => return value,
        };
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @{function}({value}) : ({}) -> !llvm.ptr",
            mlir_type(ty)
        )
        .unwrap();
        result
    }

    pub(super) fn coerce_resolved_call_arguments(
        &mut self,
        target: &severian_hir::CallTarget,
        arguments: Vec<(String, ValueType)>,
    ) -> Vec<(String, ValueType)> {
        let parameters = target
            .signature
            .as_ref()
            .map(|signature| signature.parameters.clone())
            .or_else(|| self.function_params.get(&target.id).cloned());
        let Some(parameters) = parameters else {
            return arguments;
        };
        arguments
            .into_iter()
            .enumerate()
            .map(|(index, argument)| {
                let Some(expected) = parameters.get(index).copied() else {
                    return argument;
                };
                if argument.1 == ValueType::Any && expected != ValueType::Any {
                    self.unbox_value(argument, expected)
                } else if expected == ValueType::Any && argument.1 != ValueType::Any {
                    (self.box_value(argument), ValueType::Any)
                } else {
                    argument
                }
            })
            .collect()
    }

    pub(super) fn unbox_value(
        &mut self,
        (value, ty): (String, ValueType),
        expected: ValueType,
    ) -> (String, ValueType) {
        if ty != ValueType::Any {
            return (value, ty);
        }
        let function = match expected {
            ValueType::Int => "__sev_unbox_i64",
            ValueType::Float => "__sev_unbox_f64",
            ValueType::Bool => "__sev_unbox_bool",
            ValueType::String => "__sev_unbox_string",
            ValueType::List | ValueType::Tuple | ValueType::Set | ValueType::Map => {
                "__sev_unbox_ptr"
            }
            _ => return (value, ty),
        };
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @{function}({value}) : (!llvm.ptr) -> {}",
            mlir_type(expected)
        )
        .unwrap();
        (result, expected)
    }

    pub(super) fn next_closure_symbol(&self, prefix: &str) -> String {
        let index = self.next_closure.get();
        self.next_closure.set(index + 1);
        format!("__sev_{prefix}_{index}")
    }

    pub(super) fn ensure_function_closure(
        &mut self,
        function: &severian_hir::CallTarget,
    ) -> String {
        if let Some(symbol) = self.function_closures.borrow().get(&function.id).cloned() {
            return symbol;
        }
        let symbol = self.next_closure_symbol("function_closure");
        self.function_closures
            .borrow_mut()
            .insert(function.id, symbol.clone());
        let params = self
            .function_params
            .get(&function.id)
            .cloned()
            .unwrap_or_default();
        let return_type = self
            .function_returns
            .get(&function.id)
            .copied()
            .unwrap_or(ValueType::Any);
        let mut definition = String::new();
        write!(definition, "  llvm.func @{symbol}(%environment: !llvm.ptr").unwrap();
        for index in 0..params.len() {
            write!(definition, ", %arg_{index}: !llvm.ptr").unwrap();
        }
        definition.push_str(") -> !llvm.ptr {\n");
        let mut context = self.callback_context(&mut definition);
        let arguments = params
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                context
                    .unbox_value((format!("%arg_{index}"), ValueType::Any), *ty)
                    .0
            })
            .collect::<Vec<_>>();
        let values = arguments.join(", ");
        let types = params
            .iter()
            .map(|ty| mlir_type(*ty))
            .collect::<Vec<_>>()
            .join(", ");
        let target = self
            .native_symbols
            .get(&function.id)
            .cloned()
            .unwrap_or_else(|| source_function_symbol(&function.name));
        if return_type == ValueType::Unit {
            writeln!(
                context.output,
                "    llvm.call @{target}({values}) : ({types}) -> ()"
            )
            .unwrap();
            let empty = context.fresh_value();
            writeln!(context.output, "    {empty} = llvm.mlir.zero : !llvm.ptr").unwrap();
            writeln!(context.output, "    llvm.return {empty} : !llvm.ptr").unwrap();
        } else {
            let result = context.fresh_value();
            writeln!(
                context.output,
                "    {result} = llvm.call @{target}({values}) : ({types}) -> {}",
                mlir_type(return_type)
            )
            .unwrap();
            let boxed = context.box_value((result, return_type));
            writeln!(context.output, "    llvm.return {boxed} : !llvm.ptr").unwrap();
        }
        definition.push_str("  }\n");
        self.closure_definitions.borrow_mut().push_str(&definition);
        symbol
    }

    pub(super) fn emit_lambda_closure(
        &mut self,
        params: &[BindingRef],
        body: &Expression,
    ) -> String {
        let symbol = self.next_closure_symbol("lambda");
        let mut captures = self
            .variables
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<Vec<_>>();
        captures.sort_by(|left, right| left.0.cmp(&right.0));

        let kind = self.fresh_value();
        writeln!(
            self.output,
            "    {kind} = llvm.mlir.constant(0 : i64) : i64"
        )
        .unwrap();
        let environment = self.fresh_value();
        writeln!(
            self.output,
            "    {environment} = llvm.call @__sev_collection_new({kind}) : (i64) -> !llvm.ptr"
        )
        .unwrap();
        let mut capture_classes = HashMap::new();
        for (name, value) in &captures {
            let boxed = self.box_value(value.clone());
            writeln!(self.output, "    llvm.call @__sev_collection_push({environment}, {boxed}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
            if let Some(class) = self.object_classes.get(&value.0) {
                capture_classes.insert(name.clone(), class.clone());
            }
        }

        let mut definition = String::new();
        write!(definition, "  llvm.func @{symbol}(%environment: !llvm.ptr").unwrap();
        for index in 0..params.len() {
            write!(definition, ", %arg_{index}: !llvm.ptr").unwrap();
        }
        definition.push_str(") -> !llvm.ptr {\n");
        let mut context = self.callback_context(&mut definition);
        for (index, (name, (_, ty))) in captures.iter().enumerate() {
            let position = context.fresh_value();
            writeln!(
                context.output,
                "    {position} = llvm.mlir.constant({index} : i64) : i64"
            )
            .unwrap();
            let raw = context.fresh_value();
            writeln!(context.output, "    {raw} = llvm.call @__sev_collection_get(%environment, {position}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
            let value = context.unbox_value((raw, ValueType::Any), *ty);
            if let Some(class) = capture_classes.get(name) {
                context
                    .object_classes
                    .insert(value.0.clone(), class.clone());
            }
            context.variables.insert(name.clone(), value);
        }
        for (index, param) in params.iter().enumerate() {
            context
                .variables
                .insert(param.id, (format!("%arg_{index}"), ValueType::Any));
        }
        let result = context.lower_expression(body);
        let boxed = context.box_value(result);
        writeln!(context.output, "    llvm.return {boxed} : !llvm.ptr").unwrap();
        definition.push_str("  }\n");
        self.closure_definitions.borrow_mut().push_str(&definition);

        let function = self.fresh_value();
        writeln!(
            self.output,
            "    {function} = llvm.mlir.addressof @{symbol} : !llvm.ptr"
        )
        .unwrap();
        let closure = self.fresh_value();
        writeln!(self.output, "    {closure} = llvm.call @__sev_closure_new({function}, {environment}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
        closure
    }

    pub(super) fn callback_context<'b>(&'b self, output: &'b mut String) -> LowerContext<'b> {
        LowerContext {
            output,
            strings: self.strings,
            function_returns: self.function_returns,
            function_params: self.function_params,
            function_return_classes: self.function_return_classes,
            method_return_classes: self.method_return_classes,
            closure_definitions: Rc::clone(&self.closure_definitions),
            next_closure: Rc::clone(&self.next_closure),
            function_closures: Rc::clone(&self.function_closures),
            native_symbols: self.native_symbols,
            classes: self.classes,
            field_object: None,
            field_names: HashSet::new(),
            field_types: HashMap::new(),
            field_classes: HashMap::new(),
            object_classes: HashMap::new(),
            object_class_ids: HashMap::new(),
            receiver_types: HashMap::new(),
            declared_return: ValueType::Any,
            task_results: HashMap::new(),
            channel_types: HashMap::new(),
            variables: HashMap::new(),
            next_value: 0,
            next_block: 0,
            terminated: false,
            loop_targets: Vec::new(),
            is_main: false,
            placement: TaskPlacement::Default,
        }
    }
}
