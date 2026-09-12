use super::*;

impl Analyzer<'_> {
    pub(super) fn class_needs_destruction(&self, ty: TypeId, visiting: &mut BTreeSet<TypeId>) -> bool {
        if self.types.tensor(ty).is_some() { return false; }
        if self.types.resolve_name("string") == Some(ty) || self.any_type == Some(ty) { return true; }
        if !visiting.insert(ty) { return false; }
        self.class_instances_by_type.get(&ty).is_some_and(|class|
            class.methods.iter().any(|method| method.name == "drop" && method.parameters.is_empty() && method.result.simple_name() == Some("unit")))
            || self.lowered_classes.iter().find(|class| class.id == ty).is_some_and(|class|
                class.fields.iter().any(|field| self.class_needs_destruction(field.ty, visiting)))
    }

    pub(super) fn register_class_destruction(&mut self) -> Result<(), Diagnostic> {
        let mut owners = self.lowered_classes.iter().map(|class| class.id).collect::<Vec<_>>();
        if let Some(string) = self.types.resolve_name("string") { owners.push(string); }
        for owner in owners {
            if self.class_needs_destruction(owner, &mut BTreeSet::new()) {
                self.class_destruction(owner)?;
                if !self.class_is_affine(owner, &mut BTreeSet::new()) {
                    self.class_storage_glue(owner, true)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn class_is_affine(&self, ty: TypeId, visiting: &mut BTreeSet<TypeId>) -> bool {
        if !visiting.insert(ty) { return false; }
        self.class_instances_by_type.get(&ty).is_some_and(|class|
            class.methods.iter().any(|method| method.name == "drop" && method.parameters.is_empty() && method.result.simple_name() == Some("unit")))
            || self.lowered_classes.iter().find(|class| class.id == ty).is_some_and(|class|
                class.fields.iter().any(|field| self.class_is_affine(field.ty, visiting)))
    }

    fn class_destruction(&mut self, ty: TypeId) -> Result<DefId, Diagnostic> {
        self.class_storage_glue(ty, false)
    }

    fn class_storage_glue(&mut self, ty: TypeId, retain: bool) -> Result<DefId, Diagnostic> {
        let known = if retain { self.types.retention(ty) } else { self.types.destruction(ty) };
        if let Some(definition) = known { return Ok(definition); }
        let prefix = if retain { "__sev_retain_type" } else { "__sev_destroy_type" };
        let definition = synthetic_runtime_definition(&format!("{prefix}{}", ty.0));
        if retain { self.types.register_retention(ty, definition); }
        else { self.types.register_destruction(ty, definition); }
        let unit = self.types.resolve_name("unit").unwrap();
        let binding = self.new_binding_id();
        let span = severian_source::Span::new(severian_source::SourceId(0), 0, 0);
        let receiver = Expression { id: self.next_id(), type_id: ty,
            kind: ExpressionKind::Binding(binding), span };
        if self.types.resolve_name("string") == Some(ty) || self.any_type == Some(ty) {
            let symbol = if self.any_type == Some(ty) {
                if retain { "__sev_any_retain" } else { "__sev_any_release" }
            } else if retain { "__sev_storage_retain" } else { "__sev_storage_release" };
            let call = self.runtime_call(symbol, &[ty], unit, vec![receiver], span);
            self.register_storage_glue(ty, definition, binding, prefix,
                Block { statements: vec![Statement::Expression(call), Statement::Return(None)] });
            return Ok(definition);
        }
        let record = self.lowered_classes.iter().find(|class| class.id == ty).unwrap().clone();
        let owner = self.class_instances_by_type.get(&ty).cloned().unwrap_or(ClassInstance {
            ty, name: record.name.clone(), arguments: vec![], fields: record.fields.clone(),
            source_fields: vec![], constructors: vec![], methods: vec![], operators: vec![],
        });
        self.types.register_destruction_fields(ty, owner.fields.iter().map(|field| field.ty).collect(),
            owner.methods.iter().any(|method| method.name == "drop" && method.parameters.is_empty() && method.result.simple_name() == Some("unit")));
        let mut body = Block::default();
        if let Some(method) = owner.methods.iter().find(|method| !retain && method.name == "drop" && method.parameters.is_empty() && method.result.simple_name() == Some("unit")) {
            if !method.parameters.is_empty() || method.result.simple_name() != Some("unit") {
                return Err(Diagnostic::new("E000221", "destructor must take only self and return non-throwing unit", Some(method.span)));
            }
            body.statements.push(Statement::Expression(self.lower_method_callable(
                &owner, method, receiver.clone(), &[], Some(unit), span)?));
        }
        for (index, field) in owner.fields.iter().enumerate().rev() {
            if !self.class_needs_destruction(field.ty, &mut BTreeSet::new()) { continue; }
            let child = self.class_storage_glue(field.ty, retain)?;
            let value = Expression { id: self.next_id(), type_id: field.ty,
                kind: ExpressionKind::Field { object: Box::new(receiver.clone()), index: index as u32 }, span };
            let borrowed = Expression { id: self.next_id(), type_id: field.ty,
                kind: ExpressionKind::Borrow { operand: Box::new(value.clone()), exclusive: false }, span };
            let cleanup = Statement::Expression(Expression {
                id: self.next_id(), type_id: unit, span,
                kind: ExpressionKind::Call { callee: severian_hir::Callee::Direct {
                    instance: Some(FunctionId(child.declaration.0)), function: child,
                    substitution: Default::default() }, arguments: vec![if self.types.primitive(field.ty).is_some() { value } else { borrowed }], evaluation_order: vec![0] },
            });
            if record.variants.is_empty() {
                body.statements.push(cleanup);
            } else if let Some(variant) = record.variants.iter().position(|fields| fields.contains(&(index as u32))) {
                let tag_type = record.fields[0].ty;
                let tag = Expression { id: self.next_id(), type_id: tag_type, span,
                    kind: ExpressionKind::Field { object: Box::new(receiver.clone()), index: 0 } };
                let expected = if self.types.resolve_name("bool") == Some(tag_type) {
                    Expression { id: self.next_id(), type_id: tag_type, span,
                        kind: ExpressionKind::Literal(LiteralValue::Boolean(variant != 0)) }
                } else { self.integer_expression(&variant.to_string(), tag_type, span) };
                let condition = Expression { id: self.next_id(), type_id: self.types.resolve_name("bool").unwrap(), span,
                    kind: ExpressionKind::Binary { operator: BinaryOperator::Equal, left: Box::new(tag), right: Box::new(expected) } };
                body.statements.push(Statement::If { condition,
                    then_block: Block { statements: vec![cleanup] }, else_block: Block::default() });
            }
        }
        body.statements.push(Statement::Return(None));
        self.register_storage_glue(ty, definition, binding, prefix, body);
        Ok(definition)
    }

    fn register_storage_glue(&mut self, ty: TypeId, definition: DefId, binding: BindingId, prefix: &str, body: Block) {
        let unit = self.types.resolve_name("unit").unwrap();
        let id = FunctionId(definition.declaration.0);
        self.function_definitions.insert(id, definition);
        self.function_substitutions.insert(id, Default::default());
        self.parameter_effects.insert(id, vec![ParameterEffect::Shared]);
        self.runtime_functions.push(FunctionDeclaration {
            id, definition, substitution: Default::default(), name: format!("{prefix}{}", ty.0),
            generic_parameters: vec![], type_parameters: vec![],
            parameters: vec![FunctionParameter { binding, name: "self".into(), contract: universal_boundary(ty) }],
            result: universal_boundary(unit), compile_route: severian_universal::CompileRoute::Standard,
            call_type: CallType::Severian, body: Some(body),
        });
    }
}
