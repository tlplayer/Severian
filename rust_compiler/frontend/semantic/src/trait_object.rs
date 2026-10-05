use super::*;
use severian_source::Span;

const PREFIX: &str = "source.trait_object.";

pub(super) fn register_type(
    types: &mut TypeContext,
    identity: &str,
    name: &str,
    arguments: &[TypeId],
    span: Span,
) -> Result<TypeId, Diagnostic> {
    let arguments = arguments
        .iter()
        .map(|ty| ty.0.to_string())
        .collect::<Vec<_>>()
        .join(",");
    types
        .register_source_declaration(format!("{PREFIX}{identity}|{arguments}"), name, 0)
        .map_err(|error| Diagnostic::new("E000204", error.to_string(), Some(span)))
}

impl Analyzer<'_> {
    pub(super) fn trait_type_membership(
        &mut self,
        value: Expression,
        target: TypeId,
        span: Span,
    ) -> Result<Expression, Diagnostic> {
        if let Some(members) = self.union_types.get(&value.type_id).cloned() {
            let boolean = self.types.resolve_name("bool").unwrap();
            return self.map_union_expression(value, &members, boolean, |analyzer, member| {
                analyzer.trait_type_membership(member, target, span)
            });
        }
        if self.is_trait_object(value.type_id) || value.type_id == any_type_id() {
            let identity = self.trait_object_identity(target).unwrap().to_owned();
            let storage = self.trait_object_storage(value);
            return Ok(self.erased_trait_membership(storage, &identity, span));
        }
        let matches = self.trait_object_accepts(value.type_id, target);
        Ok(self.static_type_predicate(value, matches, span))
    }
    fn trait_method_declaration(
        &self,
        ty: TypeId,
        name: &str,
    ) -> Option<(
        Option<severian_modules::ModuleId>,
        severian_ast::FunctionDeclaration,
    )> {
        let identity = self.trait_object_identity(ty)?;
        if let Some(index) = self.package_index {
            return index.definitions.iter().find_map(|(id, definition)| {
                if format!("{:x}:{:x}:{:x}", id.package, id.module, id.declaration.0) != identity {
                    return None;
                }
                let package::DefKind::Trait(declaration) = &definition.kind else {
                    return None;
                };
                declaration
                    .methods
                    .iter()
                    .find(|method| method.name == name)
                    .cloned()
                    .map(|method| (Some(definition.module), method))
            });
        }
        self.trait_declarations
            .get(identity)?
            .methods
            .iter()
            .find(|method| method.name == name)
            .cloned()
            .map(|method| (None, method))
    }

    fn materialize_trait_implementations(
        &mut self,
        identity: &str,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let applications = self
            .types
            .definitions()
            .filter_map(|definition| {
                let (constructor, arguments) = self.types.applied_parts(definition.id)?;
                if self.class_instances_by_type.contains_key(&definition.id)
                    || !self.class_trait_identities(constructor).contains(identity)
                {
                    return None;
                }
                let name = self
                    .generic_class_constructors
                    .iter()
                    .find(|(_, known)| **known == constructor)?
                    .0
                    .clone();
                Some((name, arguments))
            })
            .collect::<Vec<_>>();
        for (name, arguments) in applications {
            self.instantiate_class_types(&name, &arguments, span)?;
        }
        Ok(())
    }

    fn specialize_dynamic_method(
        &mut self,
        method: &severian_ast::FunctionDeclaration,
        types: &[TypeId],
    ) -> Result<severian_ast::FunctionDeclaration, Diagnostic> {
        if method.type_parameters.len() != types.len() {
            return Err(Diagnostic::new(
                "E000206",
                format!(
                    "method `{}` requires {} type arguments",
                    method.name,
                    method.type_parameters.len()
                ),
                Some(method.span),
            ));
        }
        let mut substitution = package::generic::Substitution::default();
        for (parameter, ty) in method.type_parameters.iter().zip(types) {
            let name = format!("__sev_method_type_{}", ty.0);
            self.active_type_aliases.insert(name.clone(), *ty);
            substitution.insert_type(parameter.clone(), name);
        }
        let mut method = package::generic::specialize_function(method, &substitution);
        method.type_parameters.clear();
        if !types.is_empty() {
            method.name = format!(
                "{}${}",
                method.name,
                types
                    .iter()
                    .map(|ty| ty.0.to_string())
                    .collect::<Vec<_>>()
                    .join("_")
            );
        }
        Ok(method)
    }

    pub(super) fn dynamic_method_call(
        &mut self,
        callee: &AstExpression,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: Span,
    ) -> Result<Option<Expression>, Diagnostic> {
        let (callee, explicit) = match &callee.kind {
            AstExpressionKind::TypeApplication { callee, arguments } => {
                (callee.as_ref(), arguments.as_slice())
            }
            _ => (callee, &[][..]),
        };
        let AstExpressionKind::Member { object, name } = &callee.kind else {
            return Ok(None);
        };
        if callable_path(callee).is_some_and(|path| self.functions.contains_key(&path)) {
            return Ok(None);
        }
        let Ok(object) = self.expression(object, None) else {
            return Ok(None);
        };
        if !self.is_trait_object(object.type_id) && explicit.is_empty() {
            return Ok(None);
        }
        let types = explicit
            .iter()
            .map(|annotation| self.resolve_source_type(annotation))
            .collect::<Result<Vec<_>, _>>()?;
        if !self.is_trait_object(object.type_id) {
            let Some(owner) = self.class_instances_by_type.get(&object.type_id).cloned() else {
                return Ok(None);
            };
            let Some(method) = owner.methods.iter().find(|method| method.name == *name) else {
                return Ok(None);
            };
            let method = self.specialize_dynamic_method(method, &types)?;
            return self
                .lower_method_callable(&owner, &method, object, arguments, expected, span)
                .map(Some);
        }
        let (origin, method) = self
            .trait_method_declaration(object.type_id, name)
            .ok_or_else(|| {
                Diagnostic::new(
                    "E000211",
                    format!("trait has no method `{name}`"),
                    Some(span),
                )
            })?;
        let method = self.specialize_dynamic_method(&method, &types)?;
        let previous_module = self.type_resolution_module;
        self.type_resolution_module = origin;
        let signature = (|| {
            let parameters = method
                .parameters
                .iter()
                .map(|parameter| {
                    Ok(SignatureParameter {
                        name: parameter.name.clone(),
                        type_id: self.resolve_source_type(&parameter.annotation)?,
                        variadic_element: None,
                        default: parameter.default.clone(),
                    })
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            Ok::<_, Diagnostic>(FunctionSignature {
                parameters,
                result: self.resolve_source_type(&method.result)?,
            })
        })();
        self.type_resolution_module = previous_module;
        let signature = signature?;
        let Some((values, _, _)) = self.resolve_signature_arguments(&signature, arguments, span)?
        else {
            return Err(Diagnostic::new(
                "E000206",
                format!("trait method `{name}` does not accept these arguments"),
                Some(span),
            ));
        };
        let trait_type = object.type_id;
        let identity = self.trait_object_identity(trait_type).unwrap().to_owned();
        self.materialize_trait_implementations(&identity, span)?;
        let mut parameters = vec![trait_type];
        parameters.extend(
            signature
                .parameters
                .iter()
                .map(|parameter| parameter.type_id),
        );
        let symbol = format!("__sev_trait_method_{}_{}", trait_type.0, method.name);
        let definition = self.ensure_runtime_function(&symbol, &parameters, signature.result);
        let helper = self
            .runtime_functions
            .iter()
            .find(|function| function.definition == definition)
            .unwrap();
        if helper.body.is_none() {
            let bindings = helper
                .parameters
                .iter()
                .map(|parameter| parameter.binding)
                .collect::<Vec<_>>();
            let values = parameters
                .iter()
                .zip(bindings)
                .map(|(ty, binding)| Expression {
                    id: self.next_id(),
                    type_id: *ty,
                    span,
                    kind: ExpressionKind::Binding(binding),
                })
                .collect::<Vec<_>>();
            let storage = self.trait_object_storage(values[0].clone());
            let previous_values = self.value_substitutions.clone();
            let call_arguments = values[1..]
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let name = format!("__sev_trait_argument_{index}");
                    self.value_substitutions.insert(name.clone(), value.clone());
                    severian_ast::CallArgument {
                        name: None,
                        spread: false,
                        expected_error: None,
                        span,
                        value: AstExpression {
                            kind: AstExpressionKind::Name(name),
                            span,
                        },
                    }
                })
                .collect::<Vec<_>>();
            let candidates = self
                .class_instances_by_type
                .values()
                .filter(|instance| self.class_trait_identities(instance.ty).contains(&identity))
                .cloned()
                .collect::<Vec<_>>();
            let mut selected = self.throw_expression(
                "trait receiver has no matching method implementation",
                signature.result,
                span,
            );
            for owner in candidates {
                let Some(implementation) = owner.methods.iter().find(|method| method.name == *name)
                else {
                    continue;
                };
                let implementation = self.specialize_dynamic_method(implementation, &types)?;
                let receiver = self.erased_record_read(storage.clone(), owner.ty, span);
                let value = self.lower_method_callable(
                    &owner,
                    &implementation,
                    receiver,
                    &call_arguments,
                    None,
                    span,
                )?;
                let value = self.coerce(value, signature.result, false)?;
                let condition = self.erased_record_is(storage.clone(), owner.ty, span);
                selected = Expression {
                    id: self.next_id(),
                    type_id: signature.result,
                    span,
                    kind: ExpressionKind::Fallback {
                        condition: Box::new(condition),
                        value: Box::new(value),
                        fallback: Box::new(selected),
                    },
                };
            }
            self.value_substitutions = previous_values;
            let helper = self
                .runtime_functions
                .iter_mut()
                .find(|function| function.definition == definition)
                .unwrap();
            helper.call_type = CallType::Severian;
            helper.body = Some(Block {
                statements: vec![Statement::Return(Some(selected))],
            });
        }
        let mut arguments = vec![object];
        arguments.extend(values);
        let value = self.runtime_call(&symbol, &parameters, signature.result, arguments, span);
        if let Some(expected) = expected {
            return self.coerce(value, expected, false).map(Some);
        }
        Ok(Some(value))
    }

    pub(super) fn trait_object_is_error(&self, ty: TypeId) -> bool {
        let Some(identity) = self.trait_object_identity(ty) else {
            return false;
        };
        if let Some(index) = self.package_index {
            return index.definitions.iter().any(|(id, definition)| {
                matches!(definition.kind, package::DefKind::Trait(_))
                    && format!("{:x}:{:x}:{:x}", id.package, id.module, id.declaration.0)
                        == identity
                    && (definition.name == "Error"
                        || package::trait_identities(index, definition.module, &definition.name)
                            .iter()
                            .any(|base| {
                                index.definitions.iter().any(|(id, definition)| {
                                    definition.name == "Error"
                                        && format!(
                                            "{:x}:{:x}:{:x}",
                                            id.package, id.module, id.declaration.0
                                        ) == *base
                                })
                            }))
            });
        }
        identity == "Error"
            || self
                .trait_declarations
                .get(identity)
                .is_some_and(|declaration| {
                    declaration
                        .bases
                        .iter()
                        .any(|base| base.simple_name() == Some("Error"))
                })
    }

    pub(super) fn union_guard_projection(
        &mut self,
        condition: &AstExpression,
        allow_mutable: bool,
    ) -> Result<Option<(String, Option<Expression>, Option<Expression>)>, Diagnostic> {
        if let AstExpressionKind::Unary {
            operator: AstUnaryOperator::Not,
            operand,
        } = &condition.kind
        {
            return self
                .union_guard_projection(operand, allow_mutable)
                .map(|guard| guard.map(|(name, yes, no)| (name, no, yes)));
        }
        let AstExpressionKind::Binary {
            operator,
            left,
            right,
        } = &condition.kind
        else {
            return Ok(None);
        };
        let Some(name) = callable_path(left) else {
            return Ok(None);
        };
        if let AstExpressionKind::Name(name) = &left.kind {
            if !allow_mutable && self
                .names
                .get(name)
                .is_some_and(|(_, variable, _)| self.mutable_variables.contains(variable))
            {
                return Ok(None);
            }
        }
        let is_none = matches!(&right.kind, AstExpressionKind::Literal(AstLiteral::None))
            || matches!(&right.kind, AstExpressionKind::Name(name) if name == "absent");
        let positive = match *operator {
            AstBinaryOperator::Identity | AstBinaryOperator::Equal if is_none => true,
            AstBinaryOperator::NotEqual if is_none => false,
            AstBinaryOperator::Identity => true,
            _ => return Ok(None),
        };
        let target = if is_none {
            self.types.resolve_name("None").unwrap()
        } else {
            let Some(target) = callable_path(right) else {
                return Ok(None);
            };
            self.resolve_source_type(&TypeAnnotation::named(target, Vec::new(), right.span))?
        };
        let value = self.expression(left, None)?;
        if self.is_trait_object(value.type_id) || value.type_id == any_type_id() {
            let narrowed = if self.is_trait_object(target) {
                self.wrap_trait_object(value, target, true)?
            } else {
                let storage = self.trait_object_storage(value);
                self.erased_record_read(storage, target, left.span)
            };
            return Ok(Some(if positive {
                (name, Some(narrowed), None)
            } else {
                (name, None, Some(narrowed))
            }));
        }
        let Some(members) = self.union_types.get(&value.type_id).cloned() else {
            return Ok(None);
        };
        if Some(target) != self.types.resolve_name("None")
            && members.iter().any(|member| self.is_trait_object(*member) || *member == any_type_id()) {
            let yes = self.map_union_expression(value.clone(), &members, target, |analyzer, member| {
                if member.type_id == target || analyzer.trait_object_accepts(member.type_id, target) {
                    analyzer.coerce(member, target, false)
                } else if analyzer.is_trait_object(member.type_id) || member.type_id == any_type_id() {
                    if analyzer.is_trait_object(target) {
                        analyzer.wrap_trait_object(member, target, true)
                    } else {
                        let span = member.span;
                        let storage = analyzer.trait_object_storage(member);
                        Ok(analyzer.erased_record_read(storage, target, span))
                    }
                } else {
                    Ok(analyzer.throw_expression("unreachable union member after a type guard", target, member.span))
                }
            })?;
            // A failed dynamic test does not exclude the entire trait member.
            let remaining = members.iter().copied().filter(|member|
                *member != target && !self.trait_object_accepts(*member, target)).collect::<Vec<_>>();
            let no = self.union_subset_projection(value, &members, &remaining)?;
            return Ok(Some(if positive { (name, Some(yes), no) } else { (name, no, Some(yes)) }));
        }
        let (yes, no): (Vec<_>, Vec<_>) = members
            .iter()
            .copied()
            .partition(|member| *member == target || self.trait_object_accepts(*member, target));
        let yes = self.union_subset_projection(value.clone(), &members, &yes)?;
        let no = self.union_subset_projection(value, &members, &no)?;
        Ok(Some(if positive {
            (name, yes, no)
        } else {
            (name, no, yes)
        }))
    }

    fn union_subset_projection(
        &mut self,
        value: Expression,
        members: &[TypeId],
        selected: &[TypeId],
    ) -> Result<Option<Expression>, Diagnostic> {
        if selected.is_empty() {
            return Ok(None);
        }
        if selected == members {
            return Ok(Some(value));
        }
        if let [ty] = selected {
            return Ok(Some(Expression {
                id: self.next_id(),
                type_id: *ty,
                span: value.span,
                kind: ExpressionKind::Field {
                    object: Box::new(value),
                    index: members.iter().position(|member| member == ty).unwrap() as u32 + 1,
                },
            }));
        }
        let ty = self.instantiate_union_type(selected);
        self.map_union_expression(value, members, ty, |analyzer, member| {
            if selected.contains(&member.type_id) {
                analyzer.coerce(member, ty, false)
            } else {
                Ok(analyzer.throw_expression(
                    "unreachable union member after a type guard",
                    ty,
                    member.span,
                ))
            }
        })
        .map(Some)
    }

    pub(super) fn static_type_condition(&self, condition: &Expression) -> Option<bool> {
        let ExpressionKind::Call {
            callee: severian_hir::Callee::Direct { function, .. },
            ..
        } = &condition.kind
        else {
            return None;
        };
        let helper = self.runtime_functions.iter().find(|candidate| {
            candidate.definition == *function
                && candidate.name.starts_with("__sev_static_type_predicate_")
        })?;
        let [Statement::Return(Some(Expression {
            kind: ExpressionKind::Literal(LiteralValue::Boolean(value)),
            ..
        }))] = helper.body.as_ref()?.statements.as_slice()
        else {
            return None;
        };
        Some(*value)
    }
    pub(super) fn is_trait_object(&self, ty: TypeId) -> bool {
        self.types
            .definition(ty)
            .is_some_and(|definition| definition.path.starts_with(PREFIX))
    }

    fn trait_object_identity(&self, ty: TypeId) -> Option<&str> {
        self.types
            .definition(ty)?
            .path
            .strip_prefix(PREFIX)?
            .split_once('|')
            .map(|(identity, _)| identity)
    }

    pub(super) fn install_trait_object_layouts(&mut self) -> Result<(), Diagnostic> {
        let types = self
            .types
            .definitions()
            .filter(|definition| definition.path.starts_with(PREFIX))
            .map(|definition| definition.id)
            .collect::<Vec<_>>();
        for ty in types {
            self.ensure_trait_object_layout(ty);
        }
        Ok(())
    }

    pub(super) fn ensure_trait_object_layout(&mut self, ty: TypeId) {
        if self.lowered_classes.iter().any(|class| class.id == ty) {
            return;
        }
        let any = self.ensure_any_type();
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: self.types.definition(ty).unwrap().path.clone(),
            variants: Vec::new(),
            fields: vec![HirClassFieldDeclaration {
                name: "__value".into(),
                ty: any,
            }],
        });
    }

    pub(super) fn trait_object_storage(&mut self, value: Expression) -> Expression {
        if !self.is_trait_object(value.type_id) {
            return value;
        }
        self.ensure_trait_object_layout(value.type_id);
        Expression {
            id: self.next_id(),
            type_id: self.ensure_any_type(),
            span: value.span,
            kind: ExpressionKind::Field {
                object: Box::new(value),
                index: 0,
            },
        }
    }

    pub(super) fn trait_object_accepts(&self, actual: TypeId, expected: TypeId) -> bool {
        let Some(identity) = self.trait_object_identity(expected) else {
            return false;
        };
        if actual == expected {
            return true;
        }
        let arguments = self
            .types
            .definition(expected)
            .unwrap()
            .path
            .split_once('|')
            .unwrap()
            .1;
        let arguments = arguments
            .split(',')
            .filter(|argument| !argument.is_empty())
            .map(|argument| TypeId(argument.parse().unwrap()))
            .collect::<Vec<_>>();
        if let Some(actual_identity) = self.trait_object_identity(actual) {
            // An erased trait family is not evidence that its type arguments
            // agree. Distinct instantiations retain distinct contracts.
            if actual_identity == identity || !arguments.is_empty() {
                return false;
            }
            if let Some(index) = self.package_index {
                return index.definitions.iter().any(|(id, definition)| {
                    matches!(definition.kind, package::DefKind::Trait(_))
                        && format!("{:x}:{:x}:{:x}", id.package, id.module, id.declaration.0)
                            == actual_identity
                        && package::trait_identities(index, definition.module, &definition.name)
                            .contains(identity)
                });
            }
            return actual_identity == identity;
        }
        if !arguments.is_empty() {
            let Some(instance) = self.class_instances_by_type.get(&actual) else {
                return false;
            };
            let Some(declaration) = self.classes.get(&instance.name) else {
                return false;
            };
            let aliases = declaration
                .type_parameters
                .iter()
                .cloned()
                .zip(instance.arguments.iter().copied())
                .collect::<BTreeMap<_, _>>();
            return declaration.traits.iter().any(|annotation| {
                let Some((name, implemented)) = annotation.named_parts() else {
                    return false;
                };
                let implementation_identity = self
                    .package_index
                    .zip(
                        self.class_defining_modules
                            .get(&actual)
                            .copied()
                            .or(self.type_resolution_module),
                    )
                    .and_then(|(index, module)| package::trait_identity(index, module, name))
                    .unwrap_or_else(|| name.to_owned());
                implementation_identity == identity
                    && implemented.len() == arguments.len()
                    && implemented
                        .iter()
                        .zip(&arguments)
                        .all(|(annotation, expected)| {
                            annotation.simple_name().and_then(|name| {
                                aliases
                                    .get(name)
                                    .copied()
                                    .or_else(|| self.types.resolve_name(name))
                            }) == Some(*expected)
                        })
            });
        }
        self.class_trait_identities(actual).contains(identity)
    }

    pub(super) fn wrap_trait_object(
        &mut self,
        value: Expression,
        expected: TypeId,
        explicit: bool,
    ) -> Result<Expression, Diagnostic> {
        let span = value.span;
        if !self.trait_object_accepts(value.type_id, expected) && !explicit {
            return Err(Diagnostic::new(
                "E000204",
                "value does not implement the required trait",
                Some(span),
            )
            .with_help(format!(
                "implement `{}` on this value's declaration",
                self.types.definition(expected).unwrap().name
            )));
        }
        self.ensure_trait_object_layout(expected);
        let value = self.box_any_value(value, span)?;
        let result = Expression {
            id: self.next_id(),
            type_id: expected,
            span,
            kind: ExpressionKind::Aggregate {
                class: expected,
                fields: vec![value.clone()],
            },
        };
        if !explicit {
            return Ok(result);
        }
        let identity = self.trait_object_identity(expected).unwrap().to_owned();
        let condition = self.erased_trait_membership(value, &identity, span);
        let failure = self.throw_expression(
            "value does not implement the required trait",
            expected,
            span,
        );
        Ok(Expression {
            id: self.next_id(),
            type_id: expected,
            span,
            kind: ExpressionKind::Fallback {
                condition: Box::new(condition),
                value: Box::new(result),
                fallback: Box::new(failure),
            },
        })
    }

    fn erased_trait_membership(
        &mut self,
        value: Expression,
        identity: &str,
        span: Span,
    ) -> Expression {
        let any = self.ensure_any_type();
        let string = self.types.resolve_name("string").unwrap();
        let boolean = self.types.resolve_name("bool").unwrap();
        let identity = self.string_expression(format!("|{identity}|"), span);
        let primitive = Expression {
            id: self.next_id(),
            type_id: boolean,
            span,
            kind: ExpressionKind::Literal(LiteralValue::Boolean(false)),
        };
        self.runtime_call(
            "__sev_any_implements",
            &[any, string, boolean],
            boolean,
            vec![value, identity, primitive],
            span,
        )
    }

    fn trait_property(
        &self,
        ty: TypeId,
        name: &str,
    ) -> Option<(
        Option<severian_modules::ModuleId>,
        severian_ast::PropertyDeclaration,
    )> {
        let identity = self.trait_object_identity(ty)?;
        if let Some(index) = self.package_index {
            let mut pending = index
                .definitions
                .iter()
                .filter_map(|(id, definition)| {
                    (matches!(definition.kind, package::DefKind::Trait(_))
                        && format!("{:x}:{:x}:{:x}", id.package, id.module, id.declaration.0)
                            == identity)
                        .then_some(*id)
                })
                .collect::<Vec<_>>();
            let mut seen = BTreeSet::new();
            while let Some(id) = pending.pop() {
                if !seen.insert(id) {
                    continue;
                }
                let definition = &index.definitions[&id];
                let package::DefKind::Trait(declaration) = &definition.kind else {
                    continue;
                };
                if let Some(field) = declaration
                    .properties
                    .iter()
                    .find(|field| field.name == name)
                {
                    return Some((Some(definition.module), field.clone()));
                }
                for base in &declaration.bases {
                    if let Some((name, _)) = base.named_parts() {
                        pending.extend(package::resolve_trait_definitions(
                            index,
                            definition.module,
                            name,
                        ));
                    }
                }
            }
            return None;
        }
        let mut pending = vec![identity.to_owned()];
        let mut seen = BTreeSet::new();
        while let Some(name_of_trait) = pending.pop() {
            if !seen.insert(name_of_trait.clone()) {
                continue;
            }
            let declaration = self.trait_declarations.get(&name_of_trait)?;
            if let Some(field) = declaration
                .properties
                .iter()
                .find(|field| field.name == name)
            {
                return Some((None, field.clone()));
            }
            pending.extend(
                declaration
                    .bases
                    .iter()
                    .filter_map(|base| base.simple_name().map(str::to_owned)),
            );
        }
        None
    }

    pub(super) fn trait_object_field(
        &mut self,
        object: Expression,
        name: &str,
        expected: Option<TypeId>,
        span: Span,
    ) -> Result<Expression, Diagnostic> {
        let (module, property) = self.trait_property(object.type_id, name).ok_or_else(|| {
            Diagnostic::new(
                "E000211",
                format!("trait has no declared field `{name}`"),
                Some(span),
            )
        })?;
        let previous = self.type_resolution_module;
        self.type_resolution_module = module;
        let result = self.resolve_source_type(&property.annotation);
        self.type_resolution_module = previous;
        let result = result?;
        if expected.is_some_and(|expected| !self.accepts_expression_type(result, expected)) {
            return Err(Diagnostic::new(
                "E000204",
                "trait field does not satisfy the expected type",
                Some(span),
            ));
        }
        let trait_type = object.type_id;
        let identity = self.trait_object_identity(trait_type).unwrap().to_owned();
        let symbol = format!("__sev_trait_field_{}_{}", trait_type.0, name);
        let definition = self.ensure_runtime_function(&symbol, &[trait_type], result);
        let function = self
            .runtime_functions
            .iter()
            .find(|function| function.definition == definition)
            .unwrap();
        if function.body.is_none() {
            let binding = function.parameters[0].binding;
            let receiver = Expression {
                id: self.next_id(),
                type_id: trait_type,
                span,
                kind: ExpressionKind::Binding(binding),
            };
            let storage = self.trait_object_storage(receiver);
            let mut selected = self.throw_expression(
                "trait receiver has no matching field implementation",
                result,
                span,
            );
            let candidates = self
                .class_instances_by_type
                .values()
                .filter(|instance| self.class_trait_identities(instance.ty).contains(&identity))
                .cloned()
                .collect::<Vec<_>>();
            for instance in candidates {
                let Some((index, field)) = instance
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, field)| field.name == name)
                else {
                    continue;
                };
                if field.ty != result {
                    continue;
                }
                let condition = self.erased_record_is(storage.clone(), instance.ty, span);
                let receiver = self.erased_record_read(storage.clone(), instance.ty, span);
                let value = Expression {
                    id: self.next_id(),
                    type_id: result,
                    span,
                    kind: ExpressionKind::Field {
                        object: Box::new(receiver),
                        index: index as u32,
                    },
                };
                selected = Expression {
                    id: self.next_id(),
                    type_id: result,
                    span,
                    kind: ExpressionKind::Fallback {
                        condition: Box::new(condition),
                        value: Box::new(value),
                        fallback: Box::new(selected),
                    },
                };
            }
            let function = self
                .runtime_functions
                .iter_mut()
                .find(|function| function.definition == definition)
                .unwrap();
            function.call_type = CallType::Severian;
            function.body = Some(Block {
                statements: vec![Statement::Return(Some(selected))],
            });
        }
        Ok(self.runtime_call(&symbol, &[trait_type], result, vec![object], span))
    }

    pub(super) fn erased_record_is(
        &mut self,
        value: Expression,
        ty: TypeId,
        span: Span,
    ) -> Expression {
        let any = self.ensure_any_type();
        let integer = self.tag_type();
        let boolean = self.types.resolve_name("bool").unwrap();
        let key = self.integer_expression(&ty.0.to_string(), integer, span);
        self.runtime_call(
            "__sev_any_record_is",
            &[any, integer],
            boolean,
            vec![value, key],
            span,
        )
    }

    pub(super) fn erased_record_read(
        &mut self,
        value: Expression,
        ty: TypeId,
        span: Span,
    ) -> Expression {
        let any = self.ensure_any_type();
        let integer = self.tag_type();
        let key = self.integer_expression(&ty.0.to_string(), integer, span);
        self.runtime_call(
            "__sev_any_record_read_aggregate",
            &[any, integer],
            ty,
            vec![value, key],
            span,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trait_type_arguments_are_not_erased_by_assignment() {
        let context = severian_bootstrap::load().unwrap();
        for (text, valid) in [
            ("trait Holder[T]:\n    value: T\nclass Integers: Holder[int]\n    value: int\ndef make() -> Holder[int]:\n    return Integers(7)\n", true),
            ("trait Holder[T]:\n    value: T\ndef invalid(value: Holder[string]) -> Holder[int]:\n    return value\n", false),
            ("trait Holder[T]:\n    value: T\nclass Strings: Holder[string]\n    value: string\ndef invalid() -> Holder[int]:\n    return Strings(\"wrong\")\n", false),
        ] {
            let source = severian_source::SourceFile::virtual_source("trait-arguments.sev", text);
            let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
            let result = analyze(&ast, &context.types);
            assert_eq!(result.is_ok(), valid, "{result:?}");
        }
    }

    #[test]
    fn negative_trait_field_guard_preserves_the_concrete_contract_on_the_surviving_path() {
        let source = severian_source::SourceFile::virtual_source("trait-field-guard.sev", "trait Named:\n    label: string\nclass Specific: Named\n    label: string\n    count: int\nclass Holder:\n    item: Named\ndef read(holder: Holder) -> int:\n    if not (holder.item is Specific):\n        return 0\n    return holder.item.count\n");
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let context = severian_bootstrap::load().unwrap();
        let program = analyze(&ast, &context.types).unwrap();
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn empty_collection_selects_the_list_member_of_a_diagnostic_union() {
        let source = severian_source::SourceFile::virtual_source("empty-result.sev", "trait Diagnostic: Error\n    note: string\ndef values() -> list[int] | Diagnostic:\n    return []\n");
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let context = severian_bootstrap::load().unwrap();
        let program = analyze(&ast, &context.types).unwrap();
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn generic_trait_methods_keep_payload_and_method_type_arguments() {
        let source = severian_source::SourceFile::virtual_source("trait-method.sev", "trait Readable:\n    def read[R](fallback: R) -> R\nclass Holder[T]: Readable\n    value: T\n    def read[R](fallback: R) -> R:\n        if self.value is R:\n            return self.value\n        return fallback\ndef read() -> int:\n    input: Readable = Holder[int](7)\n    return input.read[int](0)\n");
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let context = severian_bootstrap::load().unwrap();
        let program = analyze(&ast, &context.types).unwrap();
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn optional_trait_guard_does_not_survive_parameter_reassignment() {
        let source = severian_source::SourceFile::virtual_source("mutation.sev", "trait B:\n    name: string\ndef invalid(value: B | None) -> B:\n    if value is B:\n        value = None\n        return value\n    throw Error(\"missing\")\n");
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let context = severian_bootstrap::load().unwrap();
        assert!(analyze(&ast, &context.types).is_err());
    }

    #[test]
    fn diagnostic_guards_narrow_three_member_unions_before_throw_and_field_access() {
        let source = severian_source::SourceFile::virtual_source("diagnostic-guard.sev", "trait Diagnostic: Error\n    note: string\nclass Value:\n    count: int\ndef read(input: Value | Diagnostic | None) -> int:\n    value = input\n    if value == None:\n        throw Error(\"missing value\")\n    if value is Diagnostic:\n        throw value\n    return value.count\n");
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let context = severian_bootstrap::load().unwrap();
        let program = analyze(&ast, &context.types).unwrap();
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn trait_return_retains_declared_field_contract() {
        let source = severian_source::SourceFile::virtual_source("trait-field.sev",
            "trait Named:\n    label: string\nclass First: Named\n    label: string\nclass Second: Named\n    count: int\n    label: string\ndef choose(flag: bool) -> Named:\n    if flag:\n        return First(\"first\")\n    return Second(7, \"second\")\ndef read(flag: bool) -> string:\n    return choose(flag).label\n");
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let context = severian_bootstrap::load().unwrap();
        let (program, types) = analyze_with_context_and_types(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Build,
                module_name: "trait-field",
            },
        )
        .unwrap();
        let choose = program.modules[0]
            .functions
            .iter()
            .find(|function| function.name == "choose")
            .unwrap();
        assert_ne!(choose.result.ty, any_type_id());
        assert!(types
            .definition(choose.result.ty)
            .unwrap()
            .path
            .starts_with(PREFIX));
        severian_mir::build(&program).unwrap();
    }
}
