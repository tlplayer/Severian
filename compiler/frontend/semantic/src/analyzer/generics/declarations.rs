use super::*;

impl Specializer {
    pub(super) fn rewrite_item(
        &mut self,
        item: &mut Item,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        match item {
            Item::Function(function) => self.rewrite_function(function, context),
            Item::Class(class) => self.rewrite_class(class, context),
            Item::Trait(trait_) => self.rewrite_trait(trait_, context),
            Item::Enum(enumeration) => {
                for variant in &mut enumeration.variants {
                    for field in &mut variant.fields {
                        self.rewrite_parameter(field, context)?;
                    }
                }
                Ok(())
            }
            Item::Statement(statement) => self.rewrite_statement(statement, context),
            Item::Import(_) => Ok(()),
        }
    }

    fn rewrite_trait(
        &mut self,
        trait_: &mut severian_ast::TraitDecl,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        let context = context.with_generics(&trait_.generic_params);
        for composed in &mut trait_.composed_traits {
            self.rewrite_type(composed, &context)?;
        }
        for property in &mut trait_.properties {
            self.rewrite_type(&mut property.ty, &context)?;
            if let Some(default) = &mut property.default {
                self.rewrite_expression(default, &context)?;
            }
        }
        for method in &mut trait_.methods {
            if method
                .params
                .first()
                .is_some_and(|parameter| parameter.name.name == "self")
            {
                method.params.remove(0);
            }
            for parameter in &mut method.params {
                self.rewrite_parameter(parameter, &context)?;
            }
            if let Some(returns) = &mut method.return_type {
                self.rewrite_type(returns, &context)?;
            }
        }
        for operator in &mut trait_.operators {
            for parameter in &mut operator.params {
                self.rewrite_parameter(parameter, &context)?;
            }
            if let Some(returns) = &mut operator.return_type {
                self.rewrite_type(returns, &context)?;
            }
        }
        for behavior in &mut trait_.scoped_behaviors {
            for parameter in &mut behavior.params {
                self.rewrite_parameter(parameter, &context)?;
            }
            self.rewrite_block(&mut behavior.body, &context)?;
        }
        Ok(())
    }

    pub(super) fn rewrite_class(
        &mut self,
        class: &mut severian_ast::ClassDecl,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        let mut context = context.with_generics(&class.generic_params);
        context.self_fields = class
            .fields
            .iter()
            .map(|field| field.name.name.clone())
            .collect();
        for implemented in &mut class.traits {
            self.rewrite_type(implemented, &context)?;
        }
        for field in &mut class.fields {
            if let Some(ty) = &mut field.ty {
                self.rewrite_type(ty, &context)?;
            }
            if let Some(default) = &mut field.default {
                self.rewrite_expression(default, &context)?;
            }
            for constraint in &mut field.constraints {
                self.rewrite_expression(constraint, &context)?;
            }
        }
        for constructor in &mut class.constructors {
            if constructor
                .params
                .first()
                .is_some_and(|parameter| parameter.name.name == "self")
            {
                constructor.params.remove(0);
            }
            for parameter in &mut constructor.params {
                self.rewrite_parameter(parameter, &context)?;
            }
            self.rewrite_contract(constructor.contract.as_mut(), &context)?;
            self.rewrite_block(&mut constructor.body, &context)?;
            for test in &mut constructor.tests {
                self.rewrite_test(test, &context)?;
            }
        }
        let mut retained_methods = Vec::new();
        for mut method in std::mem::take(&mut class.methods) {
            if method
                .params
                .first()
                .is_some_and(|parameter| parameter.name.name == "self")
            {
                method.params.remove(0);
            }
            if !self.retain_typestate_method(&mut method, &context) {
                continue;
            }
            self.rewrite_function(&mut method, &context)?;
            retained_methods.push(method);
        }
        class.methods = retained_methods;
        Ok(())
    }

    /// A contract predicate over a class type parameter and transition states
    /// is a compile-time method availability clause. Specialization removes a
    /// satisfied clause and omits the method when it is false.
    fn retain_typestate_method(
        &self,
        method: &mut severian_ast::FunctionDecl,
        context: &RewriteContext,
    ) -> bool {
        let Some(contract) = &mut method.contract else {
            return true;
        };
        let mut available = true;
        contract.clauses.retain(|clause| {
            if let Some(satisfied) = self.typestate_condition(&clause.condition, context) {
                available &= satisfied;
                false
            } else {
                true
            }
        });
        available
    }

    fn typestate_condition(&self, condition: &Expr, context: &RewriteContext) -> Option<bool> {
        let Expr::Binary(binary) = condition else {
            return None;
        };
        match binary.op {
            severian_ast::BinaryOp::And => Some(
                self.typestate_condition(&binary.left, context)?
                    && self.typestate_condition(&binary.right, context)?,
            ),
            severian_ast::BinaryOp::Or => Some(
                self.typestate_condition(&binary.left, context)?
                    || self.typestate_condition(&binary.right, context)?,
            ),
            severian_ast::BinaryOp::Equal | severian_ast::BinaryOp::NotEqual => {
                let equal = self.typestate_equality(&binary.left, &binary.right, context)?;
                Some(if binary.op == severian_ast::BinaryOp::Equal {
                    equal
                } else {
                    !equal
                })
            }
            severian_ast::BinaryOp::In => {
                let Expr::Identifier(parameter) = binary.left.as_ref() else {
                    return None;
                };
                let Expr::List(states) = binary.right.as_ref() else {
                    return None;
                };
                let actual = self.typestate_argument(parameter, context)?;
                let owner = self.transition_states.get(&actual)?;
                let mut matched = false;
                for state in &states.elements {
                    let Expr::Identifier(state) = state else {
                        return None;
                    };
                    if self.transition_states.get(&state.name) != Some(owner) {
                        return None;
                    }
                    matched |= state.name == actual;
                }
                Some(matched)
            }
            _ => None,
        }
    }

    fn typestate_equality(
        &self,
        left: &Expr,
        right: &Expr,
        context: &RewriteContext,
    ) -> Option<bool> {
        let (parameter, expected) = match (left, right) {
            (Expr::Identifier(parameter), Expr::Identifier(expected))
                if context.substitutions.contains_key(&parameter.name) =>
            {
                (parameter, expected)
            }
            (Expr::Identifier(expected), Expr::Identifier(parameter))
                if context.substitutions.contains_key(&parameter.name) =>
            {
                (parameter, expected)
            }
            _ => return None,
        };
        let actual = self.typestate_argument(parameter, context)?;
        let owner = self.transition_states.get(&actual)?;
        (self.transition_states.get(&expected.name) == Some(owner)).then(|| actual == expected.name)
    }

    fn typestate_argument(
        &self,
        parameter: &severian_ast::Ident,
        context: &RewriteContext,
    ) -> Option<String> {
        context
            .substitutions
            .get(&parameter.name)
            .and_then(declaration_type_name)
            .and_then(|name| name.rsplit('.').next().map(str::to_owned))
    }

    pub(super) fn rewrite_function(
        &mut self,
        function: &mut severian_ast::FunctionDecl,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        let context = context.with_generics(&function.generic_params);
        for parameter in &mut function.params {
            self.rewrite_parameter(parameter, &context)?;
        }
        if let Some(returns) = &mut function.return_type {
            self.rewrite_type(returns, &context)?;
        }
        self.rewrite_contract(function.contract.as_mut(), &context)?;
        self.rewrite_block(&mut function.body, &context)?;
        for test in &mut function.tests {
            self.rewrite_test(test, &context)?;
        }
        Ok(())
    }

    fn rewrite_test(
        &mut self,
        test: &mut severian_ast::TestBlock,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        if let Some(returns) = &mut test.return_type {
            self.rewrite_type(returns, context)?;
        }
        self.rewrite_contract(test.contract.as_mut(), context)?;
        self.rewrite_block(&mut test.body, context)
    }

    pub(super) fn rewrite_parameter(
        &mut self,
        parameter: &mut severian_ast::Parameter,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        if let Some(ty) = &mut parameter.ty {
            self.rewrite_type(ty, context)?;
        }
        if let Some(default) = &mut parameter.default {
            self.rewrite_expression(default, context)?;
        }
        Ok(())
    }

    fn rewrite_contract(
        &mut self,
        contract: Option<&mut severian_ast::FunctionContract>,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        if let Some(contract) = contract {
            for clause in &mut contract.clauses {
                self.rewrite_expression(&mut clause.condition, context)?;
            }
        }
        Ok(())
    }
}
