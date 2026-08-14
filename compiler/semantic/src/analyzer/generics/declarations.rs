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
        for method in &mut class.methods {
            if method
                .params
                .first()
                .is_some_and(|parameter| parameter.name.name == "self")
            {
                method.params.remove(0);
            }
            self.rewrite_function(method, &context)?;
        }
        Ok(())
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
