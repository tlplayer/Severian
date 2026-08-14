use super::*;
use std::collections::{BTreeMap, HashSet, VecDeque};

mod constraints;
mod declarations;
mod support;
mod templates;
mod types;
use support::*;

#[derive(Clone)]
struct Specialization {
    template: String,
    arguments: Vec<Type>,
    name: String,
}

#[derive(Clone)]
struct GenericTemplate {
    identity: String,
    class: severian_ast::ClassDecl,
    imports: Vec<severian_ast::ImportDecl>,
}

#[derive(Clone, Default)]
struct RewriteContext {
    substitutions: BTreeMap<String, Type>,
    generic_names: HashSet<String>,
    self_fields: HashSet<String>,
    namespace: Option<String>,
}

pub(super) fn specialize_generic_classes(module: &Module) -> Result<Module, SemanticError> {
    specialize_generic_classes_with_interfaces(module, &[])
}

pub(super) fn specialize_generic_classes_with_interfaces(
    module: &Module,
    interfaces: &[PackageInterface],
) -> Result<Module, SemanticError> {
    Specializer::new(module, interfaces).run(module)
}

struct Specializer {
    templates: HashMap<String, GenericTemplate>,
    classes: HashMap<String, severian_ast::ClassDecl>,
    traits: HashMap<String, severian_ast::TraitDecl>,
    aliases: HashMap<String, String>,
    pending: VecDeque<Specialization>,
    scheduled: HashSet<String>,
    required_imports: Vec<severian_ast::ImportDecl>,
}

impl Specializer {
    fn run(mut self, module: &Module) -> Result<Module, SemanticError> {
        let mut output = Module {
            span: module.span,
            items: Vec::new(),
        };
        for item in &module.items {
            if matches!(item, Item::Class(class) if !class.generic_params.is_empty()) {
                continue;
            }
            let mut item = item.clone();
            self.rewrite_item(&mut item, &RewriteContext::default())?;
            output.items.push(item);
        }
        while let Some(specialization) = self.pending.pop_front() {
            let template = self.templates[&specialization.template].clone();
            let mut class = template.class.clone();
            class.name.name = specialization.name;
            class.generic_params.clear();
            let substitutions = template
                .class
                .generic_params
                .iter()
                .zip(specialization.arguments)
                .map(|(parameter, argument)| (parameter.name.name.clone(), argument))
                .collect();
            let context = RewriteContext {
                substitutions,
                generic_names: HashSet::new(),
                self_fields: class
                    .fields
                    .iter()
                    .map(|field| field.name.name.clone())
                    .collect(),
                namespace: template
                    .identity
                    .rsplit_once('.')
                    .map(|(namespace, _)| namespace.to_owned()),
            };
            self.rewrite_class(&mut class, &context)?;
            output.items.push(Item::Class(class));
        }
        for import in self.required_imports {
            if !output
                .items
                .iter()
                .any(|item| matches!(item, Item::Import(existing) if existing == &import))
            {
                output.items.push(Item::Import(import));
            }
        }
        Ok(output)
    }

    fn resolve_template(&self, name: &str, namespace: Option<&str>) -> Option<String> {
        if self.templates.contains_key(name) {
            return Some(name.to_owned());
        }
        if !name.contains('.') {
            if let Some(namespace) = namespace {
                let relative = format!("{namespace}.{name}");
                if self.templates.contains_key(&relative) {
                    return Some(relative);
                }
            }
            return None;
        }
        let (first, rest) = name.split_once('.')?;
        let canonical = self.aliases.get(first)?;
        let resolved = format!("{canonical}.{rest}");
        self.templates.contains_key(&resolved).then_some(resolved)
    }

    fn request(
        &mut self,
        template_name: &str,
        arguments: Vec<Type>,
        span: Span,
    ) -> Result<String, SemanticError> {
        let template = self
            .templates
            .get(template_name)
            .cloned()
            .ok_or_else(|| error(span, format!("unknown generic class `{template_name}`")))?;
        if arguments.len() != template.class.generic_params.len() {
            return Err(error(
                span,
                format!(
                    "generic class `{template_name}` expects {} type argument(s), received {}",
                    template.class.generic_params.len(),
                    arguments.len()
                ),
            ));
        }
        for (parameter, argument) in template.class.generic_params.iter().zip(&arguments) {
            self.validate_bounds(template_name, parameter, argument, span)?;
        }
        let name = specialization_name(&template.identity, &arguments);
        if self.scheduled.insert(name.clone()) {
            for import in template.imports {
                if !self.required_imports.contains(&import) {
                    self.required_imports.push(import);
                }
            }
            self.pending.push_back(Specialization {
                template: template_name.to_owned(),
                arguments,
                name: name.clone(),
            });
        }
        Ok(name)
    }

    fn rewrite_block(
        &mut self,
        block: &mut Block,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        for statement in &mut block.statements {
            self.rewrite_statement(statement, context)?;
        }
        Ok(())
    }

    fn rewrite_statement(
        &mut self,
        statement: &mut Stmt,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        match statement {
            Stmt::Function(function) => self.rewrite_function(function, context),
            Stmt::Let(binding) => {
                if let Some(ty) = &mut binding.ty {
                    self.rewrite_type(ty, context)?;
                }
                if let Some(value) = &mut binding.value {
                    self.rewrite_expression(value, context)?;
                }
                Ok(())
            }
            Stmt::DestructureLet(binding) => self.rewrite_expression(&mut binding.value, context),
            Stmt::Assign(assignment) => {
                self.rewrite_expression(&mut assignment.target, context)?;
                self.rewrite_expression(&mut assignment.value, context)
            }
            Stmt::Assert(assertion) => {
                self.rewrite_expression(&mut assertion.condition, context)?;
                if let Some(message) = &mut assertion.message {
                    self.rewrite_expression(message, context)?;
                }
                Ok(())
            }
            Stmt::TryBind(binding) => {
                if let Some(ty) = &mut binding.ty {
                    self.rewrite_type(ty, context)?;
                }
                self.rewrite_expression(&mut binding.value, context)
            }
            Stmt::Return(return_) => {
                if let Some(value) = &mut return_.value {
                    self.rewrite_expression(value, context)?;
                }
                Ok(())
            }
            Stmt::If(if_) => {
                self.rewrite_expression(&mut if_.condition, context)?;
                self.rewrite_block(&mut if_.then_block, context)?;
                if let Some(else_) = &mut if_.else_branch {
                    self.rewrite_else(else_, context)?;
                }
                Ok(())
            }
            Stmt::While(while_) => {
                if let Some(setup) = &mut while_.setup {
                    self.rewrite_statement(setup, context)?;
                }
                for capability in &mut while_.capabilities {
                    self.rewrite_expression(capability, context)?;
                }
                self.rewrite_expression(&mut while_.condition, context)?;
                self.rewrite_block(&mut while_.body, context)
            }
            Stmt::For(for_) => {
                if let Some(setup) = &mut for_.setup {
                    self.rewrite_statement(setup, context)?;
                }
                self.rewrite_pattern(&mut for_.pattern, context)?;
                self.rewrite_expression(&mut for_.iterable, context)?;
                self.rewrite_block(&mut for_.body, context)
            }
            Stmt::Switch(switch) => {
                for value in &mut switch.values {
                    self.rewrite_expression(value, context)?;
                }
                if let Some(condition) = &mut switch.repeat_condition {
                    self.rewrite_expression(condition, context)?;
                }
                if let Some(setup) = &mut switch.setup {
                    self.rewrite_statement(setup, context)?;
                }
                for arm in &mut switch.arms {
                    if let Some(source) = &mut arm.source {
                        self.rewrite_expression(source, context)?;
                    }
                    self.rewrite_pattern(&mut arm.pattern, context)?;
                    if let Some(guard) = &mut arm.guard {
                        self.rewrite_expression(guard, context)?;
                    }
                    self.rewrite_block(&mut arm.body, context)?;
                }
                Ok(())
            }
            Stmt::With(with) => {
                for resource in &mut with.resources {
                    self.rewrite_expression(resource, context)?;
                }
                self.rewrite_block(&mut with.body, context)
            }
            Stmt::Unsafe(unsafe_) => self.rewrite_block(&mut unsafe_.body, context),
            Stmt::Expr(expression) => self.rewrite_expression(expression, context),
            Stmt::Break(_) | Stmt::Continue(_) => Ok(()),
        }
    }

    fn rewrite_else(
        &mut self,
        else_: &mut severian_ast::ElseBranch,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        match else_ {
            severian_ast::ElseBranch::If(if_) => {
                self.rewrite_expression(&mut if_.condition, context)?;
                self.rewrite_block(&mut if_.then_block, context)?;
                if let Some(else_) = &mut if_.else_branch {
                    self.rewrite_else(else_, context)?;
                }
                Ok(())
            }
            severian_ast::ElseBranch::Block(block) => self.rewrite_block(block, context),
        }
    }

    fn rewrite_pattern(
        &mut self,
        pattern: &mut Pattern,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        match pattern {
            Pattern::Tuple { elements, .. }
            | Pattern::List { elements, .. }
            | Pattern::Or {
                alternatives: elements,
                ..
            } => {
                for element in elements {
                    self.rewrite_pattern(element, context)?;
                }
                Ok(())
            }
            Pattern::Constructor { name, fields, .. } => {
                self.rewrite_type(name, context)?;
                for field in fields {
                    self.rewrite_pattern(field, context)?;
                }
                Ok(())
            }
            Pattern::Wildcard(_) | Pattern::Literal(_) | Pattern::Identifier(_) => Ok(()),
        }
    }

    fn rewrite_expression(
        &mut self,
        expression: &mut Expr,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        if let Some((declared, arguments, span)) = generic_class_expression(expression) {
            let template = self.resolve_template(&declared, context.namespace.as_deref());
            if !arguments
                .iter()
                .any(|argument| contains_generic(argument, &context.generic_names))
            {
                if let Some(template) = template {
                    let name = self.request(&template, arguments, span)?;
                    *expression = Expr::Identifier(severian_ast::Ident { span, name });
                    return Ok(());
                }
            }
        }
        if let Expr::Member(member) = expression {
            if matches!(member.object.as_ref(), Expr::Identifier(identifier) if identifier.name == "self")
                && context.self_fields.contains(&member.member.name)
            {
                *expression = Expr::Identifier(member.member.clone());
                return Ok(());
            }
        }
        match expression {
            Expr::Binary(binary) => {
                self.rewrite_expression(&mut binary.left, context)?;
                self.rewrite_expression(&mut binary.right, context)
            }
            Expr::Unary(unary) => self.rewrite_expression(&mut unary.expr, context),
            Expr::Call(call) => {
                self.rewrite_expression(&mut call.callee, context)?;
                for argument in &mut call.args {
                    self.rewrite_expression(&mut argument.value, context)?;
                }
                Ok(())
            }
            Expr::Member(member) => self.rewrite_expression(&mut member.object, context),
            Expr::List(collection) | Expr::Tuple(collection) | Expr::Set(collection) => {
                for element in &mut collection.elements {
                    self.rewrite_expression(element, context)?;
                }
                Ok(())
            }
            Expr::ListComprehension(comprehension) => {
                self.rewrite_expression(&mut comprehension.element, context)?;
                self.rewrite_clauses(&mut comprehension.clauses, context)
            }
            Expr::SetComprehension(comprehension) => {
                self.rewrite_expression(&mut comprehension.element, context)?;
                self.rewrite_clauses(&mut comprehension.clauses, context)
            }
            Expr::MapComprehension(comprehension) => {
                self.rewrite_expression(&mut comprehension.key, context)?;
                self.rewrite_expression(&mut comprehension.value, context)?;
                self.rewrite_clauses(&mut comprehension.clauses, context)
            }
            Expr::Map(map) => {
                for entry in &mut map.entries {
                    self.rewrite_expression(&mut entry.key, context)?;
                    self.rewrite_expression(&mut entry.value, context)?;
                }
                Ok(())
            }
            Expr::Index(index) => {
                self.rewrite_expression(&mut index.object, context)?;
                self.rewrite_expression(&mut index.index, context)
            }
            Expr::Slice(slice) => {
                self.rewrite_expression(&mut slice.object, context)?;
                for bound in [&mut slice.start, &mut slice.end, &mut slice.step]
                    .into_iter()
                    .flatten()
                {
                    self.rewrite_expression(bound, context)?;
                }
                Ok(())
            }
            Expr::If(if_) => {
                self.rewrite_expression(&mut if_.condition, context)?;
                self.rewrite_expression(&mut if_.then_expr, context)?;
                self.rewrite_expression(&mut if_.else_expr, context)
            }
            Expr::Switch(switch) => {
                self.rewrite_expression(&mut switch.value, context)?;
                for arm in &mut switch.arms {
                    self.rewrite_pattern(&mut arm.pattern, context)?;
                    if let Some(guard) = &mut arm.guard {
                        self.rewrite_expression(guard, context)?;
                    }
                    self.rewrite_expression(&mut arm.value, context)?;
                }
                Ok(())
            }
            Expr::Lambda(lambda) => {
                for parameter in &mut lambda.params {
                    self.rewrite_parameter(parameter, context)?;
                }
                if let Some(returns) = &mut lambda.return_type {
                    self.rewrite_type(returns, context)?;
                }
                match &mut lambda.body {
                    severian_ast::LambdaBody::Expr(expression) => {
                        self.rewrite_expression(expression, context)
                    }
                    severian_ast::LambdaBody::Block(block) => self.rewrite_block(block, context),
                }
            }
            Expr::Await(await_) => self.rewrite_expression(&mut await_.value, context),
            Expr::Async(async_) => self.rewrite_expression(&mut async_.value, context),
            Expr::Channel(channel) => {
                self.rewrite_type(&mut channel.element_type, context)?;
                self.rewrite_expression(&mut channel.capacity, context)
            }
            Expr::Send(send) => {
                self.rewrite_expression(&mut send.value, context)?;
                self.rewrite_expression(&mut send.channel, context)
            }
            Expr::Ownership(ownership) => self.rewrite_expression(&mut ownership.value, context),
            Expr::ChaosRule(rule) => {
                self.rewrite_expression(&mut rule.function, context)?;
                self.rewrite_expression(&mut rule.value, context)
            }
            Expr::Literal(_) | Expr::Identifier(_) => Ok(()),
        }
    }

    fn rewrite_clauses(
        &mut self,
        clauses: &mut [severian_ast::ComprehensionClause],
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        for clause in clauses {
            self.rewrite_pattern(&mut clause.pattern, context)?;
            self.rewrite_expression(&mut clause.iterable, context)?;
            if let Some(condition) = &mut clause.condition {
                self.rewrite_expression(condition, context)?;
            }
        }
        Ok(())
    }
}

impl RewriteContext {
    fn with_generics(&self, parameters: &[severian_ast::GenericParameter]) -> Self {
        let mut context = self.clone();
        context.generic_names.extend(
            parameters
                .iter()
                .map(|parameter| parameter.name.name.clone()),
        );
        context
    }
}
