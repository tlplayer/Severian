use super::*;

impl Specializer {
    pub(super) fn rewrite_type(
        &mut self,
        ty: &mut Type,
        context: &RewriteContext,
    ) -> Result<(), SemanticError> {
        if let Type::Named(path) = ty {
            if path.args.is_empty() && path.segments.len() == 1 {
                if let Some(replacement) = context.substitutions.get(&path.segments[0].name) {
                    let span = path.span;
                    *ty = replacement.clone();
                    set_type_span(ty, span);
                    return self.rewrite_type(ty, context);
                }
            }
        }
        match ty {
            Type::Named(path) => {
                for argument in &mut path.args {
                    if let TypeArg::Type { ty, .. } = argument {
                        self.rewrite_type(ty, context)?;
                    }
                }
                let declared = path
                    .segments
                    .iter()
                    .map(|part| part.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                let Some(template) = self.resolve_template(&declared, context.namespace.as_deref())
                else {
                    return Ok(());
                };
                if path.args.is_empty() {
                    return Ok(());
                }
                let arguments = path
                    .args
                    .iter()
                    .map(|argument| argument.as_type().cloned())
                    .collect::<Option<Vec<_>>>();
                let Some(arguments) = arguments else {
                    return Ok(());
                };
                if arguments
                    .iter()
                    .any(|argument| contains_generic(argument, &context.generic_names))
                {
                    return Ok(());
                }
                let name = self.request(&template, arguments, path.span)?;
                path.segments = vec![severian_ast::Ident {
                    span: path.span,
                    name,
                }];
                path.args.clear();
                Ok(())
            }
            Type::List { element, .. }
            | Type::Set { element, .. }
            | Type::Option { some: element, .. }
            | Type::Future {
                output: element, ..
            }
            | Type::Reference { inner: element, .. } => self.rewrite_type(element, context),
            Type::Tuple { elements, .. }
            | Type::Union {
                alternatives: elements,
                ..
            } => {
                for element in elements {
                    self.rewrite_type(element, context)?;
                }
                Ok(())
            }
            Type::Map { key, value, .. }
            | Type::Result {
                ok: key,
                err: value,
                ..
            } => {
                self.rewrite_type(key, context)?;
                self.rewrite_type(value, context)
            }
            Type::Function {
                params, returns, ..
            } => {
                for parameter in params {
                    self.rewrite_type(parameter, context)?;
                }
                self.rewrite_type(returns, context)
            }
        }
    }
}
