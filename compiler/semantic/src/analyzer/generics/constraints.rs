use super::*;

impl Specializer {
    pub(super) fn validate_bounds(
        &self,
        class: &str,
        parameter: &severian_ast::GenericParameter,
        argument: &Type,
        span: Span,
    ) -> Result<(), SemanticError> {
        for constraint in &parameter.constraints {
            let constraint_name = declaration_type_name(constraint).unwrap_or_default();
            let constraint_name = constraint_name.rsplit('.').next().unwrap_or("");
            let satisfied = match constraint_name {
                "DType" | "TensorDType" | "dtype" => dtype_argument(argument).is_some(),
                "Any" | "any" => true,
                "Numeric" | "numeric" => dtype_argument(argument).is_some_and(|dtype| {
                    dtype.satisfies(severian_hir::TensorElementConstraint::Numeric)
                }),
                "Integer" | "integer" => dtype_argument(argument).is_some_and(|dtype| {
                    dtype.satisfies(severian_hir::TensorElementConstraint::Integer)
                }),
                "SignedInteger" | "signed_integer" => {
                    dtype_argument(argument).is_some_and(|dtype| {
                        dtype.satisfies(severian_hir::TensorElementConstraint::SignedInteger)
                    })
                }
                "UnsignedInteger" | "unsigned_integer" => {
                    dtype_argument(argument).is_some_and(|dtype| {
                        dtype.satisfies(severian_hir::TensorElementConstraint::UnsignedInteger)
                    })
                }
                "Float" | "float" => dtype_argument(argument).is_some_and(|dtype| {
                    dtype.satisfies(severian_hir::TensorElementConstraint::Float)
                }),
                "Complex" | "complex" => dtype_argument(argument).is_some_and(|dtype| {
                    dtype.satisfies(severian_hir::TensorElementConstraint::Complex)
                }),
                custom => self.argument_implements(argument, custom),
            };
            if !satisfied {
                return Err(error(
                    span,
                    format!(
                        "type `{}` does not satisfy `{}` for `{}.{}`",
                        declaration_type_key(argument),
                        declaration_type_key(constraint),
                        class,
                        parameter.name.name
                    ),
                ));
            }
        }
        Ok(())
    }

    fn argument_implements(&self, argument: &Type, constraint: &str) -> bool {
        let Some(name) = declaration_type_name(argument) else {
            return false;
        };
        let base = name.split("__").next().unwrap_or(&name);
        self.classes.get(base).is_some_and(|class| {
            class.traits.iter().any(|implemented| {
                declaration_type_name(implemented)
                    .and_then(|name| name.rsplit('.').next().map(str::to_owned))
                    .as_deref()
                    == Some(constraint)
            }) || self.traits.get(constraint).is_some_and(|trait_| {
                trait_.methods.iter().all(|required| {
                    class
                        .methods
                        .iter()
                        .find(|method| method.name.name == required.name.name)
                        .is_some_and(|method| callable_types_match(method, required, argument))
                })
            })
        })
    }
}
