use super::*;

impl Checker {
    pub(super) fn check_lambda(
        &mut self,
        params: &[BindingRef],
        body: &Expression,
    ) -> Result<(), OwnershipError> {
        let previous = params
            .iter()
            .map(|param| (param.clone(), self.bindings.remove(&param.id)))
            .collect::<Vec<_>>();
        for param in params {
            self.define(param.clone(), None);
        }
        self.check_expression(body, Access::Read)?;
        self.restore_closure_parameters(previous);
        Ok(())
    }

    pub(super) fn check_closure(
        &mut self,
        params: &[severian_hir::Parameter],
        body: &[Instruction],
    ) -> Result<(), OwnershipError> {
        let previous = params
            .iter()
            .map(|param| (param.name.clone(), self.bindings.remove(&param.name.id)))
            .collect::<Vec<_>>();
        for param in params {
            self.define(param.name.clone(), None);
        }
        self.check_instructions(body)?;
        self.restore_closure_parameters(previous);
        Ok(())
    }

    fn restore_closure_parameters(&mut self, previous: Vec<(BindingRef, Option<BindingState>)>) {
        for (param, state) in previous {
            if let Some(state) = state {
                self.bindings.insert(param.id, state);
            } else {
                self.bindings.remove(&param.id);
            }
        }
    }
}
