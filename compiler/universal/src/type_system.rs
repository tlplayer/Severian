use crate::{DefId, GenericParamId, InferVarId, PrimitiveId, RegionId, TyId, TypeId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Substitution(pub Vec<(GenericParamId, TyId)>);

impl Substitution {
    pub fn new(arguments: impl IntoIterator<Item = (GenericParamId, TyId)>) -> Self {
        let mut arguments = arguments.into_iter().collect::<Vec<_>>();
        arguments.sort_by_key(|(parameter, _)| *parameter);
        arguments.dedup_by_key(|(parameter, _)| *parameter);
        Self(arguments)
    }

    pub fn get(&self, parameter: GenericParamId) -> Option<TyId> {
        self.0
            .binary_search_by_key(&parameter, |(known, _)| *known)
            .ok()
            .map(|index| self.0[index].1)
    }

    pub fn values(&self) -> impl Iterator<Item = TyId> + '_ {
        self.0.iter().map(|(_, ty)| *ty)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Signature {
    pub parameters: Vec<TyId>,
    pub result: TyId,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TyKind {
    Primitive(PrimitiveId),
    Nominal(DefId, Substitution),
    Parameter(GenericParamId),
    Infer(InferVarId),
    Function(Signature),
    Tuple(Vec<TyId>),
    Union(Vec<TyId>),
    Reference {
        target: TyId,
        mutable: bool,
        lifetime: RegionId,
    },
    Resource(DefId, Substitution),
}

#[derive(Debug, Clone, Default)]
pub struct TyInterner {
    kinds: Vec<TyKind>,
    interned: BTreeMap<TyKind, TyId>,
    next_infer: u32,
}

impl TyInterner {
    pub fn intern(&mut self, kind: TyKind) -> TyId {
        if let Some(id) = self.interned.get(&kind) {
            return *id;
        }
        let id = TypeId(self.kinds.len() as u32);
        self.kinds.push(kind.clone());
        self.interned.insert(kind, id);
        id
    }

    pub fn kind(&self, id: TyId) -> Option<&TyKind> {
        self.kinds.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    pub fn fresh_infer(&mut self) -> TyId {
        let variable = InferVarId(self.next_infer);
        self.next_infer += 1;
        self.intern(TyKind::Infer(variable))
    }

    pub fn parameter(&mut self, parameter: GenericParamId) -> TyId {
        self.intern(TyKind::Parameter(parameter))
    }

    pub fn union(&mut self, members: impl IntoIterator<Item = TyId>) -> TyId {
        let mut canonical = BTreeSet::new();
        for member in members {
            match self.kind(member) {
                Some(TyKind::Union(nested)) => canonical.extend(nested.iter().copied()),
                _ => {
                    canonical.insert(member);
                }
            }
        }
        if canonical.len() == 1 {
            return *canonical.first().expect("one canonical union member");
        }
        self.intern(TyKind::Union(canonical.into_iter().collect()))
    }

    pub(crate) fn replace(&mut self, id: TyId, kind: TyKind) {
        let slot = self
            .kinds
            .get_mut(id.0 as usize)
            .expect("replaced type identities are interned");
        self.interned.remove(slot);
        *slot = kind.clone();
        self.interned.insert(kind, id);
    }

    pub fn substitute(&mut self, ty: TyId, substitution: &Substitution) -> TyId {
        let Some(kind) = self.kind(ty).cloned() else {
            return ty;
        };
        match kind {
            TyKind::Parameter(parameter) => substitution.get(parameter).unwrap_or(ty),
            TyKind::Primitive(_) | TyKind::Infer(_) => ty,
            TyKind::Nominal(definition, arguments) => {
                let arguments = Substitution::new(
                    arguments
                        .0
                        .into_iter()
                        .map(|(parameter, ty)| (parameter, self.substitute(ty, substitution))),
                );
                self.intern(TyKind::Nominal(definition, arguments))
            }
            TyKind::Resource(definition, arguments) => {
                let arguments = Substitution::new(
                    arguments
                        .0
                        .into_iter()
                        .map(|(parameter, ty)| (parameter, self.substitute(ty, substitution))),
                );
                self.intern(TyKind::Resource(definition, arguments))
            }
            TyKind::Function(signature) => {
                let parameters = signature
                    .parameters
                    .into_iter()
                    .map(|parameter| self.substitute(parameter, substitution))
                    .collect();
                let result = self.substitute(signature.result, substitution);
                self.intern(TyKind::Function(Signature { parameters, result }))
            }
            TyKind::Tuple(elements) => {
                let elements = elements
                    .into_iter()
                    .map(|element| self.substitute(element, substitution))
                    .collect();
                self.intern(TyKind::Tuple(elements))
            }
            TyKind::Union(members) => {
                let members = members
                    .into_iter()
                    .map(|member| self.substitute(member, substitution))
                    .collect::<Vec<_>>();
                self.union(members)
            }
            TyKind::Reference {
                target,
                mutable,
                lifetime,
            } => {
                let target = self.substitute(target, substitution);
                self.intern(TyKind::Reference {
                    target,
                    mutable,
                    lifetime,
                })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constraint {
    Equal(TyId, TyId),
    Implements(TraitRef),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitRef {
    pub trait_id: DefId,
    pub self_ty: TyId,
    pub substitution: Substitution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifyError {
    Mismatch(TyId, TyId),
    Occurs(InferVarId, TyId),
    Unknown(TyId),
}

#[derive(Debug, Clone, Default)]
pub struct InferenceContext {
    bindings: BTreeMap<InferVarId, TyId>,
    pub constraints: Vec<Constraint>,
}

impl InferenceContext {
    pub fn resolve(&self, interner: &TyInterner, ty: TyId) -> TyId {
        match interner.kind(ty) {
            Some(TyKind::Infer(variable)) => self
                .bindings
                .get(variable)
                .copied()
                .map_or(ty, |bound| self.resolve(interner, bound)),
            _ => ty,
        }
    }

    pub fn constrain_equal(
        &mut self,
        interner: &TyInterner,
        left: TyId,
        right: TyId,
    ) -> Result<(), UnifyError> {
        self.constraints.push(Constraint::Equal(left, right));
        self.unify(interner, left, right)
    }

    pub fn unify(
        &mut self,
        interner: &TyInterner,
        left: TyId,
        right: TyId,
    ) -> Result<(), UnifyError> {
        let left = self.resolve(interner, left);
        let right = self.resolve(interner, right);
        if left == right {
            return Ok(());
        }
        match (interner.kind(left), interner.kind(right)) {
            (Some(TyKind::Infer(variable)), _) => self.bind(interner, *variable, right),
            (_, Some(TyKind::Infer(variable))) => self.bind(interner, *variable, left),
            (Some(TyKind::Tuple(left)), Some(TyKind::Tuple(right)))
            | (Some(TyKind::Union(left)), Some(TyKind::Union(right)))
                if left.len() == right.len() =>
            {
                for (left, right) in left.iter().zip(right) {
                    self.unify(interner, *left, *right)?;
                }
                Ok(())
            }
            (Some(TyKind::Function(left)), Some(TyKind::Function(right)))
                if left.parameters.len() == right.parameters.len() =>
            {
                for (left, right) in left.parameters.iter().zip(&right.parameters) {
                    self.unify(interner, *left, *right)?;
                }
                self.unify(interner, left.result, right.result)
            }
            (
                Some(TyKind::Reference {
                    target: left,
                    mutable: left_mutable,
                    lifetime: left_lifetime,
                }),
                Some(TyKind::Reference {
                    target: right,
                    mutable: right_mutable,
                    lifetime: right_lifetime,
                }),
            ) if left_mutable == right_mutable && left_lifetime == right_lifetime => {
                self.unify(interner, *left, *right)
            }
            (Some(TyKind::Nominal(left, left_args)), Some(TyKind::Nominal(right, right_args)))
            | (
                Some(TyKind::Resource(left, left_args)),
                Some(TyKind::Resource(right, right_args)),
            ) if left == right && left_args.0.len() == right_args.0.len() => {
                for ((_, left), (_, right)) in left_args.0.iter().zip(&right_args.0) {
                    self.unify(interner, *left, *right)?;
                }
                Ok(())
            }
            (Some(_), Some(_)) => Err(UnifyError::Mismatch(left, right)),
            _ => Err(UnifyError::Unknown(if interner.kind(left).is_none() {
                left
            } else {
                right
            })),
        }
    }

    fn bind(
        &mut self,
        interner: &TyInterner,
        variable: InferVarId,
        ty: TyId,
    ) -> Result<(), UnifyError> {
        if matches!(interner.kind(ty), Some(TyKind::Infer(found)) if *found == variable) {
            return Ok(());
        }
        if occurs(interner, &self.bindings, variable, ty) {
            return Err(UnifyError::Occurs(variable, ty));
        }
        self.bindings.insert(variable, ty);
        Ok(())
    }
}

fn occurs(
    interner: &TyInterner,
    bindings: &BTreeMap<InferVarId, TyId>,
    variable: InferVarId,
    ty: TyId,
) -> bool {
    match interner.kind(ty) {
        Some(TyKind::Infer(found)) => {
            *found == variable
                || bindings
                    .get(found)
                    .is_some_and(|bound| occurs(interner, bindings, variable, *bound))
        }
        Some(TyKind::Nominal(_, substitution)) | Some(TyKind::Resource(_, substitution)) => {
            substitution
                .values()
                .any(|ty| occurs(interner, bindings, variable, ty))
        }
        Some(TyKind::Function(signature)) => signature
            .parameters
            .iter()
            .copied()
            .chain([signature.result])
            .any(|ty| occurs(interner, bindings, variable, ty)),
        Some(TyKind::Tuple(types)) | Some(TyKind::Union(types)) => types
            .iter()
            .any(|ty| occurs(interner, bindings, variable, *ty)),
        Some(TyKind::Reference { target, .. }) => occurs(interner, bindings, variable, *target),
        Some(TyKind::Primitive(_) | TyKind::Parameter(_)) | None => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImplId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplDefinition {
    pub id: ImplId,
    pub trait_id: DefId,
    pub self_ty: TyId,
    pub substitution: Substitution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImplSelection {
    Selected(ImplId),
    Missing,
    Ambiguous(Vec<ImplId>),
}

#[derive(Debug, Clone, Default)]
pub struct ImplTable {
    implementations: Vec<ImplDefinition>,
}

impl ImplTable {
    pub fn insert(&mut self, trait_id: DefId, self_ty: TyId, substitution: Substitution) -> ImplId {
        let id = ImplId(self.implementations.len() as u32);
        self.implementations.push(ImplDefinition {
            id,
            trait_id,
            self_ty,
            substitution,
        });
        id
    }

    pub fn select(&self, obligation: &TraitRef) -> ImplSelection {
        let matches = self
            .implementations
            .iter()
            .filter(|implementation| {
                implementation.trait_id == obligation.trait_id
                    && implementation.self_ty == obligation.self_ty
                    && implementation.substitution == obligation.substitution
            })
            .map(|implementation| implementation.id)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => ImplSelection::Missing,
            [selected] => ImplSelection::Selected(*selected),
            _ => ImplSelection::Ambiguous(matches),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeclarationId;

    fn definition(name: &str) -> DefId {
        DefId {
            package: 0,
            module: 0,
            declaration: DeclarationId::from_path(name),
        }
    }

    #[test]
    fn unions_are_flattened_sorted_and_deduplicated() {
        let mut types = TyInterner::default();
        let first = types.intern(TyKind::Nominal(
            definition("First"),
            Substitution::default(),
        ));
        let second = types.intern(TyKind::Nominal(
            definition("Second"),
            Substitution::default(),
        ));
        let nested = types.union([second, first]);
        assert_eq!(nested, types.union([first, nested, second]));
        assert_eq!(
            types.kind(nested),
            Some(&TyKind::Union(vec![first, second]))
        );
    }

    #[test]
    fn inference_variables_unify_structurally() {
        let mut types = TyInterner::default();
        let concrete = types.intern(TyKind::Nominal(
            definition("Value"),
            Substitution::default(),
        ));
        let variable = types.fresh_infer();
        let mut inference = InferenceContext::default();
        inference
            .constrain_equal(&types, variable, concrete)
            .unwrap();
        assert_eq!(inference.resolve(&types, variable), concrete);
    }
}
