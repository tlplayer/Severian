use crate::{DefId, GenericParamId, InferVarId, PrimitiveId, RegionId, TensorShape, TyId, TypeId};
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
pub enum TypeKind {
    Primitive(PrimitiveId),
    Nominal(DefId, Substitution),
    /// A tensor keeps its element and shape structurally. It is not an opaque
    /// nominal whose generic arguments must be reconstructed from a name.
    Tensor {
        constructor: DefId,
        element: TyId,
        shape: TensorShape,
    },
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TyInterner {
    kinds: Vec<TypeKind>,
    interned: BTreeMap<TypeKind, TyId>,
    next_infer: u32,
}

impl TyInterner {
    pub fn intern(&mut self, kind: TypeKind) -> TyId {
        if let Some(id) = self.interned.get(&kind) {
            return *id;
        }
        let id = TypeId(self.kinds.len() as u32);
        self.kinds.push(kind.clone());
        self.interned.insert(kind, id);
        id
    }

    pub fn kind(&self, id: TyId) -> Option<&TypeKind> {
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
        self.intern(TypeKind::Infer(variable))
    }

    pub fn parameter(&mut self, parameter: GenericParamId) -> TyId {
        self.intern(TypeKind::Parameter(parameter))
    }

    pub fn union(&mut self, members: impl IntoIterator<Item = TyId>) -> TyId {
        let mut canonical = BTreeSet::new();
        for member in members {
            match self.kind(member) {
                Some(TypeKind::Union(nested)) => canonical.extend(nested.iter().copied()),
                _ => {
                    canonical.insert(member);
                }
            }
        }
        if canonical.len() == 1 {
            return *canonical.first().expect("one canonical union member");
        }
        self.intern(TypeKind::Union(canonical.into_iter().collect()))
    }

    pub(crate) fn replace(&mut self, id: TyId, kind: TypeKind) {
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
            TypeKind::Parameter(parameter) => substitution.get(parameter).unwrap_or(ty),
            TypeKind::Primitive(_) | TypeKind::Infer(_) => ty,
            TypeKind::Tensor {
                constructor,
                element,
                shape,
            } => {
                let element = self.substitute(element, substitution);
                self.intern(TypeKind::Tensor {
                    constructor,
                    element,
                    shape,
                })
            }
            TypeKind::Nominal(definition, arguments) => {
                let arguments = Substitution::new(
                    arguments
                        .0
                        .into_iter()
                        .map(|(parameter, ty)| (parameter, self.substitute(ty, substitution))),
                );
                self.intern(TypeKind::Nominal(definition, arguments))
            }
            TypeKind::Resource(definition, arguments) => {
                let arguments = Substitution::new(
                    arguments
                        .0
                        .into_iter()
                        .map(|(parameter, ty)| (parameter, self.substitute(ty, substitution))),
                );
                self.intern(TypeKind::Resource(definition, arguments))
            }
            TypeKind::Function(signature) => {
                let parameters = signature
                    .parameters
                    .into_iter()
                    .map(|parameter| self.substitute(parameter, substitution))
                    .collect();
                let result = self.substitute(signature.result, substitution);
                self.intern(TypeKind::Function(Signature { parameters, result }))
            }
            TypeKind::Tuple(elements) => {
                let elements = elements
                    .into_iter()
                    .map(|element| self.substitute(element, substitution))
                    .collect();
                self.intern(TypeKind::Tuple(elements))
            }
            TypeKind::Union(members) => {
                let members = members
                    .into_iter()
                    .map(|member| self.substitute(member, substitution))
                    .collect::<Vec<_>>();
                self.union(members)
            }
            TypeKind::Reference {
                target,
                mutable,
                lifetime,
            } => {
                let target = self.substitute(target, substitution);
                self.intern(TypeKind::Reference {
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
            Some(TypeKind::Infer(variable)) => self
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
            (Some(TypeKind::Infer(variable)), _) => self.bind(interner, *variable, right),
            (_, Some(TypeKind::Infer(variable))) => self.bind(interner, *variable, left),
            (
                Some(TypeKind::Tensor {
                    constructor: left_constructor,
                    element: left_element,
                    shape: left_shape,
                }),
                Some(TypeKind::Tensor {
                    constructor: right_constructor,
                    element: right_element,
                    shape: right_shape,
                }),
            ) if left_constructor == right_constructor && left_shape == right_shape => {
                self.unify(interner, *left_element, *right_element)
            }
            (Some(TypeKind::Tuple(left)), Some(TypeKind::Tuple(right)))
            | (Some(TypeKind::Union(left)), Some(TypeKind::Union(right)))
                if left.len() == right.len() =>
            {
                for (left, right) in left.iter().zip(right) {
                    self.unify(interner, *left, *right)?;
                }
                Ok(())
            }
            (Some(TypeKind::Function(left)), Some(TypeKind::Function(right)))
                if left.parameters.len() == right.parameters.len() =>
            {
                for (left, right) in left.parameters.iter().zip(&right.parameters) {
                    self.unify(interner, *left, *right)?;
                }
                self.unify(interner, left.result, right.result)
            }
            (
                Some(TypeKind::Reference {
                    target: left,
                    mutable: left_mutable,
                    lifetime: left_lifetime,
                }),
                Some(TypeKind::Reference {
                    target: right,
                    mutable: right_mutable,
                    lifetime: right_lifetime,
                }),
            ) if left_mutable == right_mutable && left_lifetime == right_lifetime => {
                self.unify(interner, *left, *right)
            }
            (
                Some(TypeKind::Nominal(left, left_args)),
                Some(TypeKind::Nominal(right, right_args)),
            )
            | (
                Some(TypeKind::Resource(left, left_args)),
                Some(TypeKind::Resource(right, right_args)),
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
        if matches!(interner.kind(ty), Some(TypeKind::Infer(found)) if *found == variable) {
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
        Some(TypeKind::Infer(found)) => {
            *found == variable
                || bindings
                    .get(found)
                    .is_some_and(|bound| occurs(interner, bindings, variable, *bound))
        }
        Some(TypeKind::Nominal(_, substitution)) | Some(TypeKind::Resource(_, substitution)) => {
            substitution
                .values()
                .any(|ty| occurs(interner, bindings, variable, ty))
        }
        Some(TypeKind::Tensor { element, .. }) => occurs(interner, bindings, variable, *element),
        Some(TypeKind::Function(signature)) => signature
            .parameters
            .iter()
            .copied()
            .chain([signature.result])
            .any(|ty| occurs(interner, bindings, variable, ty)),
        Some(TypeKind::Tuple(types)) | Some(TypeKind::Union(types)) => types
            .iter()
            .any(|ty| occurs(interner, bindings, variable, *ty)),
        Some(TypeKind::Reference { target, .. }) => occurs(interner, bindings, variable, *target),
        Some(TypeKind::Primitive(_) | TypeKind::Parameter(_)) | None => false,
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
        let first = types.intern(TypeKind::Nominal(
            definition("First"),
            Substitution::default(),
        ));
        let second = types.intern(TypeKind::Nominal(
            definition("Second"),
            Substitution::default(),
        ));
        let nested = types.union([second, first]);
        assert_eq!(nested, types.union([first, nested, second]));
        assert_eq!(
            types.kind(nested),
            Some(&TypeKind::Union(vec![first, second]))
        );
    }

    #[test]
    fn inference_variables_unify_structurally() {
        let mut types = TyInterner::default();
        let concrete = types.intern(TypeKind::Nominal(
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

    #[test]
    fn substitutions_are_canonical_and_queryable() {
        let substitution = Substitution::new([
            (GenericParamId(2), TypeId(20)),
            (GenericParamId(0), TypeId(10)),
            (GenericParamId(2), TypeId(99)),
        ]);
        assert_eq!(
            substitution.0,
            vec![
                (GenericParamId(0), TypeId(10)),
                (GenericParamId(2), TypeId(20)),
            ]
        );
        assert_eq!(substitution.get(GenericParamId(0)), Some(TypeId(10)));
        assert_eq!(substitution.get(GenericParamId(1)), None);
        assert_eq!(
            substitution.values().collect::<Vec<_>>(),
            [TypeId(10), TypeId(20)]
        );
    }

    #[test]
    fn interning_and_substitution_cover_every_structural_type() {
        let mut types = TyInterner::default();
        assert!(types.is_empty());
        assert_eq!(types.kind(TypeId(99)), None);

        let primitive = types.intern(TypeKind::Primitive(PrimitiveId(DeclarationId::from_path(
            "core.i32",
        ))));
        assert_eq!(
            types.intern(types.kind(primitive).unwrap().clone()),
            primitive
        );
        assert_eq!(types.len(), 1);

        let parameter = types.parameter(GenericParamId(0));
        let untouched_parameter = types.parameter(GenericParamId(1));
        let inference = types.fresh_infer();
        let def = definition("Box");
        let arguments = Substitution::new([(GenericParamId(0), parameter)]);
        let nominal = types.intern(TypeKind::Nominal(def, arguments.clone()));
        let resource = types.intern(TypeKind::Resource(def, arguments));
        let function = types.intern(TypeKind::Function(Signature {
            parameters: vec![parameter],
            result: parameter,
        }));
        let tuple = types.intern(TypeKind::Tuple(vec![parameter, primitive]));
        let union = types.union([parameter, primitive]);
        let reference = types.intern(TypeKind::Reference {
            target: parameter,
            mutable: true,
            lifetime: RegionId(3),
        });
        let substitution = Substitution::new([(GenericParamId(0), primitive)]);

        assert_eq!(types.substitute(parameter, &substitution), primitive);
        assert_eq!(
            types.substitute(untouched_parameter, &substitution),
            untouched_parameter
        );
        assert_eq!(types.substitute(primitive, &substitution), primitive);
        assert_eq!(types.substitute(inference, &substitution), inference);
        assert_eq!(types.substitute(TypeId(999), &substitution), TypeId(999));
        for structural in [nominal, resource, function, tuple, union, reference] {
            let substituted = types.substitute(structural, &substitution);
            assert_ne!(substituted, structural);
        }

        let replaceable = types.intern(TypeKind::Tuple(vec![]));
        types.replace(replaceable, TypeKind::Union(vec![]));
        assert_eq!(types.kind(replaceable), Some(&TypeKind::Union(vec![])));
        assert_eq!(types.intern(TypeKind::Union(vec![])), replaceable);
    }

    #[test]
    fn inference_unifies_all_matching_structures_and_rejects_mismatches() {
        let mut types = TyInterner::default();
        let first = types.intern(TypeKind::Primitive(PrimitiveId(DeclarationId::from_path(
            "First",
        ))));
        let second = types.intern(TypeKind::Primitive(PrimitiveId(DeclarationId::from_path(
            "Second",
        ))));
        let mut inference = InferenceContext::default();
        assert_eq!(inference.unify(&types, first, first), Ok(()));

        let right_variable = types.fresh_infer();
        inference.unify(&types, first, right_variable).unwrap();
        assert_eq!(inference.resolve(&types, right_variable), first);

        let tuple_variable = types.fresh_infer();
        let left_tuple = types.intern(TypeKind::Tuple(vec![tuple_variable]));
        let right_tuple = types.intern(TypeKind::Tuple(vec![first]));
        inference.unify(&types, left_tuple, right_tuple).unwrap();

        let union_variable = types.fresh_infer();
        let left_union = types.intern(TypeKind::Union(vec![union_variable]));
        let right_union = types.intern(TypeKind::Union(vec![first]));
        inference.unify(&types, left_union, right_union).unwrap();

        let function_variable = types.fresh_infer();
        let left_function = types.intern(TypeKind::Function(Signature {
            parameters: vec![function_variable],
            result: function_variable,
        }));
        let right_function = types.intern(TypeKind::Function(Signature {
            parameters: vec![second],
            result: second,
        }));
        inference
            .unify(&types, left_function, right_function)
            .unwrap();

        let reference_variable = types.fresh_infer();
        let left_reference = types.intern(TypeKind::Reference {
            target: reference_variable,
            mutable: true,
            lifetime: RegionId(1),
        });
        let right_reference = types.intern(TypeKind::Reference {
            target: first,
            mutable: true,
            lifetime: RegionId(1),
        });
        inference
            .unify(&types, left_reference, right_reference)
            .unwrap();

        for resource in [false, true] {
            let variable = types.fresh_infer();
            let arguments = Substitution::new([(GenericParamId(0), variable)]);
            let concrete = Substitution::new([(GenericParamId(0), first)]);
            let (left, right) = if resource {
                (
                    types.intern(TypeKind::Resource(definition("Container"), arguments)),
                    types.intern(TypeKind::Resource(definition("Container"), concrete)),
                )
            } else {
                (
                    types.intern(TypeKind::Nominal(definition("Container"), arguments)),
                    types.intern(TypeKind::Nominal(definition("Container"), concrete)),
                )
            };
            inference.unify(&types, left, right).unwrap();
        }

        let short_tuple = types.intern(TypeKind::Tuple(vec![]));
        assert!(matches!(
            inference.unify(&types, short_tuple, right_tuple),
            Err(UnifyError::Mismatch(_, _))
        ));
        let short_function = types.intern(TypeKind::Function(Signature {
            parameters: vec![],
            result: first,
        }));
        assert!(matches!(
            inference.unify(&types, short_function, right_function),
            Err(UnifyError::Mismatch(_, _))
        ));
        let immutable_reference = types.intern(TypeKind::Reference {
            target: first,
            mutable: false,
            lifetime: RegionId(1),
        });
        assert!(matches!(
            inference.unify(&types, immutable_reference, right_reference),
            Err(UnifyError::Mismatch(_, _))
        ));
        assert_eq!(
            inference.unify(&types, TypeId(999), first),
            Err(UnifyError::Unknown(TypeId(999)))
        );
        assert_eq!(
            inference.unify(&types, first, TypeId(998)),
            Err(UnifyError::Unknown(TypeId(998)))
        );
    }

    #[test]
    fn occurs_check_walks_every_recursive_shape_and_bound_variable() {
        let mut types = TyInterner::default();
        let variable = types.fresh_infer();
        let TypeKind::Infer(variable_id) = *types.kind(variable).unwrap() else {
            panic!("fresh inference type");
        };
        let other = types.fresh_infer();
        let primitive = types.intern(TypeKind::Primitive(PrimitiveId(DeclarationId::from_path(
            "Value",
        ))));
        let def = definition("Recursive");
        let substitution = Substitution::new([(GenericParamId(0), variable)]);
        let structures = [
            types.intern(TypeKind::Nominal(def, substitution.clone())),
            types.intern(TypeKind::Resource(def, substitution)),
            types.intern(TypeKind::Function(Signature {
                parameters: vec![primitive],
                result: variable,
            })),
            types.intern(TypeKind::Tuple(vec![variable])),
            types.intern(TypeKind::Union(vec![variable])),
            types.intern(TypeKind::Reference {
                target: variable,
                mutable: false,
                lifetime: RegionId(0),
            }),
        ];
        let bindings = BTreeMap::new();
        for structure in structures {
            assert!(occurs(&types, &bindings, variable_id, structure));
        }
        assert!(!occurs(&types, &bindings, variable_id, primitive));
        assert!(!occurs(&types, &bindings, variable_id, TypeId(999)));

        let TypeKind::Infer(other_id) = *types.kind(other).unwrap() else {
            panic!("fresh inference type");
        };
        let bindings = BTreeMap::from([(other_id, variable)]);
        assert!(occurs(&types, &bindings, variable_id, other));

        let tuple = types.intern(TypeKind::Tuple(vec![variable]));
        let mut inference = InferenceContext::default();
        assert_eq!(
            inference.unify(&types, variable, tuple),
            Err(UnifyError::Occurs(variable_id, tuple))
        );
    }

    #[test]
    fn implementation_selection_reports_missing_unique_and_ambiguous_matches() {
        let trait_id = definition("Display");
        let self_ty = TypeId(4);
        let obligation = TraitRef {
            trait_id,
            self_ty,
            substitution: Substitution::default(),
        };
        let mut table = ImplTable::default();
        assert_eq!(table.select(&obligation), ImplSelection::Missing);
        let first = table.insert(trait_id, self_ty, Substitution::default());
        assert_eq!(table.select(&obligation), ImplSelection::Selected(first));
        let other = table.insert(definition("Other"), self_ty, Substitution::default());
        assert_ne!(first, other);
        let second = table.insert(trait_id, self_ty, Substitution::default());
        assert_eq!(
            table.select(&obligation),
            ImplSelection::Ambiguous(vec![first, second])
        );
    }
}
