use crate::{
    BinaryOperator, CompileRoute, CompilerId, DeclarationId, DefId, GenericParamId, LiteralKind,
    LiteralValue, OperatorSignature, PrimitiveId, Substitution, TyInterner, TypeKind, TypeConstraint,
    TypeId, TypePattern, UnaryOperator,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimitiveCategory {
    Boolean,
    Character,
    Integer,
    Float,
    Measured,
    Text,
    Bytes,
    Absence,
    Unit,
    Arguments,
}

impl PrimitiveCategory {
    pub fn from_contract(value: &str) -> Result<Self, TypeError> {
        match value {
            "boolean" => Ok(Self::Boolean),
            "character" => Ok(Self::Character),
            "integer" => Ok(Self::Integer),
            "float" => Ok(Self::Float),
            "measured" => Ok(Self::Measured),
            "text" => Ok(Self::Text),
            "bytes" => Ok(Self::Bytes),
            "absence" => Ok(Self::Absence),
            "unit" => Ok(Self::Unit),
            "arguments" => Ok(Self::Arguments),
            value => Err(TypeError::UnknownCategory(value.to_owned())),
        }
    }

    pub const fn literal_kind(self) -> Option<LiteralKind> {
        match self {
            Self::Boolean => Some(LiteralKind::Boolean),
            Self::Character => Some(LiteralKind::Character),
            Self::Integer => Some(LiteralKind::Integer),
            Self::Float => Some(LiteralKind::Float),
            Self::Measured => None,
            Self::Text => Some(LiteralKind::String),
            Self::Bytes => Some(LiteralKind::Bytes),
            Self::Absence => Some(LiteralKind::None),
            Self::Unit => Some(LiteralKind::Unit),
            Self::Arguments => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerWidth {
    Fixed(u16),
    Machine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatFormat {
    Ieee(u16),
    BrainFloat16,
    Machine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveRepresentation {
    Integer { bits: IntegerWidth, signed: bool },
    Float { format: FloatFormat },
    PointerInteger { signed: bool },
    Boolean,
    Character,
    String,
    Bytes,
    None,
    Unit,
    Arguments,
}

impl PrimitiveRepresentation {
    pub fn from_contract(value: &str, bits: Option<u16>, signed: bool) -> Result<Self, TypeError> {
        match value {
            "fixed-integer" => Ok(Self::Integer {
                bits: IntegerWidth::Fixed(bits.ok_or(TypeError::MissingBitWidth)?),
                signed,
            }),
            "machine-signed" => Ok(Self::Integer {
                bits: IntegerWidth::Machine,
                signed: true,
            }),
            "pointer-integer" => Ok(Self::PointerInteger { signed }),
            "machine-float" => Ok(Self::Float {
                format: FloatFormat::Machine,
            }),
            "ieee-float" => Ok(Self::Float {
                format: FloatFormat::Ieee(bits.ok_or(TypeError::MissingBitWidth)?),
            }),
            "brain-float" if bits == Some(16) => Ok(Self::Float {
                format: FloatFormat::BrainFloat16,
            }),
            "i1" => Ok(Self::Boolean),
            "unicode-scalar" => Ok(Self::Character),
            "string" => Ok(Self::String),
            "byte-string" => Ok(Self::Bytes),
            "none" => Ok(Self::None),
            "unit" => Ok(Self::Unit),
            "arguments" => Ok(Self::Arguments),
            value => Err(TypeError::UnknownRepresentation(value.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveDefinition {
    pub id: PrimitiveId,
    pub type_id: TypeId,
    pub category: PrimitiveCategory,
    pub representation: PrimitiveRepresentation,
    pub default_literal: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeDefinitionKind {
    Primitive(PrimitiveDefinition),
    Trait,
    Applied {
        constructor: TypeId,
        arguments: Vec<TypeId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDefinition {
    pub id: TypeId,
    pub declaration: DeclarationId,
    pub path: String,
    pub name: String,
    pub parameter_count: usize,
    pub kind: TypeDefinitionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedBinary {
    pub left: TypeId,
    pub right: TypeId,
    pub result: TypeId,
    pub signature: OperatorSignature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedUnary {
    pub operand: TypeId,
    pub result: TypeId,
}

#[derive(Debug, Clone, Default)]
pub struct TypeContext {
    interner: TyInterner,
    definitions: BTreeMap<TypeId, TypeDefinition>,
    by_name: HashMap<String, TypeId>,
    by_declaration: BTreeMap<DeclarationId, TypeId>,
    by_primitive: BTreeMap<PrimitiveId, TypeId>,
    defaults: BTreeMap<LiteralKind, TypeId>,
    binary: Vec<OperatorSignature>,
    unary: Vec<(UnaryOperator, TypeId, TypeId)>,
    compile_routes: BTreeMap<TypeId, CompilerId>,
    applications: BTreeMap<(TypeId, Vec<TypeId>), TypeId>,
    capabilities: BTreeMap<TypeId, BTreeSet<TypeId>>,
    trait_binary: BTreeMap<TypeId, BTreeSet<BinaryOperator>>,
    trait_unary: BTreeMap<TypeId, BTreeSet<UnaryOperator>>,
}

#[derive(Debug, Clone, Default)]
pub struct TypeContextBuilder {
    context: TypeContext,
}

impl TypeContextBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_declaration(
        &mut self,
        path: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<TypeId, TypeError> {
        self.context.register_declaration(path, name, 0)
    }

    pub fn register_generic_declaration(
        &mut self,
        path: impl Into<String>,
        name: impl Into<String>,
        parameter_count: usize,
    ) -> Result<TypeId, TypeError> {
        self.context
            .register_declaration(path, name, parameter_count)
    }

    pub fn define_primitive(
        &mut self,
        type_id: TypeId,
        category: PrimitiveCategory,
        representation: PrimitiveRepresentation,
        default_literal: bool,
    ) -> Result<PrimitiveId, TypeError> {
        self.context
            .define_primitive(type_id, category, representation, default_literal)
    }

    pub fn add_binary(&mut self, signature: OperatorSignature) {
        self.context.add_binary(signature);
    }

    pub fn add_unary(&mut self, operator: UnaryOperator, operand: TypeId, result: TypeId) {
        self.context.add_unary(operator, operand, result);
    }

    pub fn add_capability(&mut self, type_id: TypeId, trait_id: TypeId) -> Result<(), TypeError> {
        self.context.add_capability(type_id, trait_id)
    }

    pub fn add_trait_binary(&mut self, trait_id: TypeId, operator: BinaryOperator) {
        self.context
            .trait_binary
            .entry(trait_id)
            .or_default()
            .insert(operator);
    }

    pub fn add_trait_unary(&mut self, trait_id: TypeId, operator: UnaryOperator) {
        self.context
            .trait_unary
            .entry(trait_id)
            .or_default()
            .insert(operator);
    }

    pub fn set_compile_route(
        &mut self,
        type_id: TypeId,
        compiler: CompilerId,
    ) -> Result<(), TypeError> {
        self.context.set_compile_route(type_id, compiler)
    }

    pub fn instantiate(
        &mut self,
        constructor: TypeId,
        arguments: Vec<TypeId>,
    ) -> Result<TypeId, TypeError> {
        self.context.instantiate(constructor, arguments)
    }

    pub fn resolve_name(&self, name: &str) -> Option<TypeId> {
        self.context.resolve_name(name)
    }

    pub fn compiler_id(&self, declaration: TypeId) -> Result<CompilerId, TypeError> {
        self.context.compiler_id(declaration)
    }

    pub fn build(self) -> TypeContext {
        self.context
    }
}

impl TypeContext {
    pub fn builder() -> TypeContextBuilder {
        TypeContextBuilder::new()
    }

    fn register_declaration(
        &mut self,
        path: impl Into<String>,
        name: impl Into<String>,
        parameter_count: usize,
    ) -> Result<TypeId, TypeError> {
        let path = path.into();
        let name = name.into();
        let declaration = DeclarationId::from_path(&path);
        if let Some(existing) = self.by_declaration.get(&declaration) {
            let known = &self.definitions[existing];
            return if known.path == path {
                Ok(*existing)
            } else {
                Err(TypeError::IdentityCollision(known.path.clone(), path))
            };
        }
        if self.by_name.contains_key(&name) {
            return Err(TypeError::DuplicateName(name));
        }
        let id = self.interner.intern(TypeKind::Nominal(
            DefId {
                package: 0,
                module: 0,
                declaration,
            },
            Substitution::default(),
        ));
        self.definitions.insert(
            id,
            TypeDefinition {
                id,
                declaration,
                path,
                name: name.clone(),
                parameter_count,
                kind: TypeDefinitionKind::Trait,
            },
        );
        self.by_name.insert(name, id);
        self.by_declaration.insert(declaration, id);
        Ok(id)
    }

    fn define_primitive(
        &mut self,
        type_id: TypeId,
        category: PrimitiveCategory,
        representation: PrimitiveRepresentation,
        default_literal: bool,
    ) -> Result<PrimitiveId, TypeError> {
        let definition = self
            .definitions
            .get_mut(&type_id)
            .ok_or(TypeError::UnknownTypeId(type_id))?;
        if matches!(definition.kind, TypeDefinitionKind::Primitive(_)) {
            return Err(TypeError::AlreadyDefined(definition.path.clone()));
        }
        let id = PrimitiveId(definition.declaration);
        let primitive = PrimitiveDefinition {
            id,
            type_id,
            category,
            representation,
            default_literal,
        };
        if default_literal {
            let kind = category
                .literal_kind()
                .ok_or(TypeError::InvalidDefaultLiteralCategory)?;
            if self.defaults.insert(kind, type_id).is_some() {
                return Err(TypeError::DuplicateDefault(kind));
            }
        }
        definition.kind = TypeDefinitionKind::Primitive(primitive);
        self.interner.replace(type_id, TypeKind::Primitive(id));
        self.by_primitive.insert(id, type_id);
        Ok(id)
    }

    fn add_binary(&mut self, signature: OperatorSignature) {
        if !self.binary.contains(&signature) {
            self.binary.push(signature);
        }
    }

    fn add_unary(&mut self, operator: UnaryOperator, operand: TypeId, result: TypeId) {
        if !self.unary.contains(&(operator, operand, result)) {
            self.unary.push((operator, operand, result));
        }
    }

    fn add_capability(&mut self, type_id: TypeId, trait_id: TypeId) -> Result<(), TypeError> {
        self.definition(type_id)
            .ok_or(TypeError::UnknownTypeId(type_id))?;
        self.definition(trait_id)
            .ok_or(TypeError::UnknownTypeId(trait_id))?;
        self.capabilities
            .entry(type_id)
            .or_default()
            .insert(trait_id);
        Ok(())
    }

    pub fn implements(&self, type_id: TypeId, trait_id: TypeId) -> bool {
        type_id == trait_id
            || self
                .capabilities
                .get(&type_id)
                .is_some_and(|traits| traits.contains(&trait_id))
    }

    pub fn supports_binary(&self, operator: BinaryOperator, type_id: TypeId) -> bool {
        self.binary.iter().any(|signature| {
            signature.operator == operator
                && matches!(signature.left, TypePattern::Exact(left) if left == type_id)
                && matches!(signature.right, TypePattern::Exact(right) if right == type_id)
        })
    }

    pub fn supports_unary(&self, operator: UnaryOperator, type_id: TypeId) -> bool {
        self.unary
            .iter()
            .any(|(known, operand, _)| *known == operator && *operand == type_id)
    }

    pub fn trait_supports_binary(&self, trait_id: TypeId, operator: BinaryOperator) -> bool {
        self.trait_binary
            .get(&trait_id)
            .is_some_and(|operators| operators.contains(&operator))
    }

    pub fn trait_supports_unary(&self, trait_id: TypeId, operator: UnaryOperator) -> bool {
        self.trait_unary
            .get(&trait_id)
            .is_some_and(|operators| operators.contains(&operator))
    }

    pub fn resolve_name(&self, name: &str) -> Option<TypeId> {
        self.by_name.get(name).copied()
    }

    pub fn type_for_primitive(&self, id: PrimitiveId) -> Option<TypeId> {
        self.by_primitive.get(&id).copied()
    }

    pub fn definition(&self, id: TypeId) -> Option<&TypeDefinition> {
        self.definitions.get(&id)
    }

    pub fn kind(&self, id: TypeId) -> Option<&TypeKind> {
        self.interner.kind(id)
    }

    pub fn intern(&mut self, kind: TypeKind) -> TypeId {
        self.interner.intern(kind)
    }

    pub fn parameter(&mut self, parameter: GenericParamId) -> TypeId {
        self.interner.parameter(parameter)
    }

    pub fn fresh_infer(&mut self) -> TypeId {
        self.interner.fresh_infer()
    }

    pub fn union(&mut self, members: impl IntoIterator<Item = TypeId>) -> TypeId {
        self.interner.union(members)
    }

    pub fn substitute(&mut self, ty: TypeId, substitution: &Substitution) -> TypeId {
        self.interner.substitute(ty, substitution)
    }

    pub fn primitive(&self, id: TypeId) -> Option<&PrimitiveDefinition> {
        match &self.definition(id)?.kind {
            TypeDefinitionKind::Primitive(primitive) => Some(primitive),
            TypeDefinitionKind::Trait | TypeDefinitionKind::Applied { .. } => None,
        }
    }

    pub fn definitions(&self) -> impl Iterator<Item = &TypeDefinition> {
        self.definitions.values()
    }

    pub fn compiler_id(&self, declaration: TypeId) -> Result<CompilerId, TypeError> {
        let declaration = self
            .definition(declaration)
            .ok_or(TypeError::UnknownTypeId(declaration))?;
        Ok(CompilerId::from_declaration(declaration.declaration))
    }

    fn set_compile_route(
        &mut self,
        type_id: TypeId,
        compiler: CompilerId,
    ) -> Result<(), TypeError> {
        self.definition(type_id)
            .ok_or(TypeError::UnknownTypeId(type_id))?;
        if !self.by_declaration.contains_key(&compiler.declaration()) {
            return Err(TypeError::UnknownCompiler(compiler));
        }
        if let Some(existing) = self.compile_routes.insert(type_id, compiler) {
            if existing != compiler {
                self.compile_routes.insert(type_id, existing);
                return Err(TypeError::CompileRouteAlreadyDefined(type_id));
            }
        }
        Ok(())
    }

    fn instantiate(
        &mut self,
        constructor: TypeId,
        arguments: Vec<TypeId>,
    ) -> Result<TypeId, TypeError> {
        let definition = self
            .definition(constructor)
            .ok_or(TypeError::UnknownTypeId(constructor))?;
        if definition.parameter_count != arguments.len() {
            return Err(TypeError::GenericArity {
                constructor,
                expected: definition.parameter_count,
                actual: arguments.len(),
            });
        }
        for argument in &arguments {
            self.definition(*argument)
                .ok_or(TypeError::UnknownTypeId(*argument))?;
        }
        if let Some(existing) = self.applications.get(&(constructor, arguments.clone())) {
            return Ok(*existing);
        }
        let argument_paths = arguments
            .iter()
            .map(|argument| {
                self.definition(*argument)
                    .expect("arguments were validated")
                    .path
                    .as_str()
            })
            .collect::<Vec<_>>()
            .join(",");
        let path = format!("{}[{argument_paths}]", definition.path);
        let name = format!("{}[{argument_paths}]", definition.name);
        let declaration = DeclarationId::from_path(&path);
        if let Some(existing) = self.by_declaration.get(&declaration) {
            return Ok(*existing);
        }
        let substitution = Substitution::new(
            arguments
                .iter()
                .copied()
                .enumerate()
                .map(|(index, ty)| (GenericParamId(index as u32), ty)),
        );
        let id = self.interner.intern(TypeKind::Nominal(
            DefId {
                package: 0,
                module: 0,
                declaration,
            },
            substitution,
        ));
        self.definitions.insert(
            id,
            TypeDefinition {
                id,
                declaration,
                path,
                name,
                parameter_count: 0,
                kind: TypeDefinitionKind::Applied {
                    constructor,
                    arguments: arguments.clone(),
                },
            },
        );
        self.by_declaration.insert(declaration, id);
        self.applications.insert((constructor, arguments), id);
        Ok(id)
    }

    pub fn compile_route(&self, type_id: TypeId) -> Result<CompileRoute, TypeError> {
        let Some(definition) = self.definition(type_id) else {
            return match self.kind(type_id) {
                Some(
                    TypeKind::Parameter(_)
                    | TypeKind::Infer(_)
                    | TypeKind::Function(_)
                    | TypeKind::Tuple(_)
                    | TypeKind::Union(_)
                    | TypeKind::Reference { .. },
                ) => Ok(CompileRoute::Standard),
                Some(TypeKind::Primitive(_) | TypeKind::Nominal(_, _) | TypeKind::Resource(_, _))
                | None => Err(TypeError::UnknownTypeId(type_id)),
            };
        };
        let route_owner = match &definition.kind {
            TypeDefinitionKind::Applied { constructor, .. } => *constructor,
            _ => type_id,
        };
        Ok(self
            .compile_routes
            .get(&route_owner)
            .copied()
            .map_or(CompileRoute::Standard, CompileRoute::Compiler))
    }

    pub fn operator_signatures(&self) -> impl Iterator<Item = &OperatorSignature> {
        self.binary.iter()
    }

    pub fn assignable(&self, actual: TypeId, expected: TypeId) -> bool {
        if actual == expected {
            return true;
        }
        if let Some(TypeKind::Union(members)) = self.kind(expected) {
            return members
                .iter()
                .any(|member| self.assignable(actual, *member));
        }
        if let Some(TypeKind::Union(members)) = self.kind(actual) {
            return members
                .iter()
                .all(|member| self.assignable(*member, expected));
        }
        let (Some(actual), Some(expected)) = (self.primitive(actual), self.primitive(expected))
        else {
            return false;
        };
        if matches!(actual.category, PrimitiveCategory::Measured)
            || matches!(expected.category, PrimitiveCategory::Measured)
        {
            return false;
        }
        match (actual.representation, expected.representation) {
            (
                PrimitiveRepresentation::Integer {
                    bits: a,
                    signed: sa,
                },
                PrimitiveRepresentation::Integer {
                    bits: e,
                    signed: se,
                },
            ) if sa == se => integer_width_fits(a, e),
            (
                PrimitiveRepresentation::Float { format: a },
                PrimitiveRepresentation::Float { format: e },
            ) => float_width(a)
                .zip(float_width(e))
                .is_some_and(|(a, e)| a <= e),
            (
                PrimitiveRepresentation::Integer { .. }
                | PrimitiveRepresentation::PointerInteger { .. },
                PrimitiveRepresentation::Float { .. },
            ) => true,
            _ => false,
        }
    }

    pub fn resolve_literal(
        &self,
        literal: &LiteralValue,
        expected: Option<TypeId>,
    ) -> Result<TypeId, TypeError> {
        if let Some(expected) = expected {
            let primitive = self
                .primitive(expected)
                .ok_or(TypeError::InvalidLiteralForType(literal.kind(), expected))?;
            if primitive.category.literal_kind() == Some(literal.kind())
                && literal_fits(literal, primitive.representation)
            {
                return Ok(expected);
            }
            return Err(TypeError::InvalidLiteralForType(literal.kind(), expected));
        }
        self.defaults
            .get(&literal.kind())
            .copied()
            .ok_or(TypeError::NoLiteralDefault(literal.kind()))
    }

    pub fn resolve_binary(
        &self,
        operator: BinaryOperator,
        left: TypeConstraint,
        right: TypeConstraint,
        expected: Option<TypeId>,
    ) -> Result<ResolvedBinary, TypeError> {
        let mut matches = Vec::new();
        for signature in self.binary.iter().filter(|item| item.operator == operator) {
            let Some(left_type) = exact(signature.left) else {
                continue;
            };
            let Some(right_type) = resolve_pattern(signature.right, left_type, left_type) else {
                continue;
            };
            let Some(result) = resolve_pattern(signature.result, left_type, right_type) else {
                continue;
            };
            if constraint_matches(self, left, left_type)
                && constraint_matches(self, right, right_type)
                && expected.is_none_or(|expected| self.assignable(result, expected))
            {
                matches.push(ResolvedBinary {
                    left: left_type,
                    right: right_type,
                    result,
                    signature: *signature,
                });
            }
        }
        matches.sort_by_key(|item| (item.left, item.right, item.result));
        matches.dedup();
        match matches.as_slice() {
            [resolved] => Ok(*resolved),
            [] => Err(TypeError::NoMatchingOperator(operator)),
            _ => {
                if let Some(expected) = expected {
                    let exact: Vec<_> = matches
                        .iter()
                        .filter(|item| item.result == expected)
                        .copied()
                        .collect();
                    if let [resolved] = exact.as_slice() {
                        return Ok(*resolved);
                    }
                }
                let best_cost = matches
                    .iter()
                    .filter_map(|item| {
                        Some(
                            constraint_conversion_cost(self, left, item.left)?
                                + constraint_conversion_cost(self, right, item.right)?,
                        )
                    })
                    .min();
                if let Some(best_cost) = best_cost {
                    let best = matches
                        .iter()
                        .filter(|item| {
                            constraint_conversion_cost(self, left, item.left)
                                .zip(constraint_conversion_cost(self, right, item.right))
                                .is_some_and(|(left, right)| left + right == best_cost)
                        })
                        .collect::<Vec<_>>();
                    if let [resolved] = best.as_slice() {
                        return Ok(**resolved);
                    }
                }
                // Known operands and expected results are stronger constraints. If
                // only literals remain, choose the declared literal default after
                // considering every operator candidate.
                let default_kind = match (left, right) {
                    (TypeConstraint::Literal(kind), TypeConstraint::Literal(other))
                        if kind == other =>
                    {
                        Some(kind)
                    }
                    _ => None,
                };
                if let Some(default) = default_kind.and_then(|kind| self.defaults.get(&kind)) {
                    if let Some(found) = matches
                        .iter()
                        .find(|item| item.left == *default && item.right == *default)
                    {
                        return Ok(*found);
                    }
                }
                Err(TypeError::AmbiguousOperator(operator))
            }
        }
    }

    pub fn resolve_unary(
        &self,
        operator: UnaryOperator,
        operand: TypeConstraint,
        expected: Option<TypeId>,
    ) -> Result<ResolvedUnary, TypeError> {
        let matches: Vec<_> = self
            .unary
            .iter()
            .filter(|(known, input, result)| {
                *known == operator
                    && constraint_matches(self, operand, *input)
                    && expected.is_none_or(|expected| self.assignable(*result, expected))
            })
            .map(|(_, operand, result)| ResolvedUnary {
                operand: *operand,
                result: *result,
            })
            .collect();
        match matches.as_slice() {
            [resolved] => Ok(*resolved),
            [] => Err(TypeError::NoMatchingUnary(operator)),
            _ => {
                if let Some(expected) = expected {
                    let exact: Vec<_> = matches
                        .iter()
                        .filter(|item| item.result == expected)
                        .copied()
                        .collect();
                    if let [resolved] = exact.as_slice() {
                        return Ok(*resolved);
                    }
                }
                if let TypeConstraint::Literal(kind) = operand {
                    if let Some(default) = self.defaults.get(&kind) {
                        if let Some(resolved) = matches.iter().find(|item| item.operand == *default)
                        {
                            return Ok(*resolved);
                        }
                    }
                }
                Err(TypeError::AmbiguousUnary(operator))
            }
        }
    }
}

fn exact(pattern: TypePattern) -> Option<TypeId> {
    match pattern {
        TypePattern::Exact(id) => Some(id),
        _ => None,
    }
}

fn resolve_pattern(pattern: TypePattern, left: TypeId, right: TypeId) -> Option<TypeId> {
    Some(match pattern {
        TypePattern::Exact(id) => id,
        TypePattern::SameAsLeft => left,
        TypePattern::SameAsRight => right,
    })
}

fn constraint_matches(
    context: &TypeContext,
    constraint: TypeConstraint,
    candidate: TypeId,
) -> bool {
    match constraint {
        TypeConstraint::Known(actual) => context.assignable(actual, candidate),
        TypeConstraint::Literal(kind) => context.primitive(candidate).is_some_and(|primitive| {
            primitive.category.literal_kind() == Some(kind)
                || (kind == LiteralKind::Integer && primitive.category == PrimitiveCategory::Float)
        }),
    }
}

fn constraint_conversion_cost(
    context: &TypeContext,
    constraint: TypeConstraint,
    candidate: TypeId,
) -> Option<u32> {
    match constraint {
        TypeConstraint::Known(actual) if actual == candidate => Some(0),
        TypeConstraint::Known(actual) if context.assignable(actual, candidate) => {
            let actual = context.primitive(actual)?.representation;
            let candidate = context.primitive(candidate)?.representation;
            Some(match (actual, candidate) {
                (
                    PrimitiveRepresentation::Integer { .. }
                    | PrimitiveRepresentation::PointerInteger { .. },
                    PrimitiveRepresentation::Float { .. },
                ) => 100,
                _ => 1,
            })
        }
        TypeConstraint::Literal(kind) => {
            let primitive = context.primitive(candidate)?;
            if context
                .defaults
                .get(&kind)
                .is_some_and(|default| *default == candidate)
            {
                Some(0)
            } else if primitive.category.literal_kind() == Some(kind) {
                Some(1)
            } else if kind == LiteralKind::Integer && primitive.category == PrimitiveCategory::Float
            {
                Some(100)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn integer_width_fits(actual: IntegerWidth, expected: IntegerWidth) -> bool {
    match (actual, expected) {
        (IntegerWidth::Fixed(actual), IntegerWidth::Fixed(expected)) => actual <= expected,
        (IntegerWidth::Machine, IntegerWidth::Machine) => true,
        _ => false,
    }
}

fn float_width(format: FloatFormat) -> Option<u16> {
    match format {
        FloatFormat::Ieee(bits) => Some(bits),
        FloatFormat::BrainFloat16 => Some(16),
        FloatFormat::Machine => None,
    }
}

fn literal_fits(literal: &LiteralValue, representation: PrimitiveRepresentation) -> bool {
    match (literal, representation) {
        (LiteralValue::Integer(spelling), PrimitiveRepresentation::Integer { bits, signed }) => {
            integer_literal_fits(spelling, bits, signed)
        }
        (LiteralValue::Integer(spelling), PrimitiveRepresentation::PointerInteger { signed }) => {
            integer_literal_fits(spelling, IntegerWidth::Machine, signed)
        }
        (LiteralValue::Float(_), PrimitiveRepresentation::Float { .. })
        | (LiteralValue::Boolean(_), PrimitiveRepresentation::Boolean)
        | (LiteralValue::Character(_), PrimitiveRepresentation::Character)
        | (LiteralValue::String(_), PrimitiveRepresentation::String)
        | (LiteralValue::Bytes(_), PrimitiveRepresentation::Bytes)
        | (LiteralValue::None, PrimitiveRepresentation::None)
        | (LiteralValue::Unit, PrimitiveRepresentation::Unit) => true,
        _ => false,
    }
}

fn integer_literal_fits(spelling: &str, bits: IntegerWidth, signed: bool) -> bool {
    let Ok(value) = spelling.parse::<u128>() else {
        return matches!(bits, IntegerWidth::Machine);
    };
    match bits {
        IntegerWidth::Machine => true,
        IntegerWidth::Fixed(0) => false,
        IntegerWidth::Fixed(128) if !signed => true,
        IntegerWidth::Fixed(128) => value <= i128::MAX as u128,
        IntegerWidth::Fixed(bits) if signed => value < (1u128 << (bits - 1)),
        IntegerWidth::Fixed(bits) => value < (1u128 << bits),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeError {
    CompileRouteAlreadyDefined(TypeId),
    UnknownCompiler(CompilerId),
    GenericArity {
        constructor: TypeId,
        expected: usize,
        actual: usize,
    },
    DuplicateName(String),
    IdentityCollision(String, String),
    UnknownTypeId(TypeId),
    AlreadyDefined(String),
    UnknownCategory(String),
    UnknownRepresentation(String),
    MissingBitWidth,
    InvalidDefaultLiteralCategory,
    DuplicateDefault(LiteralKind),
    NoLiteralDefault(LiteralKind),
    InvalidLiteralForType(LiteralKind, TypeId),
    NoMatchingOperator(BinaryOperator),
    AmbiguousOperator(BinaryOperator),
    NoMatchingUnary(UnaryOperator),
    AmbiguousUnary(UnaryOperator),
}

impl fmt::Display for TypeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TypeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_context() -> (TypeContext, TypeId, TypeId, TypeId) {
        let mut types = TypeContextBuilder::new();
        let int = types.register_declaration("core.int", "int").unwrap();
        types
            .define_primitive(
                int,
                PrimitiveCategory::Integer,
                PrimitiveRepresentation::Integer {
                    bits: IntegerWidth::Machine,
                    signed: true,
                },
                true,
            )
            .unwrap();
        let i32 = types.register_declaration("core.i32", "i32").unwrap();
        types
            .define_primitive(
                i32,
                PrimitiveCategory::Integer,
                PrimitiveRepresentation::Integer {
                    bits: IntegerWidth::Fixed(32),
                    signed: true,
                },
                false,
            )
            .unwrap();
        let float = types.register_declaration("core.float", "float").unwrap();
        types
            .define_primitive(
                float,
                PrimitiveCategory::Float,
                PrimitiveRepresentation::Float {
                    format: FloatFormat::Machine,
                },
                true,
            )
            .unwrap();
        for ty in [int, i32, float] {
            types.add_binary(OperatorSignature {
                operator: BinaryOperator::Add,
                left: TypePattern::Exact(ty),
                right: TypePattern::Exact(ty),
                result: TypePattern::Exact(ty),
            });
        }
        (types.build(), int, i32, float)
    }

    #[test]
    fn literal_constraints_are_symmetric() {
        let (types, _, i32, _) = numeric_context();
        let left = types
            .resolve_binary(
                BinaryOperator::Add,
                TypeConstraint::Known(i32),
                TypeConstraint::Literal(LiteralKind::Integer),
                None,
            )
            .unwrap();
        let right = types
            .resolve_binary(
                BinaryOperator::Add,
                TypeConstraint::Literal(LiteralKind::Integer),
                TypeConstraint::Known(i32),
                None,
            )
            .unwrap();
        assert_eq!(left.result, i32);
        assert_eq!(right.result, i32);
    }

    #[test]
    fn mixed_integer_and_float_operators_promote_to_float() {
        let (types, int, _, float) = numeric_context();
        let resolved = types
            .resolve_binary(
                BinaryOperator::Add,
                TypeConstraint::Known(int),
                TypeConstraint::Known(float),
                None,
            )
            .unwrap();
        assert_eq!(resolved.left, float);
        assert_eq!(resolved.right, float);
        assert_eq!(resolved.result, float);
    }

    #[test]
    fn pointer_integer_representation_has_no_fixed_width() {
        let representation =
            PrimitiveRepresentation::from_contract("pointer-integer", None, false).unwrap();
        assert_eq!(
            representation,
            PrimitiveRepresentation::PointerInteger { signed: false }
        );
    }

    #[test]
    fn generic_instances_inherit_their_constructor_compile_route() {
        let mut types = TypeContextBuilder::new();
        let compiler = types
            .register_declaration("test.compiler", "TestCompiler")
            .unwrap();
        let compiler = types.compiler_id(compiler).unwrap();
        let family = types
            .register_generic_declaration("test.ir", "TestIR", 1)
            .unwrap();
        let f32 = types.register_declaration("core.f32", "f32").unwrap();
        let bf16 = types.register_declaration("core.bf16", "bf16").unwrap();
        types.set_compile_route(family, compiler).unwrap();
        let f32_instance = types.instantiate(family, vec![f32]).unwrap();
        let bf16_instance = types.instantiate(family, vec![bf16]).unwrap();
        let types = types.build();
        assert_eq!(
            types.compile_route(f32_instance).unwrap(),
            CompileRoute::Compiler(compiler)
        );
        assert_eq!(
            types.compile_route(bf16_instance).unwrap(),
            CompileRoute::Compiler(compiler)
        );
    }

    #[test]
    fn declaration_lookup_survives_inference_types_between_declarations() {
        let mut types = TypeContext::default();
        let first = types
            .register_declaration("test.First", "First", 0)
            .unwrap();
        let inference = types.fresh_infer();
        let union = types.union([first, inference]);
        let second = types
            .register_declaration("test.Second", "Second", 0)
            .unwrap();
        assert_ne!(second, union);
        assert_eq!(types.definition(first).unwrap().name, "First");
        assert_eq!(types.definition(second).unwrap().name, "Second");
        assert!(matches!(types.kind(union), Some(TypeKind::Union(_))));
    }
}
