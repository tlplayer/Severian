use crate::{GenericParamId, TypeId};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GenericParamKind {
    Type,
    Dimension,
    Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeDimId(pub u32);

impl RuntimeDimId {
    const ANONYMOUS_BASE: u32 = 0xffff_0000;

    /// Creates a dynamic extent that carries no equality relationship. Named
    /// source dimensions use ordinary stable ids; synthesized `?` dimensions
    /// use this reserved range so two unrelated axes never unify by accident.
    pub fn anonymous(axis: usize) -> Self {
        let axis = u32::try_from(axis).unwrap_or(u32::MAX - Self::ANONYMOUS_BASE);
        Self(Self::ANONYMOUS_BASE.saturating_add(axis.min(0xffff)))
    }

    pub const fn is_anonymous(self) -> bool {
        self.0 >= Self::ANONYMOUS_BASE
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShapeParameterId(pub GenericParamId);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DimExpr {
    Constant(u64),
    Parameter(GenericParamId),
    Runtime(RuntimeDimId),
    Add(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
    DivideExact(Box<Self>, Box<Self>),
}

impl DimExpr {
    pub fn substitute(&self, arguments: &GenericArguments) -> Result<Self, GenericError> {
        match self {
            Self::Parameter(parameter) => match arguments.get(*parameter) {
                Some(GenericArgument::Dimension(dimension)) => Ok(dimension.clone()),
                Some(argument) => Err(GenericError::KindMismatch {
                    parameter: *parameter,
                    expected: GenericParamKind::Dimension,
                    found: argument.kind(),
                }),
                None => Ok(self.clone()),
            },
            Self::Add(left, right) => Ok(Self::Add(
                Box::new(left.substitute(arguments)?),
                Box::new(right.substitute(arguments)?),
            )
            .simplify()?),
            Self::Multiply(left, right) => Ok(Self::Multiply(
                Box::new(left.substitute(arguments)?),
                Box::new(right.substitute(arguments)?),
            )
            .simplify()?),
            Self::DivideExact(left, right) => Ok(Self::DivideExact(
                Box::new(left.substitute(arguments)?),
                Box::new(right.substitute(arguments)?),
            )
            .simplify()?),
            Self::Constant(_) | Self::Runtime(_) => Ok(self.clone()),
        }
    }

    pub fn simplify(self) -> Result<Self, GenericError> {
        match self {
            Self::Add(left, right) => match (*left, *right) {
                (Self::Constant(left), Self::Constant(right)) => left
                    .checked_add(right)
                    .map(Self::Constant)
                    .ok_or(GenericError::DimensionOverflow),
                (left, right) => Ok(Self::Add(Box::new(left), Box::new(right))),
            },
            Self::Multiply(left, right) => match (*left, *right) {
                (Self::Constant(left), Self::Constant(right)) => left
                    .checked_mul(right)
                    .map(Self::Constant)
                    .ok_or(GenericError::DimensionOverflow),
                (left, right) => Ok(Self::Multiply(Box::new(left), Box::new(right))),
            },
            Self::DivideExact(left, right) => match (*left, *right) {
                (_, Self::Constant(0)) => Err(GenericError::DivisionByZero),
                (Self::Constant(left), Self::Constant(right)) if left % right == 0 => {
                    Ok(Self::Constant(left / right))
                }
                (Self::Constant(left), Self::Constant(right)) => {
                    Err(GenericError::NonExactDivision { left, right })
                }
                (left, right) => Ok(Self::DivideExact(Box::new(left), Box::new(right))),
            },
            expression => Ok(expression),
        }
    }

    pub fn is_runtime_dynamic(&self) -> bool {
        match self {
            Self::Runtime(_) | Self::Parameter(_) => true,
            Self::Add(left, right)
            | Self::Multiply(left, right)
            | Self::DivideExact(left, right) => {
                left.is_runtime_dynamic() || right.is_runtime_dynamic()
            }
            Self::Constant(_) => false,
        }
    }

    /// Evaluates a dimension expression from generic and runtime bindings.
    /// An absent binding is not an error: it means the expression remains a
    /// runtime obligation and must not be mistaken for a concrete extent.
    pub fn evaluate(&self, bindings: &DimensionBindings) -> Result<Option<u64>, GenericError> {
        let binary = |left: &Self, right: &Self| -> Result<Option<(u64, u64)>, GenericError> {
            Ok(left.evaluate(bindings)?.zip(right.evaluate(bindings)?))
        };
        match self {
            Self::Constant(value) => Ok(Some(*value)),
            Self::Parameter(parameter) => Ok(bindings.parameters.get(parameter).copied()),
            Self::Runtime(runtime) => Ok(bindings.runtime.get(runtime).copied()),
            Self::Add(left, right) => binary(left, right)?
                .map(|(left, right)| {
                    left.checked_add(right)
                        .ok_or(GenericError::DimensionOverflow)
                })
                .transpose(),
            Self::Multiply(left, right) => binary(left, right)?
                .map(|(left, right)| {
                    left.checked_mul(right)
                        .ok_or(GenericError::DimensionOverflow)
                })
                .transpose(),
            Self::DivideExact(left, right) => binary(left, right)?
                .map(|(left, right)| {
                    if right == 0 {
                        Err(GenericError::DivisionByZero)
                    } else if left % right != 0 {
                        Err(GenericError::NonExactDivision { left, right })
                    } else {
                        Ok(left / right)
                    }
                })
                .transpose(),
        }
    }
}

/// Values known at a specialization boundary. Dimension parameters and
/// runtime dimensions deliberately occupy different namespaces: resolving a
/// language generic is not the same operation as reading a tensor descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DimensionBindings {
    parameters: BTreeMap<GenericParamId, u64>,
    runtime: BTreeMap<RuntimeDimId, u64>,
}

impl DimensionBindings {
    pub fn bind_parameter(
        &mut self,
        parameter: GenericParamId,
        value: u64,
    ) -> Result<(), GenericError> {
        bind_dimension_value(&mut self.parameters, parameter, value)
            .map_err(|_| GenericError::ConflictingDimensionParameter(parameter))
    }

    pub fn bind_runtime(&mut self, runtime: RuntimeDimId, value: u64) -> Result<(), GenericError> {
        bind_dimension_value(&mut self.runtime, runtime, value)
            .map_err(|_| GenericError::ConflictingRuntimeDimension(runtime))
    }

    pub fn parameter(&self, parameter: GenericParamId) -> Option<u64> {
        self.parameters.get(&parameter).copied()
    }

    pub fn runtime(&self, runtime: RuntimeDimId) -> Option<u64> {
        self.runtime.get(&runtime).copied()
    }
}

fn bind_dimension_value<K: Ord + Copy>(
    bindings: &mut BTreeMap<K, u64>,
    key: K,
    value: u64,
) -> Result<(), ()> {
    match bindings.get(&key) {
        Some(existing) if *existing != value => Err(()),
        Some(_) => Ok(()),
        None => {
            bindings.insert(key, value);
            Ok(())
        }
    }
}

/// The deliberately small shape-constraint language used before backend
/// emission. It covers the relationships needed by tensor contraction,
/// reshape, broadcast and bounded dynamic kernels without becoming a theorem
/// prover.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DimensionConstraint {
    Equal(DimExpr, DimExpr),
    Range {
        value: DimExpr,
        minimum: Option<u64>,
        maximum: Option<u64>,
    },
    MultipleOf {
        value: DimExpr,
        factor: u64,
    },
    ProductEqual {
        left: Vec<DimExpr>,
        right: Vec<DimExpr>,
    },
    BroadcastCompatible(DimExpr, DimExpr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintResolution {
    Proven,
    RuntimeCheck,
}

impl DimensionConstraint {
    pub fn resolve(
        &self,
        bindings: &DimensionBindings,
    ) -> Result<ConstraintResolution, GenericError> {
        let unresolved = || Ok(ConstraintResolution::RuntimeCheck);
        let satisfied = |condition| {
            if condition {
                Ok(ConstraintResolution::Proven)
            } else {
                Err(GenericError::UnsatisfiedDimensionConstraint(self.clone()))
            }
        };
        match self {
            Self::Equal(left, right) => match (left.evaluate(bindings)?, right.evaluate(bindings)?)
            {
                (Some(left), Some(right)) => satisfied(left == right),
                _ if left == right => Ok(ConstraintResolution::Proven),
                _ => unresolved(),
            },
            Self::Range {
                value,
                minimum,
                maximum,
            } => match value.evaluate(bindings)? {
                Some(value) => satisfied(
                    minimum.is_none_or(|minimum| value >= minimum)
                        && maximum.is_none_or(|maximum| value <= maximum),
                ),
                None => unresolved(),
            },
            Self::MultipleOf { value, factor } => {
                if *factor == 0 {
                    return Err(GenericError::DivisionByZero);
                }
                match value.evaluate(bindings)? {
                    Some(value) => satisfied(value % factor == 0),
                    None => unresolved(),
                }
            }
            Self::ProductEqual { left, right } => {
                let product = |expressions: &[DimExpr]| -> Result<Option<u64>, GenericError> {
                    expressions
                        .iter()
                        .try_fold(Some(1u64), |product, expression| {
                            Ok(match (product, expression.evaluate(bindings)?) {
                                (Some(product), Some(value)) => Some(
                                    product
                                        .checked_mul(value)
                                        .ok_or(GenericError::DimensionOverflow)?,
                                ),
                                _ => None,
                            })
                        })
                };
                match (product(left)?, product(right)?) {
                    (Some(left), Some(right)) => satisfied(left == right),
                    _ if left == right => Ok(ConstraintResolution::Proven),
                    _ => unresolved(),
                }
            }
            Self::BroadcastCompatible(left, right) => {
                match (left.evaluate(bindings)?, right.evaluate(bindings)?) {
                    (Some(left), Some(right)) => {
                        satisfied(left == right || left == 1 || right == 1)
                    }
                    _ if left == right => Ok(ConstraintResolution::Proven),
                    _ => unresolved(),
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenericArgument {
    Type(TypeId),
    Dimension(DimExpr),
    Shape(Vec<DimExpr>),
}

impl GenericArgument {
    pub const fn kind(&self) -> GenericParamKind {
        match self {
            Self::Type(_) => GenericParamKind::Type,
            Self::Dimension(_) => GenericParamKind::Dimension,
            Self::Shape(_) => GenericParamKind::Shape,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenericParameter {
    pub id: GenericParamId,
    pub name: String,
    pub kind: GenericParamKind,
    pub variadic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GenericArguments(BTreeMap<GenericParamId, GenericArgument>);

impl GenericArguments {
    pub fn bind(
        &mut self,
        parameter: &GenericParameter,
        argument: GenericArgument,
    ) -> Result<(), GenericError> {
        if parameter.kind != argument.kind() {
            return Err(GenericError::KindMismatch {
                parameter: parameter.id,
                expected: parameter.kind,
                found: argument.kind(),
            });
        }
        if parameter.variadic && parameter.kind != GenericParamKind::Shape {
            return Err(GenericError::InvalidVariadicKind(parameter.id));
        }
        if let Some(existing) = self.0.get(&parameter.id) {
            if existing != &argument {
                return Err(GenericError::ConflictingBinding(parameter.id));
            }
            return Ok(());
        }
        self.0.insert(parameter.id, argument);
        Ok(())
    }

    pub fn get(&self, parameter: GenericParamId) -> Option<&GenericArgument> {
        self.0.get(&parameter)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShapeTerm {
    Ranked(Vec<DimExpr>),
    Pack(ShapeParameterId),
}

impl ShapeTerm {
    pub fn rank(&self) -> Option<usize> {
        match self {
            Self::Ranked(dimensions) => Some(dimensions.len()),
            Self::Pack(_) => None,
        }
    }

    pub fn specialize(&self, arguments: &GenericArguments) -> Result<Self, GenericError> {
        match self {
            Self::Ranked(dimensions) => dimensions
                .iter()
                .map(|dimension| dimension.substitute(arguments))
                .collect::<Result<Vec<_>, _>>()
                .map(Self::Ranked),
            Self::Pack(parameter) => match arguments.get(parameter.0) {
                Some(GenericArgument::Shape(dimensions)) => Ok(Self::Ranked(dimensions.clone())),
                Some(argument) => Err(GenericError::KindMismatch {
                    parameter: parameter.0,
                    expected: GenericParamKind::Shape,
                    found: argument.kind(),
                }),
                None => Ok(self.clone()),
            },
        }
    }

    pub fn require_ranked(&self) -> Result<&[DimExpr], GenericError> {
        match self {
            Self::Ranked(dimensions) => Ok(dimensions),
            Self::Pack(parameter) => Err(GenericError::UnresolvedShapePack(*parameter)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedTensorType {
    pub element: TypeId,
    pub dimensions: Vec<DimExpr>,
}

impl RankedTensorType {
    pub fn new(element: TypeId, shape: &ShapeTerm) -> Result<Self, GenericError> {
        Ok(Self {
            element,
            dimensions: shape.require_ranked()?.to_vec(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenericError {
    KindMismatch {
        parameter: GenericParamId,
        expected: GenericParamKind,
        found: GenericParamKind,
    },
    InvalidVariadicKind(GenericParamId),
    ConflictingBinding(GenericParamId),
    ConflictingDimensionParameter(GenericParamId),
    ConflictingRuntimeDimension(RuntimeDimId),
    UnresolvedShapePack(ShapeParameterId),
    DimensionOverflow,
    DivisionByZero,
    NonExactDivision {
        left: u64,
        right: u64,
    },
    UnsatisfiedDimensionConstraint(DimensionConstraint),
}

impl fmt::Display for GenericError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for GenericError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_dimension_is_legal_but_unresolved_shape_pack_is_not_ranked() {
        let batch = GenericParamId(1);
        let ranked = ShapeTerm::Ranked(vec![DimExpr::Parameter(batch), DimExpr::Constant(1024)]);
        assert_eq!(ranked.rank(), Some(2));
        assert!(RankedTensorType::new(TypeId(7), &ranked).is_ok());

        let pack = ShapeTerm::Pack(ShapeParameterId(GenericParamId(2)));
        assert_eq!(pack.rank(), None);
        assert!(matches!(
            RankedTensorType::new(TypeId(7), &pack),
            Err(GenericError::UnresolvedShapePack(_))
        ));
    }

    #[test]
    fn shape_pack_binding_and_exact_dimension_arithmetic_specialize_structurally() {
        let parameter = GenericParameter {
            id: GenericParamId(4),
            name: "Shape".into(),
            kind: GenericParamKind::Shape,
            variadic: true,
        };
        let mut arguments = GenericArguments::default();
        arguments
            .bind(
                &parameter,
                GenericArgument::Shape(vec![
                    DimExpr::Runtime(RuntimeDimId(0)),
                    DimExpr::Constant(128),
                ]),
            )
            .unwrap();
        let specialized = ShapeTerm::Pack(ShapeParameterId(parameter.id))
            .specialize(&arguments)
            .unwrap();
        assert_eq!(specialized.rank(), Some(2));

        let expression = DimExpr::DivideExact(
            Box::new(DimExpr::Multiply(
                Box::new(DimExpr::Constant(8)),
                Box::new(DimExpr::Constant(128)),
            )),
            Box::new(DimExpr::Constant(8)),
        );
        assert_eq!(
            expression.substitute(&GenericArguments::default()).unwrap(),
            DimExpr::Constant(128)
        );
    }

    #[test]
    fn dimension_constraints_separate_proven_runtime_and_violated_facts() {
        let sequence = RuntimeDimId(7);
        let expression = DimExpr::Runtime(sequence);
        let constraints = [
            DimensionConstraint::Range {
                value: expression.clone(),
                minimum: Some(1),
                maximum: Some(4096),
            },
            DimensionConstraint::MultipleOf {
                value: expression,
                factor: 8,
            },
        ];

        let mut bindings = DimensionBindings::default();
        assert!(constraints.iter().all(|constraint| {
            constraint.resolve(&bindings) == Ok(ConstraintResolution::RuntimeCheck)
        }));

        bindings.bind_runtime(sequence, 512).unwrap();
        assert!(constraints.iter().all(|constraint| {
            constraint.resolve(&bindings) == Ok(ConstraintResolution::Proven)
        }));

        let mut invalid = DimensionBindings::default();
        invalid.bind_runtime(sequence, 513).unwrap();
        assert!(matches!(
            constraints[1].resolve(&invalid),
            Err(GenericError::UnsatisfiedDimensionConstraint(_))
        ));
    }

    #[test]
    fn product_and_broadcast_constraints_use_shared_symbolic_bindings() {
        let heads = GenericParamId(8);
        let head_width = RuntimeDimId(9);
        let mut bindings = DimensionBindings::default();
        bindings.bind_parameter(heads, 16).unwrap();
        bindings.bind_runtime(head_width, 128).unwrap();

        let product = DimensionConstraint::ProductEqual {
            left: vec![DimExpr::Constant(2048)],
            right: vec![DimExpr::Parameter(heads), DimExpr::Runtime(head_width)],
        };
        assert_eq!(
            product.resolve(&bindings).unwrap(),
            ConstraintResolution::Proven
        );
        assert_eq!(
            DimensionConstraint::BroadcastCompatible(
                DimExpr::Constant(1),
                DimExpr::Runtime(head_width),
            )
            .resolve(&bindings)
            .unwrap(),
            ConstraintResolution::Proven
        );
    }
}
