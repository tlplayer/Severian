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
    UnresolvedShapePack(ShapeParameterId),
    DimensionOverflow,
    DivisionByZero,
    NonExactDivision {
        left: u64,
        right: u64,
    },
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
}
