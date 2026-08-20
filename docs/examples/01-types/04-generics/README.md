PrimitiveType
GenericType
NominalType
FunctionType
UnionType
TupleType
ReferenceType
CompileType

enum Type {
    Primitive(PrimitiveType),

    Named(TypeId),

    Applied {
        constructor: TypeId,
        args: Vec<TypeArg>,
    },

    Function {
        params: Vec<Type>,
        result: Box<Type>,
    },

    Union(Vec<Type>),

    Reference {
        kind: BorrowKind,
        inner: Box<Type>,
    },

    Compile(CompileType),
}