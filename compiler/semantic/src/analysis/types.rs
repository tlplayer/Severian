//! Optimizer-facing HIR type analysis.
//!
//! HIR deliberately stores only the type information needed by lowering. This
//! analysis reconstructs useful binding, tensor, task, class, and call-result
//! facts without re-running the source-language semantic checker.

use severian_hir::{
    BinaryOp, Class, Expression, Function, Instruction, MatchPattern, Program, TensorElementType,
    TensorType, UnaryOp, ValueType,
};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    pub ty: ValueType,
    /// Set for a runtime task handle whose await result is statically known.
    pub task_return: Option<ValueType>,
    /// Set for constructed class instances where HIR's `ValueType::Any`
    /// intentionally does not encode nominal class identity.
    pub class_name: Option<String>,
}

impl TypeInfo {
    pub const fn plain(ty: ValueType) -> Self {
        Self {
            ty,
            task_return: None,
            class_name: None,
        }
    }

    pub fn class(name: impl Into<String>) -> Self {
        Self {
            ty: ValueType::Any,
            task_return: None,
            class_name: Some(name.into()),
        }
    }

    pub const fn task(return_type: ValueType) -> Self {
        Self {
            ty: ValueType::Any,
            task_return: Some(return_type),
            class_name: None,
        }
    }
}

impl Default for TypeInfo {
    fn default() -> Self {
        Self::plain(ValueType::Any)
    }
}

#[derive(Debug, Clone, Default)]
pub struct FunctionTypeFacts {
    pub function: String,
    pub bindings: HashMap<String, TypeInfo>,
    pub return_type: ValueType,
}

impl FunctionTypeFacts {
    pub fn binding(&self, name: &str) -> Option<&TypeInfo> {
        self.bindings.get(name)
    }

    pub fn type_of_binding(&self, name: &str) -> Option<ValueType> {
        self.binding(name).map(|info| info.ty)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TypeAnalysis {
    pub globals: HashMap<String, TypeInfo>,
    pub functions: HashMap<String, FunctionTypeFacts>,
    signatures: HashMap<String, ValueType>,
    classes: HashMap<String, ClassInfo>,
}

#[derive(Debug, Clone, Default)]
struct ClassInfo {
    fields: HashMap<String, ValueType>,
    methods: HashMap<String, ValueType>,
}

impl TypeAnalysis {
    pub fn function(&self, name: &str) -> Option<&FunctionTypeFacts> {
        self.functions.get(name)
    }

    pub fn return_type(&self, function: &str) -> Option<ValueType> {
        self.signatures.get(function).copied()
    }

    pub fn infer_expression(
        &self,
        expression: &Expression,
        bindings: &HashMap<String, TypeInfo>,
    ) -> TypeInfo {
        infer_expression(
            expression,
            bindings,
            &self.signatures,
            &self.classes,
        )
    }
}

pub fn analyze(program: &Program) -> TypeAnalysis {
    let signatures = collect_signatures(program);
    let classes = collect_classes(program);

    let mut globals = HashMap::new();
    for global in &program.globals {
        let info = infer_expression(
            &global.value,
            &globals,
            &signatures,
            &classes,
        );
        globals.insert(global.name.clone(), info);
    }

    let mut functions = HashMap::new();

    for function in &program.functions {
        functions.insert(
            function.name.clone(),
            analyze_function_with_environment(
                function,
                &globals,
                &signatures,
                &classes,
            ),
        );
    }

    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            let qualified = format!("{}::{}", class.name, function.name);
            let mut facts = analyze_function_with_environment(
                function,
                &globals,
                &signatures,
                &classes,
            );
            facts.function = qualified.clone();
            functions.insert(qualified, facts);
        }
    }

    TypeAnalysis {
        globals,
        functions,
        signatures,
        classes,
    }
}

pub fn analyze_function(function: &Function) -> FunctionTypeFacts {
    let signatures = HashMap::from([(function.name.clone(), function.return_type)]);
    analyze_function_with_environment(
        function,
        &HashMap::new(),
        &signatures,
        &HashMap::new(),
    )
}

fn analyze_function_with_environment(
    function: &Function,
    globals: &HashMap<String, TypeInfo>,
    signatures: &HashMap<String, ValueType>,
    classes: &HashMap<String, ClassInfo>,
) -> FunctionTypeFacts {
    let mut bindings = globals.clone();

    for parameter in &function.params {
        bindings.insert(parameter.name.clone(), TypeInfo::plain(parameter.ty));
    }

    analyze_instructions(
        &function.instructions,
        &mut bindings,
        signatures,
        classes,
    );

    FunctionTypeFacts {
        function: function.name.clone(),
        bindings,
        return_type: function.return_type,
    }
}

fn collect_signatures(program: &Program) -> HashMap<String, ValueType> {
    let mut signatures = HashMap::new();

    for function in &program.functions {
        signatures.insert(function.name.clone(), function.return_type);
    }

    for class in &program.classes {
        for function in &class.methods {
            signatures.insert(
                format!("{}::{}", class.name, function.name),
                function.return_type,
            );
        }
        for constructor in &class.constructors {
            signatures.insert(
                format!("{}::{}", class.name, constructor.name),
                ValueType::Any,
            );
        }
    }

    signatures
}

fn collect_classes(program: &Program) -> HashMap<String, ClassInfo> {
    let mut result = HashMap::new();

    for class in &program.classes {
        let fields = class
            .fields
            .iter()
            .cloned()
            .zip(class.field_types.iter().copied())
            .collect();

        let methods = class
            .methods
            .iter()
            .map(|method| (method.name.clone(), method.return_type))
            .collect();

        result.insert(class.name.clone(), ClassInfo { fields, methods });
    }

    result
}

fn analyze_instructions(
    instructions: &[Instruction],
    bindings: &mut HashMap<String, TypeInfo>,
    signatures: &HashMap<String, ValueType>,
    classes: &HashMap<String, ClassInfo>,
) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
                let info = infer_expression(value, bindings, signatures, classes);
                bindings.insert(name.clone(), info);
            }

            Instruction::Assign { target, value, .. } => {
                let info = infer_expression(value, bindings, signatures, classes);
                if let Expression::Variable(name) = target {
                    bindings.insert(name.clone(), info);
                }
            }

            Instruction::If {
                then_instructions,
                else_instructions,
                ..
            } => {
                let before = bindings.clone();
                let mut then_bindings = before.clone();
                analyze_instructions(
                    then_instructions,
                    &mut then_bindings,
                    signatures,
                    classes,
                );

                let mut else_bindings = before;
                analyze_instructions(
                    else_instructions,
                    &mut else_bindings,
                    signatures,
                    classes,
                );

                *bindings = merge_bindings(&then_bindings, &else_bindings);
            }

            Instruction::While {
                setup,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    analyze_instructions(
                        std::slice::from_ref(setup.as_ref()),
                        bindings,
                        signatures,
                        classes,
                    );
                }

                let before = bindings.clone();
                let mut body = before.clone();
                analyze_instructions(instructions, &mut body, signatures, classes);
                *bindings = merge_bindings(&before, &body);
            }

            Instruction::For {
                setup,
                pattern,
                iterable,
                instructions,
            } => {
                if let Some(setup) = setup {
                    analyze_instructions(
                        std::slice::from_ref(setup.as_ref()),
                        bindings,
                        signatures,
                        classes,
                    );
                }

                let iterable_type =
                    infer_expression(iterable, bindings, signatures, classes);
                let element = iterable_element_type(&iterable_type);

                let before = bindings.clone();
                let mut body = before.clone();
                bind_pattern(pattern, element, &mut body);
                analyze_instructions(instructions, &mut body, signatures, classes);
                *bindings = merge_bindings(&before, &body);
            }

            Instruction::Switch { arms, .. } | Instruction::ChannelSwitch { arms, .. } => {
                if arms.is_empty() {
                    continue;
                }

                let before = bindings.clone();
                let mut branches = Vec::with_capacity(arms.len());

                for arm in arms {
                    let mut branch = before.clone();
                    bind_pattern(&arm.pattern, TypeInfo::default(), &mut branch);
                    analyze_instructions(
                        &arm.instructions,
                        &mut branch,
                        signatures,
                        classes,
                    );
                    branches.push(branch);
                }

                let mut merged = branches.remove(0);
                for branch in branches {
                    merged = merge_bindings(&merged, &branch);
                }
                *bindings = merged;
            }

            Instruction::With { instructions, .. } => {
                analyze_instructions(instructions, bindings, signatures, classes);
            }

            Instruction::Print(_)
            | Instruction::Assert(_)
            | Instruction::Return(_)
            | Instruction::Break
            | Instruction::Continue
            | Instruction::Evaluate(_) => {}
        }
    }
}

fn bind_pattern(
    pattern: &MatchPattern,
    info: TypeInfo,
    bindings: &mut HashMap<String, TypeInfo>,
) {
    match pattern {
        MatchPattern::Bind(name) => {
            bindings.insert(name.clone(), info);
        }
        MatchPattern::Constructor { fields, .. } => {
            for field in fields {
                bind_pattern(field, TypeInfo::default(), bindings);
            }
        }
        MatchPattern::Wildcard
        | MatchPattern::Integer(_)
        | MatchPattern::Float(_)
        | MatchPattern::Boolean(_)
        | MatchPattern::String(_) => {}
    }
}

fn merge_bindings(
    left: &HashMap<String, TypeInfo>,
    right: &HashMap<String, TypeInfo>,
) -> HashMap<String, TypeInfo> {
    let mut merged = HashMap::new();

    for name in left.keys().chain(right.keys()) {
        let info = match (left.get(name), right.get(name)) {
            (Some(left), Some(right)) => join(left, right),
            (Some(value), None) | (None, Some(value)) => value.clone(),
            (None, None) => continue,
        };
        merged.insert(name.clone(), info);
    }

    merged
}

fn join(left: &TypeInfo, right: &TypeInfo) -> TypeInfo {
    if left == right {
        return left.clone();
    }

    let ty = join_value_types(left.ty, right.ty);
    TypeInfo {
        ty,
        task_return: if left.task_return == right.task_return {
            left.task_return
        } else {
            None
        },
        class_name: if left.class_name == right.class_name {
            left.class_name.clone()
        } else {
            None
        },
    }
}

fn join_value_types(left: ValueType, right: ValueType) -> ValueType {
    if left == right {
        return left;
    }

    match (left, right) {
        (ValueType::Tensor(left), ValueType::Tensor(right)) => {
            if let Ok(tensor) = left.broadcast_with(right) {
                ValueType::Tensor(tensor)
            } else {
                ValueType::Any
            }
        }
        (ValueType::Int, ValueType::Float) | (ValueType::Float, ValueType::Int) => {
            ValueType::Float
        }
        _ => ValueType::Any,
    }
}

fn infer_expression(
    expression: &Expression,
    bindings: &HashMap<String, TypeInfo>,
    signatures: &HashMap<String, ValueType>,
    classes: &HashMap<String, ClassInfo>,
) -> TypeInfo {
    match expression {
        Expression::Integer(_) => TypeInfo::plain(ValueType::Int),
        Expression::Float(_) => TypeInfo::plain(ValueType::Float),
        Expression::Boolean(_) => TypeInfo::plain(ValueType::Bool),
        Expression::String(_) => TypeInfo::plain(ValueType::String),

        Expression::Variable(name) => bindings.get(name).cloned().unwrap_or_default(),

        Expression::Function(_) | Expression::Lambda { .. } => {
            TypeInfo::plain(ValueType::Function)
        }

        Expression::Ownership { value, .. } => {
            infer_expression(value, bindings, signatures, classes)
        }

        Expression::List(_) | Expression::ListComprehension { .. } => {
            TypeInfo::plain(ValueType::List)
        }

        Expression::Tuple(_) => TypeInfo::plain(ValueType::Tuple),

        Expression::Map(_) | Expression::MapComprehension { .. } => {
            TypeInfo::plain(ValueType::Map)
        }

        Expression::Set(_) | Expression::SetComprehension { .. } => {
            TypeInfo::plain(ValueType::Set)
        }

        Expression::Index { object, .. } => {
            let object = infer_expression(object, bindings, signatures, classes);
            match object.ty {
                ValueType::String => TypeInfo::plain(ValueType::String),
                ValueType::Tensor(tensor) => {
                    TypeInfo::plain(tensor_index_result(tensor))
                }
                _ => TypeInfo::default(),
            }
        }

        Expression::Slice { object, .. } => {
            infer_expression(object, bindings, signatures, classes)
        }

        Expression::Format { .. } => TypeInfo::plain(ValueType::String),

        Expression::PrintArgs(_) => TypeInfo::plain(ValueType::Tuple),

        Expression::Construct { class, .. } => TypeInfo::class(class.clone()),

        Expression::Member { object, member } => {
            let object = infer_expression(object, bindings, signatures, classes);
            object
                .class_name
                .as_deref()
                .and_then(|class| classes.get(class))
                .and_then(|class| class.fields.get(member))
                .copied()
                .map(TypeInfo::plain)
                .unwrap_or_default()
        }

        Expression::MethodCall { object, method, .. } => {
            let object = infer_expression(object, bindings, signatures, classes);
            object
                .class_name
                .as_deref()
                .and_then(|class| classes.get(class))
                .and_then(|class| class.methods.get(method))
                .copied()
                .map(TypeInfo::plain)
                .unwrap_or_default()
        }

        Expression::Variant { .. } => TypeInfo::plain(ValueType::Any),

        Expression::Task { value, .. } => {
            let inner = infer_expression(value, bindings, signatures, classes);
            TypeInfo::task(inner.ty)
        }

        Expression::Await(value) => {
            let task = infer_expression(value, bindings, signatures, classes);
            TypeInfo::plain(task.task_return.unwrap_or(ValueType::Any))
        }

        Expression::Channel(_) => TypeInfo::plain(ValueType::Channel),

        // Sending is asynchronous in the current runtime ABI and returns a
        // task-like handle that awaits to unit.
        Expression::Send { .. } => TypeInfo::task(ValueType::Unit),

        Expression::ChaosRule { value, .. } => {
            infer_expression(value, bindings, signatures, classes)
        }

        Expression::Conditional {
            then_expression,
            else_expression,
            ..
        } => {
            let left =
                infer_expression(then_expression, bindings, signatures, classes);
            let right =
                infer_expression(else_expression, bindings, signatures, classes);
            join(&left, &right)
        }

        Expression::FusedPipeline { input, .. } => {
            infer_expression(input, bindings, signatures, classes)
        }

        Expression::Unary { op, expression } => {
            let value = infer_expression(expression, bindings, signatures, classes);
            match op {
                UnaryOp::Negate => value,
                UnaryOp::Not => TypeInfo::plain(ValueType::Bool),
            }
        }

        Expression::Binary { left, op, right } => {
            let left = infer_expression(left, bindings, signatures, classes);
            let right = infer_expression(right, bindings, signatures, classes);
            infer_binary(left, *op, right)
        }

        Expression::Call { function, .. } => signatures
            .get(function)
            .copied()
            .map(TypeInfo::plain)
            .unwrap_or_default(),

        Expression::CallValue { return_type, .. } => TypeInfo::plain(*return_type),
    }
}

fn infer_binary(left: TypeInfo, op: BinaryOp, right: TypeInfo) -> TypeInfo {
    match op {
        BinaryOp::Equal
        | BinaryOp::NotEqual
        | BinaryOp::Less
        | BinaryOp::LessEqual
        | BinaryOp::Greater
        | BinaryOp::GreaterEqual
        | BinaryOp::And
        | BinaryOp::Or
        | BinaryOp::In => TypeInfo::plain(ValueType::Bool),

        BinaryOp::Add
        | BinaryOp::Sub
        | BinaryOp::Mul
        | BinaryOp::Div
        | BinaryOp::Mod
        | BinaryOp::Power => {
            let ty = match (left.ty, right.ty) {
                (ValueType::Tensor(left), ValueType::Tensor(right)) => left
                    .broadcast_with(right)
                    .map(ValueType::Tensor)
                    .unwrap_or(ValueType::Any),

                (ValueType::Tensor(tensor), scalar)
                | (scalar, ValueType::Tensor(tensor))
                    if scalar_compatible_with_tensor(scalar, tensor) =>
                {
                    ValueType::Tensor(tensor)
                }

                (ValueType::Float, ValueType::Float)
                | (ValueType::Float, ValueType::Int)
                | (ValueType::Int, ValueType::Float) => ValueType::Float,

                (ValueType::Int, ValueType::Int) => {
                    if op == BinaryOp::Div {
                        ValueType::Float
                    } else {
                        ValueType::Int
                    }
                }

                (ValueType::String, ValueType::String) if op == BinaryOp::Add => {
                    ValueType::String
                }

                _ => ValueType::Any,
            };

            TypeInfo::plain(ty)
        }
    }
}

fn scalar_compatible_with_tensor(scalar: ValueType, tensor: TensorType) -> bool {
    matches!(
        (scalar, tensor.element),
        (ValueType::Float, TensorElementType::F32 | TensorElementType::F64)
            | (ValueType::Int, TensorElementType::I32 | TensorElementType::I64)
    )
}

fn tensor_index_result(tensor: TensorType) -> ValueType {
    match tensor.rank {
        Some(0) | Some(1) => scalar_type(tensor.element),

        Some(rank) => {
            let rank = rank as usize;
            let mut dimensions = [severian_hir::TensorDimension::Dynamic; 8];
            dimensions[..rank - 1].copy_from_slice(&tensor.dimensions[1..rank]);

            ValueType::Tensor(TensorType {
                element: tensor.element,
                rank: Some((rank - 1) as u8),
                dimensions,
            })
        }

        None => ValueType::Tensor(TensorType::dynamic(tensor.element)),
    }
}

fn scalar_type(element: TensorElementType) -> ValueType {
    match element {
        TensorElementType::F32 | TensorElementType::F64 => ValueType::Float,
        TensorElementType::I32 | TensorElementType::I64 => ValueType::Int,
    }
}

fn iterable_element_type(iterable: &TypeInfo) -> TypeInfo {
    match iterable.ty {
        ValueType::String => TypeInfo::plain(ValueType::String),
        ValueType::Tensor(tensor) => TypeInfo::plain(tensor_index_result(tensor)),
        ValueType::List
        | ValueType::Tuple
        | ValueType::Map
        | ValueType::Set
        | ValueType::Any => TypeInfo::default(),
        _ => TypeInfo::default(),
    }
}
