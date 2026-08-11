use crate::Compilation;
use severian_hir::{BinaryOp, Expression, HirId, SourceFile, SourceSpan};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Mutation {
    pub index: usize,
    pub description: String,
    pub file: Option<PathBuf>,
    pub line: Option<u32>,
}

pub fn count(compilation: &Compilation) -> usize {
    let mut program = production_program(compilation);
    let mut count = 0;
    program.visit_expressions_mut(&mut |expression| {
        if mutation(expression).is_some() {
            count += 1;
        }
    });
    count
}

pub fn apply(compilation: &Compilation, wanted: usize) -> Option<(Compilation, Mutation)> {
    let mut hir = production_program(compilation);
    let sources = hir.metadata.sources.clone();
    let mut current = 0;
    let mut selected = None;
    hir.visit_expressions_mut(&mut |expression| {
        let Some((id, replacement, description)) = mutation(expression) else {
            return;
        };
        if current == wanted {
            let (file, line) = location(&sources, id);
            *expression = replacement;
            selected = Some(Mutation {
                index: wanted,
                description,
                file,
                line,
            });
        }
        current += 1;
    });
    let mutation = selected?;
    let mir = severian_mir::lower(&hir);
    let mlir = severian_lowering::lower(&mir);
    Some((
        Compilation {
            hir: compilation.hir.clone(),
            optimized_hir: hir,
            mir,
            mlir,
        },
        mutation,
    ))
}

fn production_program(compilation: &Compilation) -> severian_hir::Program {
    let mut program = compilation.optimized_hir.clone();
    for function in &mut program.functions {
        function.tests.clear();
    }
    for class in &mut program.classes {
        for function in class.methods.iter_mut().chain(&mut class.constructors) {
            function.tests.clear();
        }
    }
    program
}

fn mutation(expression: &Expression) -> Option<(HirId, Expression, String)> {
    let Expression::Typed {
        id,
        ty,
        expression: inner,
    } = expression
    else {
        return None;
    };
    let (replacement, description) = match inner.as_ref() {
        Expression::Boolean(value) => (
            Expression::Boolean(!value),
            format!("{} -> {}", value, !value),
        ),
        Expression::Binary { left, op, right } => {
            let replacement = replacement_op(*op)?;
            (
                Expression::Binary {
                    left: left.clone(),
                    op: replacement,
                    right: right.clone(),
                },
                format!("{} -> {}", op_name(*op), op_name(replacement)),
            )
        }
        _ => return None,
    };
    Some((
        *id,
        Expression::Typed {
            id: *id,
            ty: *ty,
            expression: Box::new(replacement),
        },
        description,
    ))
}

fn replacement_op(op: BinaryOp) -> Option<BinaryOp> {
    Some(match op {
        BinaryOp::Add => BinaryOp::Sub,
        BinaryOp::Sub => BinaryOp::Add,
        BinaryOp::Mul => BinaryOp::Div,
        BinaryOp::Div => BinaryOp::Mul,
        BinaryOp::Equal => BinaryOp::NotEqual,
        BinaryOp::NotEqual => BinaryOp::Equal,
        BinaryOp::Less => BinaryOp::LessEqual,
        BinaryOp::LessEqual => BinaryOp::Less,
        BinaryOp::Greater => BinaryOp::GreaterEqual,
        BinaryOp::GreaterEqual => BinaryOp::Greater,
        BinaryOp::And => BinaryOp::Or,
        BinaryOp::Or => BinaryOp::And,
        BinaryOp::Mod | BinaryOp::Power | BinaryOp::In => return None,
    })
}

fn op_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
        BinaryOp::Power => "**",
        BinaryOp::Equal => "==",
        BinaryOp::NotEqual => "!=",
        BinaryOp::Less => "<",
        BinaryOp::LessEqual => "<=",
        BinaryOp::Greater => ">",
        BinaryOp::GreaterEqual => ">=",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        BinaryOp::In => "in",
    }
}

fn location(sources: &severian_hir::SourceMap, id: HirId) -> (Option<PathBuf>, Option<u32>) {
    let Some(SourceSpan { file, range }) = sources.expression_span(id) else {
        return (None, None);
    };
    let Some(file) = sources.file(file) else {
        return (None, None);
    };
    (Some(file.path.clone()), line(file, range.start))
}

fn line(file: &SourceFile, byte: usize) -> Option<u32> {
    (byte <= file.source.len()).then(|| {
        file.source[..byte]
            .bytes()
            .filter(|value| *value == b'\n')
            .count() as u32
            + 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_mutations_cross_the_boundary() {
        assert_eq!(
            replacement_op(BinaryOp::Greater),
            Some(BinaryOp::GreaterEqual)
        );
        assert_eq!(replacement_op(BinaryOp::LessEqual), Some(BinaryOp::Less));
    }
}
