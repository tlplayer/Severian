use crate::Compilation;
use severian_coverage::{
    source_map::stable_region_id, CoverageRegion, CoverageRegionId, CoverageRegionKind,
    CoverageSourceMap, SourcePosition, SourceSpan,
};
use severian_hir::{
    CallTarget, DefinitionId, Expression, Function, HirId, Instruction, SourceFile, ValueType,
};

pub fn instrument(compilation: &Compilation) -> (Compilation, CoverageSourceMap) {
    let mut hir = compilation.optimized_hir.clone();
    let mut instrumenter = Instrumenter {
        sources: &hir.metadata.sources,
        map: CoverageSourceMap::default(),
        branch_ordinal: 0,
    };

    for function in &mut hir.functions {
        instrumenter.function(function);
    }
    for class in &mut hir.classes {
        for function in class.methods.iter_mut().chain(&mut class.constructors) {
            instrumenter.function(function);
        }
    }
    let map = instrumenter.map;
    let mir = severian_mir::lower(&hir);
    let mlir = severian_lowering::lower(&mir);
    (
        Compilation {
            hir: compilation.hir.clone(),
            optimized_hir: hir,
            mir,
            mlir,
        },
        map,
    )
}

struct Instrumenter<'a> {
    sources: &'a severian_hir::SourceMap,
    map: CoverageSourceMap,
    branch_ordinal: usize,
}

impl Instrumenter<'_> {
    fn function(&mut self, function: &mut Function) {
        self.branch_ordinal = 0;
        let function_region = self
            .sources
            .definition_span(DefinitionId::Function(function.id))
            .and_then(|span| self.region(&function.name, span, CoverageRegionKind::Function, None));
        let mut instructions = std::mem::take(&mut function.instructions);
        self.block(&function.name, &mut instructions);
        if let Some(id) = function_region {
            instructions.insert(0, hit(id));
        }
        function.instructions = instructions;
    }

    fn block(&mut self, function: &str, instructions: &mut Vec<Instruction>) {
        let mut output = Vec::with_capacity(instructions.len() * 2);
        for mut instruction in std::mem::take(instructions) {
            self.nested(function, &mut instruction);
            if let Some(span) = instruction_span(&instruction, self.sources) {
                if let Some(id) = self.region(function, span, CoverageRegionKind::Statement, None) {
                    output.push(hit(id));
                }
            }
            output.push(instruction);
        }
        *instructions = output;
    }

    fn nested(&mut self, function: &str, instruction: &mut Instruction) {
        match instruction {
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                let span = expression_span(condition, self.sources);
                self.branch(function, then_instructions, span);
                self.branch(function, else_instructions, span);
            }
            Instruction::While {
                instructions,
                condition,
                setup,
                ..
            } => {
                if let Some(setup) = setup {
                    self.nested(function, setup);
                }
                let span = expression_span(condition, self.sources);
                self.branch(function, instructions, span);
            }
            Instruction::For {
                instructions,
                iterable,
                setup,
                ..
            } => {
                if let Some(setup) = setup {
                    self.nested(function, setup);
                }
                let span = expression_span(iterable, self.sources);
                self.branch(function, instructions, span);
            }
            Instruction::Switch { value, arms } => {
                let span = expression_span(value, self.sources);
                for arm in arms {
                    self.branch(function, &mut arm.instructions, span);
                }
            }
            Instruction::ChannelSwitch {
                arms,
                repeat_condition,
                ..
            } => {
                let span = repeat_condition
                    .as_ref()
                    .and_then(|value| expression_span(value, self.sources));
                for arm in arms {
                    self.branch(function, &mut arm.instructions, span);
                }
            }
            Instruction::With { instructions, .. } => self.block(function, instructions),
            _ => {}
        }
    }

    fn branch(
        &mut self,
        function: &str,
        instructions: &mut Vec<Instruction>,
        fallback: Option<severian_hir::SourceSpan>,
    ) {
        let span = instructions
            .iter()
            .find_map(|instruction| instruction_span(instruction, self.sources))
            .or(fallback);
        self.block(function, instructions);
        let Some(span) = span else { return };
        let salt = self.branch_ordinal;
        self.branch_ordinal += 1;
        if let Some(id) = self.region(function, span, CoverageRegionKind::Branch, Some(salt)) {
            instructions.insert(0, hit(id));
        }
    }

    fn region(
        &mut self,
        function: &str,
        span: severian_hir::SourceSpan,
        kind: CoverageRegionKind,
        salt: Option<usize>,
    ) -> Option<CoverageRegionId> {
        let file = self.sources.file(span.file)?;
        let span = convert_span(file, span.range)?;
        // The canonical file/span is the source identity. A dependency may be
        // linked both qualified and unqualified across separate test targets;
        // including its linkage name would count the same source twice.
        let identity = salt.map_or_else(String::new, |salt| format!("branch#{salt}"));
        let id = stable_region_id(&identity, &span, kind);
        self.map.insert(CoverageRegion {
            id,
            function: function.to_owned(),
            span,
            kind,
        });
        Some(id)
    }
}

fn hit(id: CoverageRegionId) -> Instruction {
    Instruction::Evaluate(Expression::Typed {
        id: HirId::synthetic(id.0 & ((1 << 20) - 1)),
        ty: ValueType::Unit,
        expression: Box::new(Expression::Call {
            target: CallTarget::native("coverage.hit", "__sev_coverage_hit")
                .with_signature([ValueType::Int], ValueType::Unit),
            args: vec![Expression::Typed {
                id: HirId::synthetic((id.0 & ((1 << 20) - 1)) ^ 1),
                ty: ValueType::Int,
                expression: Box::new(Expression::Integer(id.0 as i64)),
            }],
        }),
    })
}

fn instruction_span(
    instruction: &Instruction,
    sources: &severian_hir::SourceMap,
) -> Option<severian_hir::SourceSpan> {
    let expression = match instruction {
        Instruction::Let { value, .. }
        | Instruction::TryLet { value, .. }
        | Instruction::Print(value)
        | Instruction::Assert(value)
        | Instruction::Evaluate(value) => value,
        Instruction::Assign { value, .. } => value,
        Instruction::Return(Some(value)) => value,
        Instruction::If { condition, .. } | Instruction::While { condition, .. } => condition,
        Instruction::For { iterable, .. } => iterable,
        Instruction::Switch { value, .. } => value,
        Instruction::ChannelSwitch {
            channels,
            repeat_condition,
            ..
        } => {
            return repeat_condition
                .as_ref()
                .and_then(|value| expression_span(value, sources))
                .or_else(|| {
                    channels
                        .first()
                        .and_then(|value| expression_span(value, sources))
                });
        }
        Instruction::With {
            resources,
            instructions,
            ..
        } => {
            return resources
                .first()
                .and_then(|value| expression_span(value, sources))
                .or_else(|| {
                    instructions
                        .iter()
                        .find_map(|value| instruction_span(value, sources))
                });
        }
        Instruction::Return(None) | Instruction::Break | Instruction::Continue => return None,
    };
    expression_span(expression, sources)
}

fn expression_span(
    expression: &Expression,
    sources: &severian_hir::SourceMap,
) -> Option<severian_hir::SourceSpan> {
    expression
        .hir_id()
        .and_then(|id| sources.expression_span(id))
}

fn convert_span(file: &SourceFile, range: severian_hir::SourceRange) -> Option<SourceSpan> {
    Some(SourceSpan {
        file: std::fs::canonicalize(&file.path).unwrap_or_else(|_| file.path.clone()),
        start: position(&file.source, range.start)?,
        end: position(&file.source, range.end)?,
    })
}

fn position(source: &str, byte: usize) -> Option<SourcePosition> {
    if byte > source.len() || !source.is_char_boundary(byte) {
        return None;
    }
    let prefix = &source[..byte];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() as u32 + 1;
    let column = prefix
        .rsplit_once('\n')
        .map_or(prefix.chars().count(), |(_, tail)| tail.chars().count()) as u32
        + 1;
    Some(SourcePosition { line, column, byte })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_one_based_and_utf8_aware() {
        assert_eq!(position("a\nλx", 4).unwrap().line, 2);
        assert_eq!(position("a\nλx", 4).unwrap().column, 2);
    }
}
