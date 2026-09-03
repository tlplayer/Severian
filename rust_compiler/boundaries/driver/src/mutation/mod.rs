mod apply;
mod discover;
mod model;
mod report;
mod runner;

pub(crate) use runner::run;

#[cfg(test)]
mod tests {
    use super::model::MutationKind;
    use super::{apply, discover};
    use severian_ast::{BinaryOperator, ExpressionKind, Item, Statement};
    use severian_modules::{ModuleGraph, ModuleId, PackageId, ResolvedModule};
    use severian_source::SourceFile;
    use std::path::PathBuf;

    fn graph(text: &str) -> ModuleGraph {
        let source = SourceFile::virtual_source("fixture.sev", text);
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        ModuleGraph {
            modules: vec![ResolvedModule {
                id: ModuleId(1),
                path: PathBuf::from("fixture.sev"),
                source,
                package: PackageId(0),
                ast,
                imports: Vec::new(),
            }],
        }
    }

    #[test]
    fn discovers_production_mutations_but_not_test_assertions() {
        let graph = graph(
            "def positive(value: int) -> bool:\n    if true and value > 0:\n        return value + 1 > 1\n    return false\n\ntest:\n    assert(positive(10) == true)\n",
        );
        let mutations = discover::discover(&graph).unwrap();
        let edits = mutations
            .iter()
            .map(|mutation| format!("{} -> {}", mutation.original, mutation.replacement))
            .collect::<Vec<_>>();
        assert!(edits.contains(&"condition -> !condition".into()));
        assert!(edits.contains(&"and -> or".into()));
        assert!(edits.contains(&"> -> >=".into()));
        assert!(edits.contains(&"+ -> -".into()));
        assert!(edits.contains(&"true -> false".into()));
        assert!(edits.contains(&"false -> true".into()));
        assert!(!edits.contains(&"== -> !=".into()));
        assert_eq!(
            mutations
                .iter()
                .map(|mutation| mutation.id)
                .collect::<Vec<_>>(),
            (1..=mutations.len()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn applying_one_candidate_changes_exactly_one_ast_node() {
        let mut graph = graph("def add(left: int, right: int) -> int:\n    return left + right\n");
        let mutation = discover::discover(&graph)
            .unwrap()
            .into_iter()
            .find(|mutation| mutation.original == "+")
            .unwrap();
        assert!(apply::apply(&mut graph, &mutation));
        let Item::Function(function) = &graph.modules[0].ast.items[0] else {
            panic!("expected function")
        };
        let Statement::Return {
            value: Some(value), ..
        } = &function.body.as_ref().unwrap()[0]
        else {
            panic!("expected return")
        };
        assert!(matches!(
            value.kind,
            ExpressionKind::Binary {
                operator: BinaryOperator::Subtract,
                ..
            }
        ));
    }

    #[test]
    fn discovers_every_initial_binary_operator_pair() {
        let graph = graph(
            "equal = 1 == 1\nnot_equal = 1 != 1\nless = 1 < 1\nless_equal = 1 <= 1\ngreater = 1 > 1\ngreater_equal = 1 >= 1\nadd = 1 + 1\nsubtract = 1 - 1\nmultiply = 1 * 1\ndivide = 1 / 1\nconjunction = true and false\ndisjunction = true or false\n",
        );
        let edits = discover::discover(&graph)
            .unwrap()
            .into_iter()
            .filter(|mutation| mutation.kind != MutationKind::BooleanLiteral)
            .map(|mutation| format!("{} -> {}", mutation.original, mutation.replacement))
            .collect::<Vec<_>>();
        for expected in [
            "== -> !=",
            "!= -> ==",
            "< -> <=",
            "<= -> <",
            "> -> >=",
            ">= -> >",
            "+ -> -",
            "- -> +",
            "* -> /",
            "/ -> *",
            "and -> or",
            "or -> and",
        ] {
            assert!(
                edits.iter().any(|edit| edit == expected),
                "missing {expected}"
            );
        }
        assert_eq!(edits.len(), 12);
    }

    #[test]
    fn conditional_mutation_negates_the_if_condition() {
        let mut graph = graph(
            "def choose(value: bool) -> bool:\n    if value:\n        return true\n    return false\n",
        );
        let mutation = discover::discover(&graph)
            .unwrap()
            .into_iter()
            .find(|mutation| mutation.kind == MutationKind::Conditional)
            .unwrap();
        assert!(apply::apply(&mut graph, &mutation));
        let Item::Function(function) = &graph.modules[0].ast.items[0] else {
            panic!("expected function")
        };
        let Statement::If { condition, .. } = &function.body.as_ref().unwrap()[0] else {
            panic!("expected if statement")
        };
        assert!(matches!(
            condition.kind,
            ExpressionKind::Unary {
                operator: severian_ast::UnaryOperator::Not,
                ..
            }
        ));
    }
}
