# Source compiler migration inventory

The executable entry is `sev_compiler/src/main.sev`, through
`boundaries/driver/src/pipeline/source.sev`. The Rust seed builds that entry;
it must never compile acceptance subjects on its behalf.

## Permanent boundary

The compiler retains declaration bootstrapping, source/module identities,
definition and type resolution, typed callable checking and specialization,
capability checking, canonical CFG construction and verification, and backend
operations. Source contracts must select syntax, operand protocols, effects,
evaluation order, and language semantics. A source declaration's removal must
disable its behavior even when the compiler executable is unchanged.

## Implementation and removal ledger

Paths below are relative to `sev_compiler/`. A row is not complete merely
because a small acceptance fixture passes.

| Area | Current implementation | Required source owner and retirement work |
| --- | --- | --- |
| Symbols | `frontend/lexer/src/syntax/mod.sev`, `scanner/mod.sev` | Resolved `G` syntax properties; remove closed compound-assignment token kinds and the bootstrap operator descriptor alternative. Preserve byte spans, lexical boundaries and definition identity. |
| Imports | `frontend/modules/src/source.sev` | Discover dependency declarations before bodies; resolve qualified identities, aliases and conflicts deterministically; invalidate per-compilation state when source changes. |
| Precedence | `frontend/parser/src/statement/mod.sev` | Registered contracts supply precedence, associativity and operand forms. Declaration grammar stays in the compiler. |
| Assignment | Same parser, `frontend/semantic/src/callable.sev` | Place-mutating source contracts; remove parser rewriting to scalar binary expressions. Preserve place/RHS evaluation order and writable storage requirements. |
| Definitions | `frontend/semantic/src/definitions.sev` | Shared definition registration for `G`, inherited requirements, aliases, ordinary traits, types, families and implementations. Retire separate descriptor registration once all consumers resolve definitions. |
| Compiler terms | `universal/expression/expression.sev`, `universal/grammar/contracts.sev`, `universal/cfg/` | G exposes the full metadata schema. Compiler frames distinguish source operands, runtime values, block handles and absence; control-flow metadata gates CFG access. Complete inherited metadata validation and the remaining effect/policy checks. |
| Semantic methods | `frontend/semantic/src/callable.sev`, `transforms/mir/src/callable.sev` | Control contracts execute bindings and typed CFG/lower calls; lazy boolean contracts expand their source expression bodies. Ordinary typed operator helpers remain callable implementations. General compiler-context helpers, arbitrary semantic control flow, `valid`, const evaluation and folding remain unfinished. |
| Generic operators | Same callable analyzer | Resolve requirements and implementations with substitutions and conversion relations. Remove spelling-only operator selection and scalar fallback selection. |
| Primitive operations | `universal/operator/scalar.sev`, `universal/primitive/numeric/operators.sev` | Actual `int.sev` and `float.sev` contracts, including families, policies and conversions. The numeric subset and compiler scalar table are migration aids to remove. |
| Truth and laziness | Callable expression analysis and source G semantic bodies | User truth and source-defined lazy boolean expressions pass native tests. Extend the compiler evaluator beyond the currently supported expression-body form. |
| Collections | Callable parser/analyzer and primitive library | Source indexing, iteration, truth, allocation and storage protocols; arbitrary places must not duplicate side effects. |
| Matching | `frontend/semantic/src/match.sev` | Source pattern/equality protocols and CFG edges; remove manufactured scalar equality and nested-if topology. |
| Control flow | `universal/cfg/`, `transforms/mir/src/callable.sev` | If, While, Break, Continue and Return run source semantic bodies into the module CFG table. Custom Body/ConditionBody grammars share that evaluator. Structured executable If/Loop/Choose nodes and their old MIR builder are removed. Typed errors and cleanup remain. |
| Ownership | `transforms/mir/src/ownership.sev`, `transforms/mir/src/cfg.sev` | The active pass checks value availability/dominance and records storage/edge aliases on the canonical CFG. String storage is promoted to block arguments for MLIR buffer deallocation. General resource loans, move/drop state and cleanup edges remain unimplemented; alias facts are not a complete ownership checker. |
| Backend | `transforms/mlir/src/emit/callable.sev` | MLIR consumes verified CFG blocks and terminators directly. Native tool pipelines lift CF for buffer ownership and convert the resulting UB dialect before LLVM translation. Scalar fallback dispatch still must be retired after real library migration. |
| Inspection | Driver pipeline and `transforms/mir/src/agent_ir.sev` | `--emit agent-ir` serializes the same verified CFG table consumed by MLIR, including definition provenance, block parameters, operations, values and terminators. It does not synthesize a separate graph. |
| Error values | Parser, callable analyzer, CFG and backend | Typed Error implementation selection, propagation, identity, recovery and cleanup. |

## Baseline observations

The installed seed accepts `(cd sev_compiler && ../target/debug/sev build)`;
it does not accept `--manifest-path`. The initial executable failed all thirty
migration tests: ordinary compilation first failed at `<` in the IO prelude.
Discovery omitted bare inherited contracts in trait bodies. After that repair,
the next common failure was a formatted literal being interpreted as a unary
operator because its token carried the interpolation syntax registry.

The nested match fixture's intended even-number check uses remainder `%`.
Its independently calculated totals are 3, 27 and **133**: the last case adds
1, 2, 4, 20 and 6 before returning with an additional 100. The older callable
fixture had both `//` and the incorrect final expectation 131.

Additional baseline repairs preserve typed diagnostics through callable and
variadic specialization, normalize NaN formatting to the source library's
existing `nan` expectation, and replace the obsolete generic-identity rejection
with a positive identity test and a negative unconstrained-operator test.
Isolated parser/HIR tests now supply their syntax fixtures explicitly instead
of depending on an implicit compiler operator table.

Operator constraints use `[G:Add]`, following `[parameter:constraint]`. The
parser retains those ordinary generic parameters and constraints; registration
resolves the bound definition and distinguishes compiler grammar parameters
from runtime type parameters. `[Syntax:Fuse]` exercises the same resolution.

Run the migration, callable, diagnostic, bootstrap and semantic-IR suites after
building. Keep their failures separate: thirty green migration tests are an
acceptance floor, not completion evidence for all rows above.
