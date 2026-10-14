Language Milestones

Status: *Draft* | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors: Timothy PLayer
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

## Context

## Goal(s)

- Source text goes to MLIR (see )
-- syntax ->(grammars) source + lexer (token stream) -> parser + syntax -> ()
-- .........(grammars) source + lexer (token stream) -> parser + syntax -> ()
-- parser + syntax -> (HIR Graph) -> 
-- (HIR Graph composed of submodules composed of blocks) -> THIR submodules
- Packages are reusable compilable artifacts
- Realization of the code happens for artifacts 
- Code is optimized and safe
- memory, lifetime, and ownership resolved in MIR

## Problems & Examples

1. Syntax for all types, literals, complex types, blocks, 


1. All numeric (int and float) conversions are seamless and both prevent bugs and allow flexibility

- Lexical analysis 
- Parse into HIR graph of nodes
- 

1.0. CFG: match, while, for, if, else, elif, def, class, 

1.1. Complex types (enum, union, classes, )

1.2. traits and extending existing classes/types

1.3. contracts (with) for wrapping functions and @annotation with ...

2. Generics resolve to their corresponding implementers
- primtiives
- complex
- lists, vector, array, etc.

3. traits implemented by generics

4. utils use those traits on the generics to accomplish useful things like type, assert, max, min, etc. 

5. utils allow for complex operations and generics are able to be created such as linked lists, dictionaries (hash maps) maps, binary tree, red/black tree, and B-Tree. 

6. Algorithms for sorting/search/graph traversal

7. Dynamic types for runtime flexible variables

8. Concurrency

Phase	Milestone	Core Objective	Key Deliverables
Phase 1: Foundations	
1. Lexer & Arithmetic Calculator	
Convert a stream of characters into a stream of tokens, and evaluate basic mathematical expressions.	
Tokenizer, Abstract Syntax Tree (AST), 
REPL (Read-Eval-Print Loop) for math (3 + 5 * 2).
2. Context-Free Grammar (CFG)	Formalize the language syntax and implement a parser to validate structural rules.	BNF/EBNF specification grammar file, LL/LR Parser (or recursive descent parser).


Phase 2:
 Control Flow	

3. Variables & Scope	Introduce memory management via identifiers and block-level scoping.	
 Symbol table, variable assignment, lexical scope resolution (local vs global).

```
module.x = 3
def function():
	local.x = 3
class OtherScope:
	x: int
	# self is not needed, it's implied for all class methods. local is the differentiator. Explicit self is allowed but not needed
	def function():
		x = 1
		return x


4. Control Flow & Booleans	Implement conditional execution and loops.	Boolean operators, if/else statements, while/for loops.
Phase [0-9]:
 Subroutines	
 5. Functions & Execution Stack	Support code reusability, modular encapsulation, and scoping boundaries for execution blocks.	Function definitions, argument parsing, return values, call stack architecture.
Phase [0-9]:
 OOP	6. Custom Types & Classes	Transition to object-oriented paradigms by bundling state (data) and behavior (methods) together.	class keywords, this/self references, object instantiation, heap-allocated properties.
	7. Inheritance & Polymorphism	Establish hierarchical relationships and dynamic runtime behavior between classes.	Subclassing (extends), method overriding, virtual method tables (V-tables).
Phase [0-9]:
 Abstraction	8. Interfaces & Protocols	Decouple design from implementation by defining structural contracts without concrete code.	interface/protocol keyword, compile-time contract validation, explicit/structural subtyping.
Phase [0-9]:
 Scale	9. Modular System	Allow programs to scale across multiple files and namespaces.	import/export directives, modules, packages, compilation-unit management.


Sequential Implementation Steps

To build this systematically, follow these procedural steps:
1. Write the EBNF Grammar: Define your token structures and rules for operator precedence explicitly before writing any parser code.
2. Build the Interpreter Pipeline: Structure your pipeline linearly: Source Code Text → Lexer (Tokens) → Parser (AST) → Evaluator/Runtime.
3. Incorporate Type Checking: Introduce a type checker layer right after parsing. For advanced milestones like interfaces, this layer will verify that concrete classes implement all declared interface methods before passing the AST to execution.




- Grouping phases into passes.

### Programming language basics

- Static versus dynamic behavior.
- Environments and states.
- Static scope and block structure.
- Explicit access control.
- Dynamic scope.
- Parameter-passing mechanisms.
- Aliasing.

## Examples

### Numeric conversions

- Seamless integer and floating-point conversions.
- Prevent bugs while allowing flexibility.

### Control flow and declarations

- `match`.
- `while`.
- `for`.
- `if`, `else`, and `elif`.
- `def`.
- `class`.


### Traits, generics, and contracts

- Extend existing classes and types through traits.
- Resolve generics to their corresponding implementers.
- Implement traits for generic types.
- Use `with` contracts to wrap functions.
- Support `@annotation with ...`.
- Use traits and generics in utilities such as:
  - `type`.
  - `assert`.
  - `max`.
  - `min`.

### Data structures and algorithms

- Linked lists.
- Dictionaries and hash maps.
- Maps.
- Binary trees.
- Red-black trees.
- B-trees.
- Sorting algorithms.
- Search algorithms.
- Graph traversal.

### Variables and scope

```sev
module.x = 3

def function():
    local.x = 3

class OtherScope:
    x: int

    # self is implied for class methods.
    # local distinguishes local variables.
    # Explicit self is allowed but not required.
    def function():
        x = 1
        return x
```

### Concurrency

- Concurrent execution.

## Problem(s)

## Testing
-Language fundamentals
  - Finite automata.
  - Regular expressions to automata.
  - Lexical-analyzer generator design.
  - DFA-based pattern-matcher optimization.
- Syntax analysis:
  - Context-free grammars.
  - Grammar construction.
  - Top-down and bottom-up parsing.
  - SLR, LR, and LALR parsing.
  - Ambiguous grammars.
  - Parser generators.
- Syntax-directed translation:
  - Syntax-directed definitions.
  - Attribute evaluation orders.
  - Applications.
  - Translation schemes.
  - L-attributed definitions.
- Intermediate code generation:
  - Syntax-tree variants.
  - Three-address code.
  - Types and declarations.
  - Expression translation.
  - Type checking.
  - Control flow.
  - Backpatching.
  - Switch statements.
- Runtime environments:
  - Stack allocation.
  - Access to nonlocal data.
  - Heap management.
  - Garbage collection.
  - Trace-based collection.
  - Short-pause collection.
  - Advanced collection techniques.
- Code generation:
  - Target language.
  - Target-code addresses.
  - Basic blocks and flow graphs.
  - Basic-block optimization.
  - Simple code generation.
  - Peephole optimization.
  - Register allocation.
  - Tree-rewriting instruction selection.
  - Optimal expression code generation.
  - Dynamic programming.
- Machine-independent optimization:
  - Sources of optimization.
  - Data-flow analysis and its foundations.
  - Constant propagation.
  - Partial-redundancy elimination.
  - Loops in flow graphs.
  - Region-based analysis.
  - Symbolic analysis.
- Instruction-level parallelism:
  - Scheduling constraints.
  - Basic-block scheduling.
  - Global scheduling.
  - Software pipelining.
- Parallelism and locality:
  - Matrix multiplication.
  - Iteration spaces.
  - Affine array indexes.
  - Data reuse.
  - Array dependence analysis.
  - Synchronization-free parallelism.
  - Synchronization between loops.
  - Pipelining.
  - Locality optimization.
- Interprocedural analysis:
  - Basic concepts.
  - Datalog.
  - Pointer-analysis algorithms.
  - Context-insensitive analysis.
  - Context-sensitive pointer analysis.
  - Binary decision diagrams.

## Appendix

### Reference inventory

- Terms.
- File formats.
- Variables.
- Classes.
- Functions.
- Traits.

### Simple syntax-directed translator

- Syntax definition:
  - Grammars.
  - Derivations.
  - Parse trees.
  - Ambiguity.
  - Operator associativity and precedence.
- Syntax-directed translation:
  - Postfix notation.
  - Synthesized attributes.
  - Simple syntax-directed definitions.
  - Tree traversals.
  - Translation schemes.
- Parsing:
  - Top-down parsing.
  - Predictive parsing.
  - Use of ε-productions.
  - Predictive-parser design.
  - Left recursion.
- Expression translator:
  - Abstract and concrete syntax.
  - Adapting translation schemes.
  - Procedures for nonterminals.
  - Simplifying the translator.
  - Complete translator program.
- Lexical analysis:
  - Removing whitespace and comments.
  - Reading ahead.
  - Constants.
  - Recognizing keywords and identifiers.
  - Lexical-analyzer implementation.
- Symbol tables:
  - One symbol table per scope.
  - Use of symbol tables.
- Intermediate code generation:
  - Intermediate-representation variants.
  - Syntax-tree construction.
  - Static checking.
  - Three-address code.

### Lexical analysis

- Role of the lexical analyzer:
  - Lexical analysis versus parsing.
  - Tokens, patterns, and lexemes.
  - Token attributes.
  - Lexical errors.
- Input buffering:
  - Buffer pairs.
  - Sentinels.
- Token specification:
  - Strings and languages.
  - Operations on languages.
  - Regular expressions.
  - Regular definitions.
  - Extensions of regular expressions.
- Token recognition:
  - Transition diagrams.
  - Reserved words and identifiers.
  - Completing the running example.
  - Architecture of a transition-diagram-based analyzer.
- Lex:
  - Usage.
  - Program structure.
  - Conflict resolution.
  - Lookahead operator.
- Finite automata:
  - Nondeterministic finite automata.
  - Transition tables.
  - Acceptance of input strings.
  - Deterministic finite automata.
- Regular expressions to automata:
  - NFA-to-DFA conversion.
  - NFA simulation.
  - Efficiency of NFA simulation.
  - NFA construction from regular expressions.
  - Efficiency of string-processing algorithms.
- Lexical-analyzer generator design:
  - Generated-analyzer structure.
  - NFA-based pattern matching.
  - DFAs for lexical analyzers.
  - Lookahead implementation.
- DFA-based pattern-matcher optimization:
  - Important NFA states.
  - Functions computed from syntax trees.
  - `nullable`, `firstpos`, and `lastpos`.
  - `followpos`.
  - Direct regular-expression-to-DFA conversion.
  - DFA state minimization.
  - State minimization in lexical analyzers.
  - Time-space tradeoffs in DFA simulation.

### Syntax analysis

- Parser responsibilities:
  - Representative grammars.
  - Syntax-error handling.
  - Error-recovery strategies.
- Context-free grammars:
  - Formal definition.
  - Notational conventions.
  - Derivations.
  - Parse trees and derivations.
  - Ambiguity.
  - Verifying the generated language.
  - Context-free grammars versus regular expressions.
- Grammar construction:
  - Lexical versus syntactic analysis.
  - Eliminating ambiguity.
  - Eliminating left recursion.
  - Left factoring.
  - Non-context-free language constructs.
- Top-down parsing:
  - Recursive descent.
  - `FIRST` and `FOLLOW`.
  - LL(1) grammars.
  - Nonrecursive predictive parsing.
  - Predictive-parser error recovery.
- Bottom-up parsing:
  - Reductions.
  - Handle pruning.
  - Shift-reduce parsing.
  - Shift-reduce conflicts.
- Simple LR parsing:
  - Reasons for using LR parsers.
  - Items and the LR(0) automaton.
  - LR-parsing algorithm.
  - SLR parsing-table construction.
  - Viable prefixes.
- More powerful LR parsers:
  - Canonical LR(1) items.
  - LR(1) item-set construction.
  - Canonical LR(1) parsing tables.
  - LALR parsing-table construction.
  - Efficient LALR table construction.
  - LR table compaction.
- Ambiguous grammars:
  - Resolving conflicts through precedence and associativity.
  - Dangling-else ambiguity.
  - LR error recovery.
- Parser generators:
  - Yacc.
  - Ambiguous grammars in Yacc.
  - Creating Yacc lexical analyzers with Lex.
  - Yacc error recovery.

### Syntax-directed translation

- Syntax-directed definitions:
  - Inherited and synthesized attributes.
  - Evaluating definitions at parse-tree nodes.
- Evaluation orders:
  - Dependency graphs.
  - Attribute-evaluation ordering.
  - S-attributed definitions.
  - L-attributed definitions.
  - Semantic rules with controlled side effects.
- Applications:
  - Syntax-tree construction.
  - Type structure.
- Translation schemes:
  - Postfix translation schemes.
  - Parser-stack implementation.
  - Actions inside productions.
  - Eliminating left recursion.
  - Schemes for L-attributed definitions.
- Implementing L-attributed definitions:
  - Translation during recursive-descent parsing.
  - On-the-fly code generation.
  - LL parsing.
  - Bottom-up parsing.

### Intermediate code generation

- Syntax-tree variants:
  - Directed acyclic graphs for expressions.
  - Value-number construction of DAGs.
- Three-address code:
  - Addresses and instructions.
  - Quadruples.
  - Triples.
  - Static single-assignment form.
- Types and declarations:
  - Type expressions.
  - Type equivalence.
  - Declarations.
  - Storage layout for local names.
  - Declaration sequences.
  - Record and class fields.
- Expression translation:
  - Operations within expressions.
  - Incremental translation.
  - Addressing array elements.
  - Translating array references.
- Type checking:
  - Type-checking rules.
  - Type conversions.
  - Function and operator overloading.
  - Type inference and polymorphic functions.
  - Unification.
- Control flow:
  - Boolean expressions.
  - Short-circuit code.
  - Flow-of-control statements.
  - Control-flow translation of Boolean expressions.
  - Avoiding redundant jumps.
  - Boolean values and jumping code.
- Backpatching:
  - One-pass code generation.
  - Boolean expressions.
  - Flow-of-control statements.
  - `break`, `continue`, and `goto`.
- Switch statements:
  - Translation.
  - Syntax-directed translation.
- Intermediate code for procedures.

### Runtime environments

- Storage organization:
  - Static versus dynamic allocation.
- Stack allocation:
  - Activation trees.
  - Activation records.
  - Calling sequences.
  - Variable-length stack data.
- Nonlocal stack data:
  - Access without nested procedures.
  - Issues with nested procedures.
  - Languages with nested procedure declarations.
  - Nesting depth.
  - Access links.
  - Manipulating access links.
  - Access links for procedure parameters.
  - Displays.
- Heap management:
  - Memory manager.
  - Memory hierarchy.
  - Program locality.
  - Fragmentation reduction.
  - Manual deallocation requests.
- Garbage-collection foundations:
  - Design goals.
  - Reachability.
  - Reference counting.
- Trace-based collection:
  - Basic mark-and-sweep.
  - Basic abstraction.
  - Mark-and-sweep optimization.
  - Mark-and-compact.
  - Copying collectors.
  - Cost comparisons.
- Short-pause collection:
  - Incremental collection.
  - Incremental reachability analysis.
  - Partial collection.
  - Generational collection.
  - Train algorithm.
- Advanced collection:
  - Parallel and concurrent collection.
  - Partial object relocation.
  - Conservative collection for unsafe languages.
  - Weak references.

### Code generation

- Generator design:
  - Input.
  - Target program.
  - Instruction selection.
  - Register allocation.
  - Evaluation order.
- Target language:
  - Simple target-machine model.
  - Program and instruction costs.
- Target-code addresses:
  - Static allocation.
  - Stack allocation.
  - Runtime addresses for names.
- Basic blocks and flow graphs:
  - Basic blocks.
  - Next-use information.
  - Flow graphs.
  - Flow-graph representation.
  - Loops.
- Basic-block optimization:
  - DAG representation.
  - Local common subexpressions.
  - Dead-code elimination.
  - Algebraic identities.
  - Array-reference representation.
  - Pointer assignments and procedure calls.
  - Reassembling basic blocks from DAGs.
- Simple code generator:
  - Register and address descriptors.
  - Code-generation algorithm.
  - Design of `getReg`.
- Peephole optimization:
  - Redundant load and store elimination.
  - Unreachable-code elimination.
  - Control-flow optimization.
  - Algebraic simplification and strength reduction.
  - Machine idioms.
- Register allocation and assignment:
  - Global allocation.
  - Usage counts.
  - Assignment for outer loops.
  - Graph coloring.
- Instruction selection by tree rewriting:
  - Tree-translation schemes.
  - Input-tree tiling.
  - Pattern matching through parsing.
  - Semantic-checking routines.
  - General tree matching.
- Optimal expression code generation:
  - Ershov numbers.
  - Labeled expression trees.
  - Evaluation with insufficient registers.
- Dynamic-programming code generation:
  - Contiguous evaluation.
  - Dynamic-programming algorithm.

### Machine-independent optimization

- Sources of optimization:
  - Causes of redundancy.
  - Quicksort example.
  - Semantics-preserving transformations.
  - Global common subexpressions.
  - Copy propagation.
  - Dead-code elimination.
  - Code motion.
  - Induction variables and strength reduction.
- Data-flow analysis:
  - Data-flow abstraction.
  - Analysis schema.
  - Schemas on basic blocks.
  - Reaching definitions.
  - Live-variable analysis.
  - Available expressions.
- Data-flow foundations:
  - Semilattices.
  - Transfer functions.
  - Iterative algorithm for general frameworks.
  - Meaning of a data-flow solution.
- Constant propagation:
  - Data-flow values.
  - Meet operation.
  - Transfer functions.
  - Monotonicity.
  - Nondistributivity.
  - Interpretation of results.
- Partial-redundancy elimination:
  - Sources of redundancy.
  - Limits of redundancy elimination.
  - Lazy-code-motion problem.
  - Expression anticipation.
  - Lazy-code-motion algorithm.
- Loops in flow graphs:
  - Dominators.
  - Depth-first ordering.
  - Edges in depth-first spanning trees.
  - Back edges and reducibility.
  - Flow-graph depth.
  - Natural loops.
  - Convergence speed of iterative data-flow algorithms.
- Region-based analysis:
  - Regions.
  - Region hierarchies for reducible flow graphs.
  - Analysis overview.
  - Transfer-function assumptions.
  - Analysis algorithm.
  - Nonreducible flow graphs.
- Symbolic analysis:
  - Affine expressions of reference variables.
  - Data-flow problem formulation.
  - Region-based symbolic analysis.

### Instruction-level parallelism

- Processor architectures:
  - Instruction pipelines and branch delays.
  - Pipelined execution.
  - Multiple instruction issue.
- Scheduling constraints:
  - Data dependence.
  - Dependences among memory accesses.
  - Register-usage versus parallelism tradeoffs.
  - Ordering register allocation and scheduling.
  - Control dependence.
  - Speculative-execution support.
  - Basic machine model.
- Basic-block scheduling:
  - Data-dependence graphs.
  - List scheduling.
  - Prioritized topological orders.
- Global scheduling:
  - Primitive code motion.
  - Upward and downward code motion.
  - Updating data dependences.
  - Global scheduling algorithms.
  - Advanced code-motion techniques.
  - Interaction with dynamic schedulers.
- Software pipelining:
  - Introduction.
  - Loop pipelining.
  - Register allocation and code generation.
  - Do-across loops.
  - Goals and constraints.
  - Software-pipelining algorithm.
  - Scheduling acyclic and cyclic dependence graphs.
  - Pipelining-algorithm improvements.
  - Modular variable expansion.
  - Conditional statements.
  - Hardware support.

### Parallelism and locality

- Basic concepts:
  - Multiprocessors.
  - Application parallelism.
  - Loop-level parallelism.
  - Data locality.
  - Affine-transform theory.
- Matrix-multiplication example:
  - Algorithm.
  - Optimizations.
  - Cache interference.
- Iteration spaces:
  - Construction from loop nests.
  - Loop-nest execution order.
  - Matrix formulation of inequalities.
  - Symbolic constants.
  - Execution-order control.
  - Changing axes.
- Affine array indexes:
  - Affine accesses.
  - Affine and nonaffine accesses in practice.
- Data reuse:
  - Types of reuse.
  - Self reuse.
  - Self-spatial reuse.
  - Group reuse.
- Array data-dependence analysis:
  - Dependence of array accesses.
  - Integer linear programming.
  - GCD test.
  - Integer-programming heuristics.
  - Solving general integer linear programs.
- Synchronization-free parallelism:
  - Introductory example.
  - Affine space partitions.
  - Space-partition constraints.
  - Solving space-partition constraints.
  - Simple code-generation algorithm.
  - Eliminating empty iterations.
  - Eliminating tests from innermost loops.
  - Source-code transformations.
- Synchronization between parallel loops:
  - Constant number of synchronizations.
  - Program-dependence graphs.
  - Hierarchical time.
  - Parallelization algorithm.
- Pipelining:
  - Definition.
  - Successive over-relaxation example.
  - Fully permutable loops.
  - Pipelining fully permutable loops.
  - General theory.
  - Time-partition constraints.
  - Solving constraints with Farkas’ lemma.
  - Code transformations.
  - Parallelism with minimum synchronization.
- Locality optimization:
  - Temporal locality of computed data.
  - Array contraction.
  - Partition interleaving.
  - Combining the optimizations.
- Other affine-transform applications:
  - Distributed-memory machines.
  - Multiple-instruction-issue processors.
  - Vector and SIMD instructions.
  - Prefetching.

### Interprocedural analysis

- Basic concepts:
  - Call graphs.
  - Context sensitivity.
  - Call strings.
  - Cloning-based context-sensitive analysis.
  - Summary-based context-sensitive analysis.
- Applications:
  - Virtual method invocation.
  - Pointer alias analysis.
  - Parallelization.
  - Software-error and vulnerability detection.
  - SQL injection.
  - Buffer overflow.
- Logical representation of data flow:
  - Introduction to Datalog.
  - Datalog rules.
  - Intensional and extensional predicates.
  - Datalog execution.
  - Incremental evaluation.
  - Problematic rules.
- Simple pointer-analysis algorithm:
  - Difficulties in pointer analysis.
  - Pointer and reference model.
  - Flow insensitivity.
  - Datalog formulation.
  - Use of type information.
- Context-insensitive analysis:
  - Effects of method invocation.
  - Call-graph discovery in Datalog.
  - Dynamic loading and reflection.
- Context-sensitive pointer analysis:
  - Contexts and call strings.
  - Adding context to Datalog rules.
  - Additional sensitivity considerations.
- Datalog implementation using binary decision diagrams:
  - Binary decision diagrams.
  - BDD transformations.
  - Representing relations.
  - Relational operations as BDD operations.
  - Points-to analysis.

### Complete front-end example

- Source language.
- Main entry point.
- Lexical analyzer.
- Symbol tables and types.
- Intermediate code for expressions.
- Jumping code for Boolean expressions.
- Intermediate code for statements.
- Parser.
- Creating the front end.

## Misc.

### Implementation milestones

1. **Foundations: Lexer and arithmetic calculator**
   - Objective: Convert characters to tokens and evaluate basic mathematical expressions.
   - Deliverables: Tokenizer, AST, and arithmetic REPL.
   - Example: `3 + 5 * 2 → 13`.

2. **Foundations: Context-free grammar**
   - Objective: Formalize syntax and validate structural rules.
   - Deliverables: BNF/EBNF grammar and LL, LR, or recursive-descent parser.

3. **Control flow: Variables and scope**
   - Objective: Introduce identifiers and block-level scoping.
   - Deliverables: Symbol table, assignment, and local/global lexical-scope resolution.

4. **Control flow: Booleans and loops**
   - Objective: Support conditional execution and repetition.
   - Deliverables: Boolean operators, `if`/`else`, and `while`/`for`.

5. **Subroutines: Functions and execution stack**
   - Objective: Support reuse, modular encapsulation, and execution scopes.
   - Deliverables: Function definitions, arguments, return values, and call-stack architecture.

6. **OOP: Custom types and classes**
   - Objective: Bundle state and behavior.
   - Deliverables: Classes, implicit or explicit `self`, object construction, and heap-allocated properties.

7. **OOP: Inheritance and polymorphism**
   - Objective: Establish class relationships and dynamic behavior.
   - Deliverables: Subclassing, method overriding, and virtual method tables.

8. **Abstraction: Interfaces and protocols**
   - Objective: Separate contracts from implementations.
   - Deliverables: Interface/protocol declarations, compile-time contract validation, and explicit or structural subtyping.

9. **Scale: Module system**
   - Objective: Support multiple files and namespaces.
   - Deliverables: Imports/exports, modules, packages, and compilation-unit management.

