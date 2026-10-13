SIP-0000: Title

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

## Context

## Goal(s)

- Source text goes to MLIR
- Packages are reusable compilable artifacts
- Realization of the code happens for artifacts 
- Code is optimized and safe
- 

## Examples

1. All numeric (int and float) conversions are seamless and both prevent bugs and allow flexibility



Phase	Milestone	Core Objective	Key Deliverables
Phase 1: Foundations	
1. Lexer & Arithmetic Calculator	
Convert a stream of characters into a stream of tokens, and evaluate basic mathematical expressions.	
Tokenizer, Abstract Syntax Tree (AST), 
REPL (Read-Eval-Print Loop) for math (3 + 5 * 2).
2. Context-Free Grammar (CFG)	Formalize the language syntax and implement a parser to validate structural rules.	BNF/EBNF specification grammar file, LL/LR Parser (or recursive descent parser).
Phase [0-9]:
 Control Flow	
 3. Variables & Scope	Introduce memory management via identifiers and block-level scoping.	
 Symbol table, variable assignment, lexical scope resolution (local vs global).
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




## Problem(s)

## Testing


