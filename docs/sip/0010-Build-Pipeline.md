SIP-0010: Build Pipeline

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
Current compilers follow a narrowing funnel for simplicity. This is good for processing speed but is bad for ensuring good software gets through.
Severian will needs to define it's syntax within itself. Usually this is hidden in IR in other languages and frankly results in a desync between
interface/implementation details of the language. For example, str/string/byte-strings are all represented in hidden c whereas if they were written in the same structure that classes etc. were written in it would enable easier fixing/modifications and shows a better compiler pipeline.

## Structure

package is what's used to assembly/create the output artifacts

 These package.pkg/debug/build/<compiler pipeline step>/<error,lint,warning,etc.> will tell us what's wrong with our builds at which step to better fix bugs and get a hollistic picture. 

### Phases of the Compiler

### Syntax

Defining primitives, generic types, modules, and general compiler structures and processes 

example:
```sev
if x == y:
    return 0
```

a[0] = a[1] + b

sentence array_assign[var,"[",int,"]",=, ]


Here we have a block which looks at two variables, unknown types, which must 
implemnet operator/sentence sentence equal[self,"==": Y,comparator: string|int|float|char] 

The block is an if block (these can be dynamically defined and are all automatically registered without tweaking the whole compiler stack)
inside we have an operation that interacts with the CFG and returns to the caller block return 0 which looks at the current block and returns to the caller block's 
previous cursor with the Result[R] type with value 0 which could've been an error

The lower levels should not have things like IfBlock or BlockIf in enums and then do custom handling for a big reason. It limits interop and hides the handling behind the compiler when the opposite should occur. The compiler should read the rules of the language to self express. 
that information needs to be passed over from HIR to MIR after the HIR graph goes to MIR after passing CFG, Strongly Connected Component (SCC), and Memory operations to make the MIR graph. 


#### Primtiives

1. int

The integer is a basic building block of any programming language
```sev
# Assign a to a literal int
a = 1
print(1) 



```




### Frontend
Syntax: define token parsing, block, sentence, operations, types, and values 
Source: Loads the source text and other imports as needed  
lexer: Converts source text into tokens and lexemes from syntax's definition
Parsing: Groups blocks, sub blocks, sentence, operations, types, and values per syntax resulting in block IR

### HIR

The HIR graph contains the refined output from applying the rules/information from syntax onto source outputting a graph of values/types/operation/sentences/blocks/modules etc. into a graph of objects we can begin to understand and apply rules onto to catch bugs.


### MIR

This is where HIR has applied the checks that make sense at that level we can now work on ownership/CFG 
of the submodules HIR has given us. submodules map to ~.o/.so etc compilable units essentially this lets us know what objects contribute to the package's output. We can then apply our mir steps and output the issues or link up the submodules into propoer modules to expose to LIR which allows the LIR a good time finally compiling the output


### LIR

This step lowers the finalized modules into MLIR compilable objects and generates executables/.o/.so files with the .pkgi to interface with other objects. 



## Problem(s)

## Examples

## Testing

1. Does the frontend pull literals, variables, block structure from syntax?
2. Does that structure remain intact until MLIR lowering?
3. Does submodule grouping happen in HIR?
4. Does module grouping happen in MIR and linking correctly reuse components?
5. Does ownership/CFG have clean structure between blocks/submodules?
6. Does LIR/MLIR have enough information to make valid MLIR code to optimize and creature artifacts for other packages/release artifacts/executables?


