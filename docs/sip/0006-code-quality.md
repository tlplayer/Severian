SIP-0000: Title

Status: *Draft* | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Summary

Severian lacks code coverage, linting, build pipelining, enforcing a default standard in the package library. We should also 
have go to definition, usages, and debugging functionality to inspect/review and watch variables while coding to catch bugs



## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

```sev
package.lint
package.coverage
```

## Context (list of items)

- Identify code
- Remove/find candidates quickly and allow A/B testing quickly
- 

## Problem(s)

Linting

- Long parameter lists
- God classes
- Large class
- Long method
- Large file (>1000 LOC)


- Switch abuse instead of polymorphism indexing dicts/maps
- Alternative Classes with Different Interfaces
- Divergent Change (Code churn means this class has change in the last 7/10 commits CODE SMELL)


### The Dispensables: Unnecessary clutter that impacts readability.

- Lazy Class / Freeloader: A class that does too little to justify its maintenance cost.
- Dead Code: Variables, functions, or files that are never called or executed.
- Speculative Generality: Writing complex scaffolding "just in case" it's needed in the future
- Standalone file that isn't exposed in the package --deprecate should propose/show code that has no output/is removed during compilation


### The Couplers: Excessive or inappropriate connections between classes.

- Middle Man: A class whose entire existence is simply delegating work to another class


### Test-Specific Smells: Design flaws present inside unit/integration tests.

- Assertion Roulette: A test containing multiple assertions without descriptive messages, making failure tracking difficult.
- Eager Test: A single test method checking multiple functionalities of production code at once.
- Mystery Guest: A test that depends on external data files or configurations to pass, hiding its true logic.

### Code Coverage & Testing Rigor 


- (Dynamic Level)Dynamic verification ensures that as the system runs, its codebase behaves correctly under explicit metrics.Coverage MetricDescriptionStatement / Line CoverageMeasures the percentage of individual source code lines executed during testing.- Branch / Decision CoverageEnsures every path at a control structure (e.g., both the true and false branches of an if statement) is tested.Function / Method CoverageTracks whether every defined function or method in the codebase has been called at least once.Condition CoverageEvaluates whether every boolean sub-expression in a complex conditional evaluates to both true and false.Path CoverageThe most rigorous standard; tests every conceivable linear path through a function from start to finish.Mutation TestingIntroduces intentional bugs ("mutants") into your code to see if your test suite successfully catches and fails them.


## Examples


## Illustration


## Testing


## Performance

## Optimization

## Rollback Procedure

## Migration
1. All package.toml files become json files for readability and easier handling a json version that allows comments
2. All package.pkg should use the golden path initial options with comments of possible values automatically update by package.options to include all potential values to avoid doc/user desync. 
3. All builds should have limits for LOC <1000, code/branch coverage > 90% and no dead code/unused variables as warnings as well as a O(...) notation for functions etc. which allows for cleaner profiling testing and better optimizations
4. Methods should be smaller than 400 LOCs. 
5. Matches should have less than 10 cases otherwise they should be a dict/map lookup
6. Expensive/long running code with many allocations should be flagged
7. Leaks should be flagged and cause compiler errors per an option
8. Old code might be killable, looking at LOC written early will give hints to TODO/leftover work
9. New code might be in progress and also killable worth looking into
10. Duplicate functionality should be flagged and collapsed/refactored
11. Comment blocks are helpful for understanding the code. We should have comment blocks like that with notes etc. 
12. Remove pointless comments
13. stacked if statements for handling are the same as match
14. Finish implementing the examples/testing for all those cases and add testing for those throughout the compiler

## Deprecation
N/A should not result in deprecations

## Milestones (max 5)

1. sev build runs lint pipeline, according to package.toml 


