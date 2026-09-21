# Sentence dispatch

`Grammar` contains explicit literal symbols and named captures. `TermRole`
distinguishes patterns, expressions, types, values, containers, callables, blocks
and nested sentences. Nested term children reference the parser's node arena;
source offsets follow the string contract's Unicode scalar indexing.

`match_sentence` filters all candidates by structural role and literal spelling
before invoking semantic constraint predicates. Exactly one matching declaration
is required. Neither declaration order nor predicate count breaks ambiguity.
`iteration_sentences` supplies basic and `with` forms sharing `Iteration` as their
semantic interface. The binding is a pattern; `Iterable` constrains the source.

The parser and universal packages expose this package through declared dependency
aliases. This is the composition/dispatch library; parsing the new `sentence [...]`
declaration syntax and lowering its semantic objects remain integration work.
