# regex

Regular-expression values and matching. The package owns its POSIX provider and
reaches it through the typed `abi`/`ffi` boundary; compiler lowering contains no
regex parser or execution engine. Literal-pattern compile-time diagnostics
remain future work; invalid dynamic patterns preserve the existing safe
fallbacks (`false`, an empty match list, unchanged split input, or unchanged
substitution input).
