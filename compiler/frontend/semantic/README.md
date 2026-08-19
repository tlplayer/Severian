Refactor the semantic layer 
Adding a new declared type must never require changing semantic source code.
semantic/
├── context.rs
├── registry/
│   ├── mod.rs
│   ├── types.rs
│   ├── traits.rs
│   └── functions.rs
├── resolve/
│   ├── names.rs
│   ├── types.rs
│   ├── calls.rs
│   └── operators.rs
├── analyze/
│   ├── expressions.rs
│   ├── statements.rs
│   └── declarations.rs
└── pipeline.rs