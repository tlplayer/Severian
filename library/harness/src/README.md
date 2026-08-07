┌────────────────────────────────────────────────────────────┐
│                    Agent / User Goal                       │
│        "Implement distributed inference scheduler"         │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                     Goal Planner                           │
│                                                            │
│ GoalNode                                                   │
│ ├─ id                                                      │
│ ├─ objective                                               │
│ ├─ status                                                  │
│ ├─ parent                                                  │
│ └─ dependencies[]                                          │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                     Task Graph                             │
│                                                            │
│                         ROOT                               │
│                        /    \                              │
│                     Task A  Task B                         │
│                    /    \                                  │
│                  A1      A2                                │
│                                                            │
│ DFS executor maintains current reasoning path              │
└──────────────────────────┬─────────────────────────────────┘
                           │
             current_node + current_path
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                  Context Graph Store                       │
│                                                            │
│ ContextNode                                                │
│ ├─ content                                                 │
│ ├─ embedding                                               │
│ ├─ type                                                    │
│ ├─ priority                                                │
│ └─ token_cost                                              │
│                                                            │
│ ContextEdge                                                │
│ ├─ REQUIRES                                                │
│ ├─ CONSTRAINS                                              │
│ ├─ EVIDENCE_FOR                                            │
│ ├─ PRODUCED_BY                                             │
│ └─ RELATED_TO                                              │
└───────────────┬──────────────────────┬─────────────────────┘
                │                      │
       structural traversal      semantic search
                │                      │
                ▼                      ▼
        ┌──────────────┐       ┌───────────────┐
        │ Graph Search │       │ Vector Search │
        │ DFS/BFS/etc. │       │ ANN similarity│
        └───────┬──────┘       └───────┬───────┘
                └──────────┬────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│                   Context Compiler                         │
│                                                            │
│  Candidate spans                                           │
│       ↓                                                    │
│  relevance scoring                                        │
│       ↓                                                    │
│  dependency closure                                        │
│       ↓                                                    │
│  token-budget allocation                                   │
│       ↓                                                    │
│  eviction / compression                                    │
│       ↓                                                    │
│  final ordered context                                     │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                         LLM                                │
│                                                            │
│ [global goal]                                              │
│ [current DFS path]                                         │
│ [constraints]                                              │
│ [retrieved context]                                        │
│ [dependency results]                                       │
│ [current task]                                             │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                    Structured Result                       │
│                                                            │
│ result                                                     │
│ discovered_dependencies[]                                  │
│ new_context[]                                              │
│ confidence                                                 │
│ status                                                     │
└──────────────────────────┬─────────────────────────────────┘
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
          update task graph   update context graph
                  │                 │
                  └────────┬────────┘
                           ▼
                    DFS backtrack
                           │
                           ▼
                      Next task