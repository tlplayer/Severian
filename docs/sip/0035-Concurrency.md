
Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Tags: types | containers | blocks | ... (topics relevant to the document)
## Appendix

 all terms in a list
 - term 1: description
 -- subterms
 - term 2: description

| Executor | Meaning | Memory |
|--------|--------|--------|
| `virtual` | Runtime-scheduled lightweight task | Shared |
| `thread` | OS thread (pthread on POSIX) | Shared |
| `process` | OS process | Isolated address space |
| `runtime` | Executor-managed task | Shared by default |

- absent: this is unsafe concurrency, you know that it will finish and will handle the await/it will finish unawaited
- local: function level ownership task completes in the block
- self: class scoped ownership when the class dies this will have finished
- module: this task will complete before the module is finished

## Goal(s)

List of goals to 

1. Concurrency is fast
2. Concurrency is easy
3. Concurrnecy is safe

## High Level Design

1. Scope ownership is enforced in MIR, including cleanup and synchronization requirements.
2. Scopes can represent arbitrary lifecycle owners, with local, self, and module being predefined.
3. Cancellation and task failure are part of the scope contract, rather than conventions implemented in libraries.


## Examples & Problems

Blocks of severian code to show the desired behavior

Examples of corner cases

### Data Model

```sev
Task {
    owner: ScopeId
    execution: Virtual | Thread | Process
    captures: [OwnershipContract]
    cancellation: CancellationPolicy
    failure: FailurePolicy
}
```

### Basic Concurrency

```sev
def work(value: int) -> int:
    return value * 2

test "tasks are awaitable":
    task = async work(21) with local
    assert(await task == 42)

def main():
    # Tasks owned by self cannot outlive main.
    module.first = async work(10) with local
    module.second = async work(21) with local

test "async work is done at exit when with is declared":
    main()
    assert(module.first == 42)
    assert.module.

```

### Scopes and Type of concurrency

```sev

# Lightweight task, scoped to function
a = async foo() with local, virtual

# OS thread, scoped to function
b = async foo() with local, thread

# OS process, scoped to function
c = async foo() with local, process

# Object-owned OS thread
d = async foo() with self, thread

# Module-owned OS process
e = async foo() with module, process
```

### 

### Not every task belongs to a lexical, object, or module scope

Consider an HTTP server.

A request arrives, starts three subtasks, and then completes.

Those subtasks should belong to the request, not necessarily the server object or module.

Go addresses this problem using request contexts that propagate cancellation across functions and services. Go

Severian could support arbitrary owners:
def handle(request: Request):
    a = async authenticate() with request
    b = async load_data() with request

This raises an important possibility:

What if local, self, and module are predefined owners rather than the only permitted scopes?

That generalizes your design without replacing the syntax.


## Tests

Blocks of severian code to test behavior and problems to resolve/reject certain behaviors
