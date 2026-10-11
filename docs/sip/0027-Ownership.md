# SIP-0027: Ownership — Storage and Effects over MIR

Status: Draft
Type: Compiler
Authors: Timothy Player 
Created: 2026-10-03
Target: `sev_compiler/mir/ownership`
Supersedes: None
Superseded by: None

## Appendix

| Term / name | Kind | Meaning |
|---|---|---|
| Ownership | Analysis | Whether storage may be read, mutated, transferred, borrowed or released at an execution point. |
| `Realization`, `CfgBody` | MIR contracts | Executable specialization and its checked flow graph. |
| `ValueId`, `Place`, `Projection` | Identity / storage contracts | Realization-local value, storage location and field/index/dereference path. |
| `EffectContract` | Proposed provider contract | Reads, writes, transfers, loans, escapes, allocation, destruction and exceptional effects. |
| `FlowState` | Proposed class | Definite initialization, possible moves and active loans at a program point. |
| `Loan` | Proposed class | Borrowed place, access permission, validity region and origin. |
| `OwnershipPlan` | Proposed analysis result | Flow facts, callable effects and required cleanup tied to input revisions. |
| `CleanupAction` | Proposed class | Resolved destructor/release action at an identified exit with ordering and origin. |
| `AnalysisInputs` | Proposed class | Body revision plus type/provider/callee effect revisions. |
| `analyze`, `apply_cleanup`, `verify` | Proposed functions | Compute facts, materialize cleanup and check final ownership readiness. |
| `realization`, `body`, `effects`, `plan` | API parameters | Owner, checked execution, resolved effect contracts and analysis result. |
| `partition_owner` | Existing block field | Compilation grouping only; does not describe memory ownership. |
| `Diagnostic`, `OperationOrigin` | Result / provenance | Conflicting use and origin of ownership obligations. |
| `.sev` | File format | Source implementation; no ownership archive format is introduced. |


## Ownership Overview

Move: Transfers exclusive ownership of a field.
Borrow: Temporarily grants access to a field without transferring ownership.
View: Grants read-only access to a field.
Share: Creates another owning reference to the same field.
Copy: Creates an independent value.
_: infer ownership
`a: _ infer = 0 #move int useful for inferring ownership and type to demark the method of ownership resolution `


Ownership is a guard against mutations causing bugs. If ownership of an item is clear to the compiler, bugs can be caught statically and dynamic bugs can be handled safely. 

A big problem with this syntax layer is:
- Ceremony around what is actually happening who owns this? How will its lifetime perspire? In this scenario how do I handle this? What is the default behavior of ownership for X?

These problems partially arise because configuration of ownership was seen as a black and white issue. Whereas if 
there's flexibility for these scenarios we can avoid bugs without having a lot of ceremony.

Scalar Types
```sev
a: int = 0
b: float = 0.0

def add(x,y):
    return x+y

add(a,b)
```

In the above snippet, who owns a and b in the add function? In some languages it would copy, pass by reference, an immutable view of the values, etc. 

This is simple for scalar types, we can easily copy the values but passing by reference is not always desired since it's slower if we do not need to mutate x,y which usually don't happen through function calls vs direct assignment within the scope. 

However, most programs are not simple addition. There can be growing data structures, database connections which maintain 
connections through the singleton structure etc. 

```py
a: list[int] = [0]*10000

def sum(l: list[int]) -> int:
    result = 0
    for n in l:
        result +=1
    return result

# Do we copy all a? 
sum(a)
```

The question and problem arise, how do we know when to pass a copy, a view, a move, borrow, share, or mirror? 
There are some hints we can get through compiling and our constraint based design. 

Defaulting to view becomes arduous for scalars

```

def modify(a,b)
```
```sev
# Proposed Severian ownership inference.
#
# Goals:
# 1. Infer ownership from type properties and callable effects.
# 2. Avoid unnecessary copies, allocations, and reference counting.
# 3. Preserve explicit move, borrow, view, copy, share, and mirror.
# 4. Reject invalid mutations, escapes, and ownership transfers.
# 5. Allow the compiler to optimize physical argument passing.
#
# Important distinction:
#   Semantic ownership != physical calling convention.
#
# A value can semantically be copied but physically passed in a register.
# A value can semantically be viewed but physically passed by value.
# Optimization cannot change observable ownership behavior.
```

### 1. Scalar types: default to copy

```sev
a: int = 10
b: float = 20.0

def add(x: int, y: float) -> float:
    return x + y

result := add(a, b)

# Inferred:
#   x: copy int
#   y: copy float
#
# Ownership:
#   Caller owns a and b.
#   Callee owns independent scalar values x and y.
#
# Lowering:
#   x -> register
#   y -> register
#
# No heap allocation.
# No reference counting.
# No memory copy required for register-sized primitives.
```

### 2. Collections: default to view when only reading

```sev
a: list[int] = [0] * 10000

def sum(values: list[int]) -> int:
    result := 0

    for n in values:
        result += n

    return result

total := sum(a)

# Inferred:
#   values: view list[int]
#
# Compiler observes:
#   reads(values)
#   no writes(values)
#   no transfers(values)
#   no escapes(values)
#
# Lowering:
#   pass pointer + length
#
# No list copy.
# No ownership transfer.
# Caller retains ownership.
```

### 3. Mutation: infer borrow

```sev
values := [1, 2, 3]

def append_one(values: list[int]):
    values.append(1)

append_one(values)

assert values == [1, 2, 3, 1]

# Inferred:
#   values: borrow list[int]
#
# Effects:
#   writes(values)
#
# The caller retains ownership.
# The callee temporarily receives mutation permission.
# Permission returns when the call completes.
#
# Reject overlapping incompatible access while borrow is active.
```

### 4. Consuming a value: infer move when ownership transfer is required

```sev
class Database:
    connection: pointer[Connection]

def close(db: Database):
    db.connection.close()
    # Explicitly consuming db would require its effect contract
    # to declare a transfer or destruction.

database := Database()

close(move database)

# Explicit:
#   move database
#
# The caller loses ownership.
# The callee becomes responsible for final cleanup.
#
# Reject:
#   database.query("SELECT 1")
#   # database has already been moved.
```

### 5. Infer from effects, not size alone

```sev
class Tensor:
    data: pointer[float]
    shape: list[int]

def inspect(t: Tensor):
    print(t.shape)

def update(t: Tensor):
    t.data[0] = 10.0

def consume(move t: Tensor):
    process(move t)

tensor := Tensor(...)

inspect(tensor)
# Infer view:
#   effects = {read(t.shape)}

update(tensor)
# Infer borrow:
#   effects = {write(t.data[0])}

consume(move tensor)
# Explicit move:
#   effects = {transfer(t)}
#
# Size affects the calling convention.
# Effects determine semantic access permissions.
```

### 6. Explicit ownership overrides inference

```sev
def foo(
    a: view Tensor,
    b: borrow Tensor,
    c: copy Tensor,
    d: move Tensor,
    e: share Tensor,
    f: mirror Tensor,
):
    ...

# view   -> read-only access; external mutation depends on aliasing rules
# borrow -> temporary access with permitted mutation
# copy   -> independent value
# move   -> exclusive ownership transfer
# share  -> another owning reference
# mirror -> flat copy-on-write relationship
#
# Explicit contracts are checked rather than silently changed.
```

### 7. Field-level inference

```sev
class User:
    name: string
    age: int
    history: list[string]

def update_age(user: User):
    user.age += 1

u := User("Alice", 30, [])

update_age(u)

# Inferred:
#   borrow u.age
#
# Effects:
#   writes(Place(u, .age))
#
# The compiler does not need mutation permission for:
#   u.name
#   u.history
#
# This permits disjoint-field borrowing.
# Whole-object aliases must still be checked for overlap.
```

### 8. Partial ownership transfer

```sev
class User:
    name: string
    history: list[string]

def extract_history(user: User) -> list[string]:
    return move user.history

u := User("Alice", ["login", "logout"])

history := extract_history(u)

print(u.name)       # Accept.
print(u.history)    # Reject: field moved.

# Inferred effects:
#   transfer(Place(user, .history))
#
# Caller:
#   u.name remains owned.
#   u.history becomes uninitialized.
#   history owns the transferred list.
#
# Cleanup:
#   Drop u.name when required.
#   Do not drop u.history again.
```

### 9. Escaping values must influence inference

```sev
def identity(x: list[int]) -> list[int]:
    return x

values := [1, 2, 3]
result := identity(values)

# A read-only view cannot become a new independent owner.
#
# Possible contracts:
#
# def identity(x: view list[int]) -> view list[int]:
#     return x
#
# Or:
#
# def identity(x: move list[int]) -> list[int]:
#     return move x
#
# Or:
#
# def identity(x: copy list[int]) -> list[int]:
#     return x
#
# Recommendation:
#   Do not infer an ownership transfer solely from the return type.
#   Require explicit move when transferring caller ownership.
#   Infer a lifetime-bound view if the return contract permits it.
```

### 10. Sharing must not be inferred from ordinary reads

```sev
class Database:
    connection: pointer[Connection]

def inspect(db: Database):
    print(db.connection)

db := Database(...)

inspect(db)

# Inferred:
#   view Database
#
# Not:
#   share Database
#
# Sharing introduces additional ownership and cleanup obligations.
# Do not infer share merely because the object is large or aliased.

shared := share db

# Explicit:
#   shared and db jointly own the designated shared allocation.
#   Cleanup occurs when the last owner releases it.
```

### 11. Copy-on-write should be explicit at the ownership boundary

```sev
data := [1, 2, 3]

snapshot := mirror data

data.append(4)

assert data == [1, 2, 3, 4]
assert snapshot == [1, 2, 3]

# mirror:
#   Initially aliases the same backing allocation.
#   Mutation triggers detachment when necessary.
#
# Optimization:
#   No immediate copy if neither side mutates.
#
# Do not infer mirror automatically for a regular function call.
# Hidden copy-on-write can introduce unpredictable allocations.
```

### 12. Cross-function inference

```sev
def read_first(values: list[int]) -> int:
    return values[0]

def read_wrapper(values: list[int]) -> int:
    return read_first(values)

def mutate_wrapper(values: list[int]):
    values.append(10)

# Callable summaries:
#
# read_first:
#   reads(values[0])
#
# read_wrapper:
#   reads(values[0])
#
# mutate_wrapper:
#   writes(values)
#
# Inferred:
#   read_wrapper(values: view list[int])
#   mutate_wrapper(values: borrow list[int])
#
# MIR propagates effects through the call graph.
# Recursive call graphs require fixed-point effect analysis.
```

### 13. Ownership inference must remain conservative around unknown calls

```sev
def external(value: any):
    # Implementation unavailable.
    ...

def foo(value: list[int]):
    external(value)

# Cannot assume:
#   external only reads value.
#
# Require an imported or declared effect contract.
#
# Example:
#   external(value: view any)
#
# Otherwise:
#   Ownership inference is incomplete.
#   Emit a diagnostic rather than assume the call is harmless.
```

### 14. Separate inferred effects from selected ownership

```sev
# Proposed MIR analysis model.

enum Effect:
    Read
    Write
    Transfer
    Escape
    Allocate
    Destroy

enum Ownership:
    View
    Copy
    Borrow
    Move
    Share
    Mirror

class ParameterEffects:
    reads: set[Place]
    writes: set[Place]
    transfers: set[Place]
    escapes: set[Place]

class InferredParameter:
    type: Type
    ownership: Ownership
    effects: ParameterEffects
    lifetime: Lifetime
    explicit: bool

# Suggested inference:
#
# No mutation + no escape:
#   view for aggregate/nontrivial values
#   copy for cheap scalar values
#
# Mutation + no escape:
#   borrow
#
# Transfer:
#   move, provided the caller explicitly authorizes transfer
#
# Escape:
#   determine whether the result is a borrowed view,
#   an independent copy, or an explicit ownership transfer
#
# Shared ownership:
#   explicit share
#
# Copy-on-write:
#   explicit mirror
```

### 15. Optimize after ownership is proven

```sev
def dot(a: list[float], b: list[float]) -> float:
    result := 0.0

    for i in range(len(a)):
        result += a[i] * b[i]

    return result

# Semantic inference:
#   a: view list[float]
#   b: view list[float]
#
# MIR:
#   effects = {read(a), read(b)}
#   ownership = {view(a), view(b)}
#
# LIR may select:
#   pointer + length
#   slice descriptor
#   registers for small fixed-size arrays
#   specialized vector arguments
#
# Ownership remains unchanged by the physical representation.
```

### 16. Recommended compiler contracts

```sev
# HIR:
#   Resolve types.
#   Resolve explicit ownership annotations.
#   Preserve unannotated parameter ownership as inferred.
#   Establish callable identities and type constraints.
#
# MIR:
#   Resolve operations and their effects.
#   Infer ownership contracts across the call graph.
#   Determine aliasing, field projections, and escapes.
#   Validate transfers, loans, lifetimes, and cleanup.
#   Reject unresolved or invalid contracts.
#
# LIR:
#   Select registers, pointers, descriptors, and memory layouts.
#   Eliminate copies and temporary allocations when proven safe.
#   Refine borrow scopes.
#   Lower cleanup, sharing, and copy-on-write operations.
#
# Rule:
#   Ownership inference determines legal behavior.
#   Optimization determines how that behavior is implemented.
```

### 17. Compiler rejection tests

```sev
test with compiler "infer read-only view":
    values := [1, 2, 3]

    def read(values: list[int]) -> int:
        return values[0]

    accept:
        assert read(values) == 1

    # Expected: view list[int].
    # Expected: zero deep copies.


test with compiler "infer mutable borrow":
    values := [1, 2, 3]

    def modify(values: list[int]):
        values.append(4)

    modify(values)

    accept:
        assert values == [1, 2, 3, 4]

    # Expected: borrow list[int].


test with compiler "reject implicit ownership transfer":
    values := [1, 2, 3]

    def consume(values: list[int]):
        owned := move values

    reject:
        consume(values)

    # A callee cannot silently consume caller-owned storage.
    # Require the caller to authorize move.


test with compiler "reject escaped local view":
    reject:
        def invalid() -> view list[int]:
            values := [1, 2, 3]
            return view values

    # Expected: returned view outlives its owner.


test with compiler "optimize scalar view into register":
    def increment(x: view int) -> int:
        return x + 1

    x := 10

    assert increment(x) == 11

    # Semantic ownership: view.
    # Physical representation: register.
    # No memory allocation required.
```

```sev
# Final proposed defaults:
#
# int, float, bool, char, byte:
#   copy
#
# Fixed-size trivial aggregates:
#   copy when the type's copy contract permits it
#
# list, dict, array, tensor, string, class:
#   view when read-only
#   borrow when mutation is required
#
# Ownership transfer:
#   explicit move at the caller boundary
#
# Reference-counted ownership:
#   explicit share
#
# Copy-on-write:
#   explicit mirror
#
# Unknown effects:
#   require a contract or emit a diagnostic
#
# Most important:
#   Infer view/borrow from effects.
#   Infer cheap copies from type properties.
#   Do not silently infer move/share/mirror.
#   Keep semantic ownership separate from ABI optimization.
```

```sev
class Y[T]:
    a: pointer[T]

# Assuming a is an owning pointer:
#
# Copying Y has two possible meanings:
#
# 1. Shallow copy:
#    Copy 8 bytes on a 64-bit machine.
#    Time:  O(1)
#    Space: O(1)
#    Problem: Two objects now reference the same allocation.
#    Both cannot independently own and destroy that allocation.
#
# 2. Deep copy:
#    Allocate new storage and copy the pointed-to object.
#    Time:  O(copy(T))
#    Space: O(size(T))
#
# Therefore:
#   sizeof(Y) != cost(copy(Y))
```

```sev
class Y[T]:
    a: pointer[T]

    def copy(self) -> Y[T] with {
        # Explicit deep-copy contract.
        complexity.time := complexity.of(copy[T])
        complexity.space := complexity.of(T)
    }:
        return Y(copy(self.a[0]))

# For T = int:
#   Time:  O(1)
#   Space: O(1)
#
# For T = list[int] of length N:
#   Time:  O(N)
#   Space: O(N)
#
# Assuming copy(T) recursively copies owned elements.
```

```sev
# Recommended distinction:
# Complexity should inform ownership inference,
# but must not determine ownership legality.

class Y[T]:
    a: pointer[T]

def inspect(y: Y[int]):
    print(y.a[0])

# Inferred: view Y[int]
# Time: O(1)
# Additional space: O(1)

def modify(y: Y[int]):
    y.a[0] = 10

# Inferred: borrow Y[int]
# Time: O(1)
# Additional space: O(1)

def duplicate(y: Y[int]) -> Y[int]:
    return copy y

# Explicit copy.
# Cost comes from Y's copy contract.
```

```sev
# An important distinction for compiler optimization:

class Y[T]:
    a: pointer[T]

# Three different costs:
#
# sizeof(Y)            = O(1)
# copy(pointer[T])      = O(1)
# copy(owned_object(T)) = O(size(T))  # If T has linear deep-copy cost.
#
# A pointer's copy complexity does not describe the
# complexity of copying the object it owns.
#
# Ownership depth determines which copy contract applies.
```

```sev
# Suggested compiler inference priority:
#
# 1. Ownership correctness.
# 2. Function effects (read/write/escape/transfer).
# 3. Type copy/move contracts.
# 4. Time and space complexity.
# 5. Target-specific lowering cost.
#
# Example:
#
# scalar int:
#   copy -> O(1), usually registers
#
# Y[int]:
#   view -> O(1), no allocation
#   move -> O(1), ownership transfer
#   copy -> O(1), if copying the owned int
#
# Y[list[int]]:
#   view -> O(1)
#   move -> O(1)
#   copy -> O(N), assuming deep-copy ownership
#
# Important:
#   O(1) copies are not necessarily cheap.
#   Reference counting, atomics, allocation, and cache misses
#   can all have O(1) asymptotic complexity.
#
# Recommendation:
#   Use complexity contracts for optimization decisions.
#   Use ownership and effect contracts for correctness.
#   Keep both available to MIR.
```
## Source block → MIR → ownership fields

`sev` blocks use the compiler-test syntax from [ownership examples](../examples/06-ownership/15-compiler-rejections.sev).






### Move x; keep y and z

Fixture contract: `Fields` permits partial moves, fields have disjoint storage, and cleanup releases each remaining owned string exactly once.

```sev
class Fields:
    x: string
    y: string
    z: string

test with compiler "moving x preserves y and z":
    fields := Fields("one", "two", "three")
    taken := move fields.x
    accept:
        assert(fields.y == "two")
        assert(fields.z == "three")
        assert(taken == "one")
    reject:
        print(fields.x)
    reject:
        print(fields)
```

```text
B0:
  fields = construct Fields("one", "two", "three")
  taken = move Place(fields, .x)
  read Place(fields, .y)
  read Place(fields, .z)
  read taken
  cleanup remaining owned places
  Finish

FlowState after move:
  place       initialized  moved  active_loans  destruction_obligation
  fields.x    false        true   []            none (transferred to taken)
  fields.y    true         false  []            release fields.y
  fields.z    true         false  []            release fields.z
  taken       true         false  []            release taken

assert read(fields.y) and read(fields.z) are allowed
assert read(fields.x) and whole-value read(fields) are rejected
assert diagnostic identifies both move and invalid read spans
assert cleanup releases {taken, fields.y, fields.z} once each
assert cleanup never releases fields.x again
assert cleanup order follows the type/scope destruction contract
```

### Move on one branch, then read after the join

```sev
test with compiler "possible move rejects joined read":
    reject:
        def bad(flag: bool):
            x := "owned"
            if flag:
                taken := move x
                print(taken)
            print(x)
```

```text
B0: x = own "owned"; Conditional(flag, B1, B2)
B1: taken = move x; print(view taken); release taken; Jump(B3)
B2: Jump(B3)
B3: print(view x); Finish

state at B3:
  predecessor  x.initialized  x.possibly_moved  x.cleanup
  B1           false          true              none
  B2           true           false             release x
  joined       false          true              only if still owned

assert initialized == intersection(incoming initialized places)
assert possibly_moved == union(incoming possibly moved places)
assert read at B3 is rejected
assert no ReadyMir and no target artifact

valid variant: remove print(x) from B3
assert cleanup releases taken on B1 and x on B2
assert no unconditional release(x) at B3
assert unreachable predecessors do not affect the join

initialization variant:
  B1 initializes x; B2 leaves x uninitialized
  assert joined x.initialized == false
  assert joined read(x) is rejected
```

### Borrow y; reject overlapping mutation; allow z

```sev
test with compiler "field loans use storage projections":
    fields := Fields("one", "two", "three")
    shared := borrow fields.y
    accept:
        fields.z = "changed"
        print(shared)
    reject:
        exclusive := borrow mut fields.y
        print(shared)
    reject:
        taken := move fields.y
        print(shared)
```

```text
loan L0:
  place: Place(fields, .y)
  permission: shared read
  origin: span of "borrow fields.y"
  live through: print(shared)

assert overlap(fields.y, fields.y) == true
assert overlap(fields.y, fields.z) == false under Fields' layout contract
assert mutable loan or move of fields.y conflicts with live L0
assert replacing fields.z releases its old value exactly once
assert different value IDs aliasing fields.y still conflict
assert unknown index i may overlap items[0]
assert loans end according to lifetime/use contracts
assert partition_owner never grants storage permissions
```

### Escaping borrow and repeated loop move

```sev
test with compiler "local borrow cannot escape":
    reject:
        def escaping() -> borrow string:
            local := "temporary"
            return borrow local

test with compiler "backedge carries the move":
    reject:
        def repeated(count: int):
            value := "once"
            index := 0
            while index < count:
                taken := move value
                print(taken)
                index += 1
```

```text
escaping:
  result.loan.place = local
  local.lifetime ends at function exit
  assert result lifetime cannot be satisfied

repeated:
  entry → header: value is initialized
  body → header: value is moved
  assert fixed-point state marks value possibly moved at header
  assert next move is rejected
```

### Cleanup and invalidation

```text
test "every supported exit":
  fixture initially owns x, y, z
  for exit in [normal return, early return, typed error]:
    assert each still-owned place is released exactly once on that path
    if that path transfers y to the result:
      assert callee cleanup releases x and z, but never y
    otherwise:
      assert callee cleanup releases x, y and z
    assert cleanup carries original obligation and generated-action origins
  apply_cleanup twice
  assert second application adds zero actions
  assert changed body is checked by CFG and ownership again

test "callee effects are inputs":
  plan records body, type, provider and callee effect revisions
  change callee from read(x) to move(x)
  assert old plan is rejected even when caller body is unchanged
  assert recursive summaries converge or use declared complete contracts
  assert unknown effects prevent readiness
  assert synchronized memory access still requires valid storage and loans
```

## Responsibilities

| Owner | Responsibility |
|---|---|
| Type provider | Define storage, copy/move/destruction and permitted borrowing behavior. |
| Operation/callable provider | Declare effects and argument/result ownership relationships. |
| Ownership analysis | Propagate permissions and obligations through CFG edges to a fixed point. |
| Cleanup insertion | Add required resolved actions while preserving control and effect order. |
| MIR verifier | Require ownership results for current inputs before publishing readiness. |
| LIR | Preserve these effects or implement an explicitly checked refinement. |

## Data Models

```text
FlowState(place)
  definitely initialized on every incoming path?
  possibly moved on any incoming path?
  active loans and their access permissions
  remaining destruction obligation

OwnershipPlan
  owner: realization identity
  inputs: AnalysisInputs
  facts: operation/edge -> FlowState
  cleanup: ordered CleanupAction records
  effects: checked callable summary
```

## API

```text
analyze(realization, body: checked CfgBody, effects) -> OwnershipPlan | diagnostics
apply_cleanup(realization, plan: OwnershipPlan) -> changed body | diagnostics
verify(realization, body, plan) -> checked ownership revision | diagnostics
```

## Structure

| Proposed file | Responsibility |
|---|---|
| `ownership.sev` | Public analysis and verification entry points. |
| `places.sev` | Storage identities, projections and overlap. |
| `effects.sev` | Resolve type/operation/callable effect contracts. |
| `flow.sev` | State joins, transfer functions and fixed-point traversal. |
| `loans.sev` | Permissions, escapes and loan validity. |
| `cleanup.sev` | Obligation planning and idempotent insertion. |
| `diagnostic.sev` | Original obligation and conflicting-use locations. |
