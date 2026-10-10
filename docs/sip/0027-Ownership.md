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
