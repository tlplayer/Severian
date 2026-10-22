SIP-0025: Testing

Status: *Draft* | Accepted | Implementing | Implemented | Rejected | Superseded
Tags: Testing
Authors: Timothy
Target:
Supersedes:
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

- mock: when this occurs it should return this instead
- throws: the following operation should throw something
- assert: if this is 
- when: if a condition happens these mocked values and or values should equal this 
- fix: during the execution of this block this condition is *always true (if we leave the block we can't promise no change)
- suffix: at the end of this test, this should be true
- prefix: at the start of this block, this should be true
- with: conditions for the test as well as packed imports test with profile does many imports/exposes variables and tests for devices. 



## Examples & Problem(s) & Testing

```
#Basic function test
test "foobar":
    foobar()

# With allows sev test --integ to only call the functions within that scope inclusive 
test with integ "Print outputs X":
    assert("X" in stdout)

test with compiler "MIR error":
    reject:
        x = view 1
        #We would be modifying a view which the compiler rejects even though it's a literal
        x += 1
    accept:
        x = borrow 1
        x += 1

test with profile "Operation takes less than 10s" with
{
    fix profile.time < 10s,#If we ever go over 10s clip the test
    suffix profile.time < 5s, # If we go over 5s by the end we've failed
}
    foobar()
```



### Fuzz

Fuzz will randomly generate input to the function to get coverage. Store the tests that add coverage in package.pkg/debug/test/fuzz/filename.sev in order to add more/find more tests

```
def fib(n: int) -> int with {
    0 <= n < 10
}:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


test with fuzz "fibonacci constraints":
    fuzz(fib)
```

fuzz(fib)
1. Inspect parameter types and declared constraints.
2. Generate random inputs, boundary values, and mutations.
3. Execute the function.
4. Detect crashes, constraint violations, and undefined behavior.
5. Track code and branch coverage.
6. Minimize failing inputs into reproducible test cases.
Unlike profile, fuzzing doesn't need to generate inputs in increasing order.

### Profile

Profile is used to make sure given input X the function's runtime/big O is upheld. 

```sev
def fib(n: int) -> int with {
    fib.complexity.time = BigO.exponential(n),
    fib.complexity.space = BigO.linear(n)
}:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


test with profile "fibonacci complexity":
    profile(fib)
```

profile(fib) should:
1. Inspect fib's parameter types and complexity contracts.
2. Generate inputs of increasing size.
3. Execute fib with instrumentation.
4. Measure operations, execution time, and peak memory.
5. Compare observed growth against the declared Big-O bounds.
6. Report violations and generate time/space complexity graphs.

To run this explicitly for a file, 
```bash
def fib(n: int) -> int with {
    0 <= n <= 40,
    fib.complexity.time = BigO.exponential(n),
    fib.complexity.space = BigO.linear(n)
}:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


test with profile "fibonacci complexity":
    profile(fib)

cli:
sev test file.sev --profile

PROFILE fib

Input domain:       [0, 40]
Inputs tested:      41 / 41
Coverage:           Exhaustive over declared input domain

Time:
    Declared:       O(2^N)
    Observed:       Θ(φ^N)
    Result:         CONSISTENT

Space:
    Declared:       O(N)
    Observed:       Θ(N)
    Result:         CONSISTENT

Execution:
    Max input:      40
    Max depth:      40
    Graphs:         time, space

Overall:            PASS (empirical, bounded domain)
```
### Mutate

```sev

def foobar(a):
    if a < 10:
        return -1
    return 1

test with mutate "Mutations on the function should cause tests to fail":
    mutate(foobar)
```

### Differential

Differential is useful in today's coding age to compare 1 implementation n other implementations

```sev
def foo(a: int):
    return a + 1

def bar(b: float):
    return b + 1

test with differential "implementation 1 makes implemenation 2":
    # This will pass value adheres to int max -> int min randomly and cross compare. Appending differentials to the test block
    differential(foo,bar)

```


### Memory
I would use **`memory`**. It is consistent with Severian's existing `memory[T]` interface, and it describes what the test instruments rather than a particular implementation such as Valgrind.

The important design choice is that **`test with memory` instruments every memory operation executed inside the test**, including operations inside called functions.

## 1. Basic API

```sev
test with memory "Find memory leaks and invalid accesses":
    foobar()
```

By default, `memory` should detect:

| Category | Bugs detected |
|---|---|
| Leaks | Lost allocations, indirect leaks, unreleased resources |
| Bounds | Buffer overflow, buffer underflow, out-of-bounds indexing |
| Lifetime | Use-after-free, use-after-return, use-after-scope |
| Deallocation | Double-free, invalid free, mismatched allocator/deallocator |
| Initialization | Reads of uninitialized memory |
| Ownership | Invalid ownership transfers, dropped references, incorrect shared ownership |
| Alignment | Accesses violating alignment requirements |
| Allocation | Invalid allocation sizes, integer overflow in allocation calculations |

This combines capabilities found in Valgrind Memcheck, AddressSanitizer, LeakSanitizer, and MemorySanitizer. [valgrind.org](https://valgrind.org/docs/manual/mc-manual.html/manual-core-adv.html?utm_source=chatgpt.com)

The handler should enable runtime instrumentation, run the test, and fail when it detects a memory violation.

## 2. Using Severian's existing `with` constraints



```sev
test with memory "Memory safety" with {
    fix memory.errors == 0,
    suffix memory.leaks == 0
}:
    foobar()
```

Semantics:

- `fix memory.errors == 0`: No memory violation is permitted during execution.
- `suffix memory.leaks == 0`: At completion, no allocation may remain classified as leaked.

The `memory` test capability exposes a runtime statistics object:

```sev
memory.errors
memory.leaks
memory.allocations
memory.deallocations
memory.live_bytes
memory.peak_bytes
memory.invalid_accesses
memory.uninitialized_reads
memory.use_after_free
```

This also permits tests with specific resource expectations:

```sev
test with memory "Memory usage" with {
    fix memory.live_bytes < 100MB,
    suffix memory.leaks == 0
}:
    foo()
```

Notice that `memory.live_bytes` does not have to equal zero when the test finishes. A program may legitimately retain allocations.

What matters is whether those allocations are still owned, reachable, or expected to survive the test.

## 3. Testing `memory[int](100)`

Consider direct memory access:

```sev
def foo():
    a = memory[int](100)

    for i in range(100):
        a[i] = i

    a[100] = 42  # Invalid access


test with memory "Bounds checking":
    foo()
```

The test should report:

```text
MEMORY ERROR: OutOfBounds

Allocation:
    Type:        int
    Elements:    100
    Valid range: [0, 99]

Access:
    Index:       100
    Operation:   write

Location:
    foo(): a[100] = 42

Result: FAIL
```

This becomes especially useful for raw memory access, C interoperability, custom allocators, and compiler-generated operations.

Safe Severian code may already reject this access through bounds checking. The memory test should additionally catch violations arising from unsafe operations, incorrect lowering, or foreign code.

## 4. Leak detection needs an allocation ledger

This is where Severian's memory model can do more than simply count allocations and deallocations.

Every allocation should have a record:

```text
Allocation {
    id
    address
    size
    alignment
    memory_location
    allocator
    owner
    references
    allocation_site
    deallocation_site
    state
}
```

The memory runtime maintains this ledger during testing.

For example:

```text
Allocation 17
    Type:         int[100]
    Size:         800 bytes
    Location:     heap
    Owner:        foo
    State:        allocated

Allocation 18
    Type:         int[200]
    Size:         1600 bytes
    Location:     heap
    Owner:        bar
    State:        freed
```

At the end of the test, the handler examines the remaining allocations.

Importantly, there are multiple leak classifications.

| Classification | Meaning | Default |
|---|---|---|
| Definitely lost | No valid reference to the allocation remains | Fail |
| Indirectly lost | Allocation is reachable only from another lost allocation | Fail |
| Possibly lost | Reachability cannot be established reliably | Report |
| Still reachable | Allocation remains accessible but has not been released | Report |
| Intentionally retained | Allocation was transferred to a valid owner outside the test | Pass |

This follows the distinction used by Valgrind's leak detector, with additional ownership information available from Severian. [valgrind.org](https://valgrind.org/docs/manual/faq.html?utm_source=chatgpt.com)

For example:

```sev
test with memory "Allocator lifecycle":
    exercise_allocator()
```

Could produce:

```text
MEMORY LEAK DETECTED

Allocation 47:
    Size:          400 bytes
    Allocated at:  allocator.sev:42
    Last owner:    buffer
    Last reference lost at: allocator.sev:51

    Classification: DefinitelyLost

Summary:
    Allocations:       142
    Deallocations:     141
    Lost allocations:  1
    Lost bytes:        400

Result: FAIL
```

For `memory[int](100)`, whether failing to call `deallocate` constitutes a leak depends on the allocation's ownership semantics.

If the object owns the allocation and its scope cleanup releases it, there is no leak. If the allocation requires explicit release and its final reference is lost, there is.

The ledger needs to understand both.

## 5. Finding bugs during execution

Leak checking occurs at a checkpoint, generally after the test completes.

Other errors need to be detected during execution.

I would instrument MIR operations:

```text
memory.allocate
memory.deallocate
memory.load
memory.store
memory.copy
memory.reallocate
memory.transfer
memory.drop
```

These are conceptual operations; they do not all require distinct public MLIR operations.

Instrumentation should check:

**Allocation:** Was a valid region created? Was its size calculated correctly?

**Load:** Is the address valid, the allocation live, the access within bounds, and the data initialized?

**Store:** Is the address valid, writable, and within bounds?

**Deallocation:** Is this the correct allocator? Has the allocation already been released?

**Ownership:** Is the reference valid? Did a move invalidate its previous owner? Did the final shared owner release the allocation?

For initialization tracking, the runtime needs shadow metadata identifying initialized bytes. For use-after-free detection, it can poison freed regions and optionally quarantine them before reuse.

This is consistent with the techniques used by existing sanitizers, but Severian could integrate them with its ownership and allocation metadata.

### One useful distinction


- **Ownership violations:** Detectable using MIR ownership analysis, including many errors before execution.
- **Memory violations:** Detected from actual runtime accesses, particularly through raw pointers, FFI, and incorrect compiler lowering.

Memory testing should report both without requiring the user to understand which compiler stage found the issue.

---

## 6. Memory testing should compose with fuzzing

This is probably the most useful combination:

```sev
test with memory, fuzz "Memory safety under generated inputs":
    fuzz(foo)
```

The handlers have separate responsibilities.

`fuzz` generates inputs and explores execution paths.

`memory` instruments the executions and detects memory failures.

For example, fuzzing might discover that a collection corrupts its allocator state only after this sequence:

```text
append(1)
append(2)
remove(0)
append(3)
clear()
append(4)
```

The memory handler catches the invalid access.

The fuzzer shrinks the sequence and stores it as a regression test.

The same approach works for concurrency:

```sev
test with memory, schedule "Concurrent memory safety":
    schedule(foo)
```

Here, `schedule` varies thread interleavings, while `memory` checks memory validity. Detecting races themselves would also require race-specific instrumentation.

---

### Implementation in the compiler


Use the existing memory infrastructure:

```text
MIR
 │
 ├── CFG verification
 │
 ├── Ownership analysis
 │
 ├── Cleanup insertion
 │
 ├── Memory instrumentation (test builds)
 │     ├── Allocation tracking
 │     ├── Bounds checking
 │     ├── Lifetime checking
 │     ├── Initialization tracking
 │     └── Ownership validation
 │
 ↓
LIR
 │
 ↓
C / MLIR backend
 │
 ↓
Memory test runtime
       ├── Allocation ledger
       ├── Shadow memory
       ├── Violation reports
       └── Leak analysis
```

This has three advantages.

First, Severian knows the source-level type and ownership information associated with an allocation.

Second, instrumentation can remain enabled across compiler lowering, allowing tests to expose mistakes introduced by memory transformations or cleanup insertion.

Third, every allocation interface can use the same tracking infrastructure:

```sev
allocate[int](100)
memory[int](100)
list[int]()
array[int, 100]()
```

The scope should eventually include C allocation APIs, GPU allocations, arenas, and memory pools. Those require allocator-specific hooks; MIR instrumentation alone cannot observe memory operations inside uninstrumented foreign libraries.

For pools and arenas, the ledger should understand parent-child relationships. Releasing an arena legitimately releases its allocations without requiring each individual object to be freed separately.

---

**Default memory validation**

```sev
test with memory "Memory safety":
    foo()
```

**Specific memory contracts**

```sev
test with memory "Allocation constraints" with {
    fix memory.errors == 0,
    fix memory.live_bytes < 100MB,
    suffix memory.leaks == 0
}:
    foo()
```

**Generated memory testing**

```sev
test with memory, fuzz "Find memory bugs":
    fuzz(foo)
```

One final requirement: **the memory checker must distinguish bugs from intentional resource retention.**

A global cache, a returned allocation, or an arena transferred to another owner should not automatically count as leaked memory.

That is where Severian's existing `move`, `borrow`, `view`, `share`, and `mirror` semantics become useful: they give the checker information about which references represent ownership, which are temporary, and when an allocation should actually become unreachable.

I would implement the allocation ledger and leak classification first, then add bounds, lifetime, and initialization instrumentation. Those components form the foundation for the rest of `test with memory`.