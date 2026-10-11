SIP-0033: Memory

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors: Timothy PLayer
Created: YYYY-MM-DD
Target:
Supersedes: docs/sip/0009-Control-Memory-Blocks.md
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

### High Level overview
pointer = location
allocation = raw owned memory
storage = typed initialized region
allocator = allocation strategy
ownership semantics = legal program access
array[T] what is returned by allocation

### usafe operations
### pointers
pointer[T]: pointer to address can dereference with pointer[0] notation
address[T]: & address of value 

### Allocation
allocate[T](size:byte|N) ex. allocate[u8](4) gives 4 bytes of memory for u8's type size which is 1 byte. also allocate(4B) raw can be used. 
deallocate[T](memory_pointer:pointer[T]) 

### construction and destruction


### Data Structures
references[T]: references to T
references[T:atomic]: atomic references to T
#Arena of any type, can be forced to be singular type 
arena[T = any]: block of memory deallocates all at once when all is 



## Context

## Goal(s)

1. Easy memory usage for safe operations
2. Efficient memory usage through standard memory.storage interface
3. Memory safety so that no deallocations/reallocations occur
4. No memory leaks
5. Avoid excessive memory usage



## Problems

### 1. Problems
- **Memory location:** Stack, heap, cache, GPU global, GPU shared, GPU local, registers, distributed, and custom memory.
- **Inference vs. explicit:** Compiler inference should be available, but engineers must be able to override placement.
- **Type defaults:** Types should declare preferred memory locations through syntax/type metadata, not MIR heuristics.
- **Allocation strategy:** Memory location, allocator, alignment, layout, and lifetime are separate concerns.
- **Ownership:** Allocation determines where memory lives; pointers and ownership determine who can access, modify, and release it.
- **Escape analysis:** Stack allocations that escape their lifetime must be rejected or explicitly promoted.
- **Hardware constraints:** Not all locations support dynamic allocation, arbitrary pointers, or equivalent access semantics.
- **Custom memory:** Users should be able to define memory locations and allocation strategies without modifying the compiler.

### 2. Proposed Severian Interface
- `allocate[T=u8,F=memory.infer](SIZE:BYTE)` — Allocates using the type's default memory policy.
- `allocate[T, memory.heap](SIZE:BYTE)` — Explicit heap allocation.
- `allocate[T, memory.stack](SIZE:BYTE)` — Explicit stack allocation.
- `allocate[T, memory.infer](SIZE:BYTE)` — Compiler selects placement from type constraints, lifetime, and target.
- `allocate[T, memory.gpu.shared](SIZE:BYTE)` — GPU shared-memory allocation.
- `allocate[T, memory.custom](SIZE:BYTE)` — User-defined memory location.
- `pointer[T]` — Typed pointer; location inferred from its allocation.
- `pointer[T, memory.heap]` — Pointer constrained to heap memory.
- `memory.layout[T](alignment=64, packing=...)` — Explicit memory layout contract.
- `memory.location` — Extensible interface for stack, heap, GPU, and custom backends.
- `class A: _memory_location = memory.heap` — Type-defined default allocation policy.
- **Resolution:** Syntax/type definitions supply memory constraints → HIR resolves allocation intent → MIR performs escape/lifetime analysis and validates placement → LIR lowers to target-specific memory operations.

**Key distinction:** `memory.infer` should be a policy, not a physical memory location. Type defaults establish intent; the compiler validates feasibility.

## Examples

Pointers are an amalgimation of a couple concepts, they serve as boxes, and slices just with ownership semantics specified. 
All these operations are unsafe but they are omitted outside the blocks for brevity

Pointers:
- owns allocation
- contains initialized T
- move transfers ownership
- drop destroys T
- drop deallocates storage

```
a:int = 0
# a has been moved to p default op is move for pointers
p:pointer[int] = &a
# Can view a slice, compiler fails on drop q because q doesn't own the view on p
q:pointer[int] = view p[0]
```
deallocate         
drop(owning pointer) → destroy initialized contents
drop(view pointer)  → compiler rejection
view leaves scope   → end access; no destruction or deallocation


### Allocating on the stack
```
# scalar types usually are on the stack
a:int = 0
a:static int = 0

# collection types are usually on the heap
b:list = [1,'a',"abc"]

# collections can also be on the stack if static
b:list = [1,2,3,4]

# 
```

### Allocation 
allocation lowers to MLIR 
for 
```
unsafe: 
    pointer[int] = allocate[int](128)
```
which translates to the following MLIR:
```mlir
//unsafe: 
//    int_pointer: pointer[int] = allocate[int](128)
%buffer = memref.alloc() : memref<128xi32>

//unsafe: 
//    value = int_pointer[i]
%v = memref.load %buffer[%i] : memref<128xi32>

//unsafe: 
//    int_pointer[i] = value
memref.store %value, %buffer[%i] : memref<128xi32>

//unsafe: 
//    deallocate[int](int_pointer)
memref.dealloc %buffer : memref<128xi32>
```

## Allocation optimizations

**Arena memory optimizes allocation and deallocation of groups of objects. Pool memory optimizes repeated allocation and deallocation of similarly sized objects.**

Both reduce calls to general-purpose allocators, but they solve different problems.

## 1. Arena vs. Pool

| Property | Arena | Pool |
|---|---|---|
| Allocation | Sequential allocation from a region | Take a reusable block |
| Deallocation | Release the entire region together | Release individual blocks |
| Reuse | Usually after reset | Immediately after individual free |
| Object sizes | Variable | Usually fixed or size-classed |
| Fragmentation | Low internal management overhead | Depends on size classes |
| Allocation speed | Often pointer increment | Often free-list operation |
| Memory efficiency | Can retain dead objects | Can reuse dead objects |
| Best application | Shared lifetimes | Repeated object creation/destruction |

### Arena

```text
Arena: 1024 bytes

Allocate A (100 bytes)
Allocate B (200 bytes)
Allocate C (300 bytes)

[A:100][B:200][C:300][free:424]

Drop B:
[A:100][B:200][C:300][free:424]
        ^
        Memory not individually reclaimed

Reset:
[free:1024]
```

Useful for parsers, ASTs, temporary computations, and collections whose members share a lifetime.

### Pool

```text
Pool: 4 blocks, 64 bytes each

[A][B][C][free]

Drop B:
[A][free][C][free]

Allocate D:
[A][D][C][free]
```

Useful for linked lists, trees, graph nodes, queues, and objects frequently created and destroyed.

A pool can also manage variable-sized allocations through size classes.

## 2. What should Severian collections use?

| Collection | Default recommendation | When arena helps | When pool helps |
|---|---|---|---|
| `array[T,N]` | Inline storage | Many arrays sharing a lifetime | Repeated arrays of the same size |
| `list[T]` | Contiguous dynamic buffer | Lists built and discarded together | Reusing backing buffers |
| `stack[T]` | Contiguous dynamic buffer | Temporary stacks | Repeated stack instances |
| `queue[T]` | Circular buffer | Temporary queues | Linked-node queues with churn |
| `map[K,V]` | Hash table | Maps discarded together | Node-based maps or reusable buffers |
| `set[T]` | Hash table | Temporary sets | Node-based sets |
| `tree[T]` | Node allocation | Entire trees discarded together | Frequent insertion/deletion |
| `graph[T]` | Contiguous adjacency structures or nodes | Graph-wide lifetime | Dynamic graph nodes and edges |

**Important:** The collection's data structure matters more than its allocator.

A contiguous `list[T]` doesn't benefit from a pool per element because its elements should generally reside in one buffer.

A linked list does benefit from pooled nodes because each node requires independently managed storage.

## 3. Severian allocation examples

Default behavior:

```python
# Fixed size; inline or stack storage when possible
a := allocate[array[int, 32]]()

# Dynamic contiguous allocation
b := allocate[list[int]]()

# Hash table
c := allocate[map[int, float]]()

# Tree with individually allocated nodes
d := allocate[tree[int]]()
```

Explicit strategies:

```python
# All nodes share the tree's lifetime
ast := allocate[
    tree[Node],
    strategy=memory.arena
]()

# Nodes frequently inserted and removed
tree := allocate[
    tree[Node],
    strategy=memory.pool
]()

# Temporary list with known capacity
values := allocate[
    list[int],
    strategy=memory.arena,
    capacity=1000
]()
```

The important difference between the tree examples:

- Arena tree: Individual node deletion does not reclaim arena space.
- Pool tree: Deleted nodes return their slots to the pool for reuse.

An arena-backed list with repeated growth can waste memory because old buffers may remain allocated until the arena is released. Reserving capacity avoids this.

## 4. How the compiler should infer strategies

Consider:

```python
def build_tree(data: list[int]) -> tree[int]:
    result := tree[int]()

    for x in data:
        result.insert(x)

    return result
```

The compiler cannot automatically select an arena just because this is a tree.

It must establish whether individual nodes can be deleted, whether storage escapes, and whether nodes share a lifetime.

Compare:

```python
def process(data: list[int]):
    tree := tree[int]()

    for x in data:
        tree.insert(x)

    analyze(tree)

    # Entire tree dies here
```

This is a candidate for arena allocation.

Now:

```python
def process(tree: tree[int]):
    tree.insert(10)
    tree.remove(20)
    tree.insert(30)
```

This is a candidate for pooled nodes, particularly if these operations occur repeatedly.

However, **the compiler should not select a pool simply because it sees insertions and deletions**. It needs cost information or explicit strategy selection.

## 5. What belongs in `memory` and MIR?

I would separate three decisions:

1. **Collection representation:** Contiguous buffer, linked nodes, tree nodes, hash buckets.
2. **Allocation strategy:** Stack, arena, pool, general-purpose allocator.
3. **Memory location:** CPU heap, GPU global, GPU shared, etc.

These are independent dimensions.

For example:

```python
nodes := allocate[
    tree[int],
    strategy=memory.pool,
    location=memory.heap
]()
```

MIR could represent this as:

```text
Collection:
    type: tree[int]
    representation: node_based

Allocation:
    location: heap
    strategy: pool
    block_size: sizeof(tree.node[int])
    lifetime: inferred
    ownership: inferred
```

The collection's implementation determines *what* gets allocated. `memory` determines *how and where* those allocations occur.

One design decision is especially important for Severian:

**Should allocation parameters apply only to the collection object or also to its internally allocated elements?**

I recommend that `allocate[tree[T], strategy=memory.pool]()` propagates the strategy to the tree's internal allocations by default. Otherwise, the caller selects a pool but the tree's nodes may still allocate through the general heap.

The type's syntax can define which internal allocations inherit the policy.

This allows you to optimize collections without introducing separate `arena_list`, `pool_tree`, or `pool_map` types.

### Data models

pointer[T]
    address
    allocation_id
    offset


## Testing



