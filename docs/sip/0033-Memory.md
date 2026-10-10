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

### Data models

pointer[T]
    address
    allocation_id
    offset


## Testing



