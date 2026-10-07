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

```
                    language ownership
             view / borrow / move / copy / mirror
                           |
                           v
                     Object / Container
                           |
                           v
                       Storage[T]
                /          |           \
             owned       arena       external
                \          |           /
                           v
                       Allocation
                           |
                           v
                        Allocator
                /        |        \
             heap      arena      pool
                           |
                           v
                      core.memory
                           |
                           v
                       pointer[T]
                           |
                           v
                         MLIR
```
## Goal(s)

1. Easy memory usage for safe operations
2. Efficient memory usage through standard memory.storage interface
3. Memory safety so that no deallocations/reallocations occur
4. No memory leaks
5. Avoid excessive memory usage

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

## Responsibilities

## Data Models

## API

## Structure

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

Allocation
    base
    size
    alignment
    allocator
    initialized_range
    owner

## Problem(s)
| Problem | Required test |
|---|---|
| Allocation is initially uninitialized | Allocate four elements → reading before writing is rejected when provable |
| Initialization can contain holes | Initialize indexes `0` and `3` → cleanup destroys exactly those two |
| Partial moves create new holes | Move out index `0` → cleanup does not destroy it again |
| Slice bounds differ from allocation bounds | Slice covers one element → accessing its second element fails even if backing allocation is larger |
| Owning pointer accidentally copied | Ordinary aliasing cannot create two release obligations |
| `&a` points into expired local storage | Returning ownership preserves storage; returning a local view is rejected |
| Same address reused | An old handle never becomes valid merely because another allocation occupies that address |
| Allocation arithmetic overflows | Excessive `count * sizeof(T)` fails before allocation |
| Constructor fails halfway | Destroy completed fields/elements only; release allocation once |
| Owner moved on one branch | Later unconditional owner use is rejected; cleanup follows the actual branch |
| Arena reset with outstanding access | Reset followed by use of the old pointer is rejected |
| `arena[any]` contains different types | Each initialized object gets its own correct destructor and alignment |
| Reference cycles | Weak edges or a defined cycle policy; counts alone cannot guarantee reclamation |
| Atomic reference counts | Concurrent handle operations are safe; payload mutation still requires its own access rules |
## Testing


