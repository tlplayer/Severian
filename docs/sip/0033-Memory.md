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

### usafe operations
### pointers
pointer[T]: pointer to address can dereference with pointer[0] notation
address[T]: & address of value 

### Allocation
allocate[T](size:byte|N) ex. allocate[u8](4) gives 4 bytes of memory for u8's type size which is 1 byte. also allocate(4B) raw can be used. 
deallocate[T](memory_pointer:pointer[T]) 

### construction and destruction


### Data Structures
box[T]: box
references[T]: references to T
references[T:atomic]: atomic references to T
arena: block of memory deallocates all at once when all is 



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


### Box


## Problem(s)

## Testing


