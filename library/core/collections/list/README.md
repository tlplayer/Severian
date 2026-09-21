# list[T]

The list package owns sequence operations and borrowed slice regions. Storage
ownership, growth, bounds, initialized elements and destruction are delegated to
core.storage, which uses core.memory. Iterators and slices borrow their list;
they never release its allocation. Index reads borrow elements and removals move
them out. copy creates independent element ownership.
