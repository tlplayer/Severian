# Hash table storage

Private associative storage shared by `set[T]` and `map[K, V]`. Public set and
map semantics stay in their own packages. The initial representation is dense
and deterministic; bucket indexing is an internal optimization that will not
change either public API.
