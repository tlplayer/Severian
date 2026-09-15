# package.cache

Verified content-addressed storage, independent of compiler IR and package
resolution. `Store(root)` owns immutable blobs and small JSON completion records.
`put_text` and `put_file` return `CacheValue(path, checksum, reused)`.
`lookup` verifies the record and blob; corruption is a cache miss. `publish`
atomically replaces a completed receipt. Callers coordinate concurrent writers
to a receipt; `package.build` uses a lock per stage key.

Owned temporary files can be consumed on publication, avoiding another bulk
copy. Identical bytes share a blob, regardless of the source edit that produced
them. SHA-256 and filesystem operations currently use the hosted Unix boundary.
