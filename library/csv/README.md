# CSV

`csv` owns in-memory and path-backed CSV documents. `file.read("data.csv")`
uses this package through a reader adapter and returns the same `CSV` type.

CSV implements `data.Source`. Call `.data()` for format-independent table
operations, then pass the result to `.set_data()` when the document should be
serialized or written again.
