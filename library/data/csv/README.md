# CSV

`csv` owns in-memory and path-backed CSV documents. Importing `file` loads its
reader provider, so `file.read("data.csv")` dispatches through the standard
`File` registry. CSV documents implement the shared `data_format.Data`
contract.

CSV implements `data.Source`. Call `.data()` for format-independent table
operations, then pass the result to `.set_data()` when the document should be
serialized or written again.
