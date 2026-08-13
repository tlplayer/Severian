# Data

`data` is Severian's format-independent table layer. Formats implement
`data.Source` when they can expose tabular values:

```sev
table = file.read("dialogs.csv").data()
table = table.require_columns(["npc_name", "text"])
table = table.filter(|row| size(row.get("text")) > 0)
table = table.unique(["npc_name", "text"])
```

The source document continues to own parsing, quoting, encoding, paths, and
writes. `Data` owns rows, columns, schema projection, transformation, filtering,
ordering, and deduplication. CSV, JSON, Parquet, SQL, and Arrow sources can
therefore share the same operations without duplicating them.

Lambdas are first-class closures, so `filter`, `transform`, and `unique_by` can
capture application values and pass through ordinary library helper calls.
