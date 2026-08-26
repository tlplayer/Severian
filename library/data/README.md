# Data

`data` is Severian's format-independent table layer. Serializable documents
implement the shared `data_format.Data` contract, and formats implement
`data.Source` when they can additionally expose tabular values:

```sev
table = file.read("dialogs.csv").data()
table = table.require_columns(["npc_name", "text"])
table = table.filter(|row| size(row.get("text")) > 0)
table = table.unique(["npc_name", "text"])
groups = table.group(["npc_name"])
```

Selected groups can be replaced by an aggregate row while unselected rows keep
their source positions. The merged row occupies the first selected row's
position:

```sev
merged = table.group_merge(
    |row| row.get("dialog_type") == "item_text",
    ["npc_name", "zone", "quest_id"],
    |group| merge_pages(group),
)
```

The merge callback receives the complete `Data` group and returns one `Row`,
so aggregation policy remains application-defined.

The source document continues to own parsing, quoting, encoding, paths, and
writes. `Data` owns rows, columns, schema projection, transformation, filtering,
ordering, grouping, and deduplication. CSV, JSON, Parquet, SQL, and Arrow sources can
therefore share the same operations without duplicating them.

Lambdas are first-class closures, so `filter`, `transform`, and `unique_by` can
capture application values and pass through ordinary library helper calls.

## Query expressions

`Data` also has a lazy query path. Column expressions are data, rather than
opaque callbacks, so the plan can be inspected and eventually optimized before
execution:

```sev
from data import Data, column

adults = people
    .where(column("age").greater_or_equal(18))
    .select(["name", "age"])
    .sort_descending("age")
    .limit(100)

print(adults.explain())
result = adults.collect()
```

Boolean trees use `.and_(...)`, `.or_(...)`, and `.negate()`. This is the
compiler-safe expression API today; operator sugar such as `data["age"] >= 18`
can lower to the same tree when user-defined operator and index dispatch land.
The existing callback form, `data.filter(|row| ...)`, remains eager and is kept
for application-defined predicates that cannot be represented as query IR.

Instance SQL is a second frontend to those same query steps:

```sev
result = people.sql("""
    SELECT name, age
    FROM self
    WHERE active = true
    ORDER BY age DESC
    LIMIT 100
""").collect()
```

The initial SQL subset is deliberately small: projection, one `WHERE`
comparison, `ORDER BY`, and `LIMIT`. Unsupported clauses fail rather than
silently switching to another execution engine. CSV remains a `data.Source`;
JSON and YAML retain document semantics unless explicitly adapted from a
record-shaped value.
