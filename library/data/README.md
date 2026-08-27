# Data

`data` is Severian's format-independent table layer. CSV, JSON, and YAML
documents implement the shared `data_format.Data` contract (`text`, `columns`,
and `values`) and can be adapted to the same relational table API:

```sev
import data
import csv

dialogs := data.table(csv.read("dialogs.csv"))
    .where_not_empty("text")
    .unique(["npc_name", "text"])
    .lower("dialog_type")

items = dialogs.collapse_text(
    "dialog_type", ["item_text"], ["npc_name"], "text",
)
```

`collapse_text` follows dataframe-style grouped aggregation: unmatched rows
retain their order, grouped rows are sorted by key, and one merged row per
non-empty group is appended. Text values are globally deduplicated, matching
the BetterQuest Python preparation pipeline.

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
writes. `Data` owns rows, columns, schema inference, projection,
transformation, filtering, ordering, grouping, deduplication, lazy plans, and
SQL-style queries. New formats implement `data_format.Data` to expose the same
tabular contract.

## Query expressions

`Data` also has a lazy query path. Column expressions are data, rather than
opaque callbacks, so the plan can be inspected and eventually optimized before
execution:

```sev
import data

adults = people
    .where(data.greater_or_equal("age", 18))
    .select(["name", "age"])
    .sort_descending("age")
    .limit(100)

print(adults.explain())
result = adults.collect()
```

Boolean trees use `data.all_of(...)` and `data.any_of(...)`. The callback forms
remain available for application-defined eager operations that cannot be
represented as query IR.

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
silently switching to another execution engine.
