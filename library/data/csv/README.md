# CSV

`csv` owns in-memory and path-backed CSV documents. Importing `file` loads its
reader provider, so `file.read("data.csv")` dispatches through the standard
`File` registry. CSV documents implement `data.DataSource`, and path reads
return `data.Data` so relational operations are immediately available.
