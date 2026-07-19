

## 7. Data race

Fixture: `docs/examples/bugs/threads/data_race/invalid.sev`

- [ ] Define task capture access rules.
- [x] Require captured values to be frozen, atomic, or mutex-guarded.
- [ ] Define the operations provided by atomic values.
- [ ] Specify and test shared-mutation diagnostics.
- [ ] Merge the rule into ownership and concurrency documentation.
- [ ] Remove `docs/examples/bugs/threads/data_race`.

End result: concurrent mutable state is uniquely owned or synchronized.
