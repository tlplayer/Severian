pub(super) const SOURCE: &str = r#"static _Thread_local const char *sev_runtime_path = "";
static _Thread_local int64_t sev_runtime_line = 0;
static _Thread_local int64_t sev_runtime_column = 0;
static _Thread_local int64_t sev_runtime_end_column = 0;
void __sev_runtime_set_site(void *path_raw, int64_t line, int64_t column, int64_t end_column) { sev_runtime_path = path_raw ? (const char *)path_raw : ""; sev_runtime_line = line; sev_runtime_column = column; sev_runtime_end_column = end_column; }
static _Noreturn void sev_runtime_fail(const char *code, const char *message, const char *detail) {
  const char *report_path = getenv("SEVERIAN_RUNTIME_DIAGNOSTIC");
  if (report_path && *report_path) {
    FILE *report = fopen(report_path, "w");
    if (report) {
      fprintf(report, "SEVERIAN_RUNTIME_DIAGNOSTIC_V2\n%s\n%s\n%s\n%ld\n%ld\n%ld\n%s\n", code, message, sev_runtime_path, sev_runtime_line, sev_runtime_column, sev_runtime_end_column, detail ? detail : "");
      fclose(report);
      _Exit(70);
    }
  }
  fprintf(stderr, "error[%s]: %s\n", code, message);
  if (sev_runtime_path && *sev_runtime_path) fprintf(stderr, " --> %s:%ld:%ld\n", sev_runtime_path, sev_runtime_line, sev_runtime_column);
  if (detail && *detail) fprintf(stderr, " note: %s\n", detail);
  _Exit(70);
}
static _Noreturn void sev_runtime_fail_bounds(const char *kind, int64_t index, int64_t length) {
  char detail[192];
  snprintf(detail, sizeof(detail), "%s index %ld is invalid; length is %ld", kind, index, length);
  sev_runtime_fail("E000910", "index is out of bounds", detail);
}
static _Noreturn void sev_runtime_fail_invariant(const char *detail) { sev_runtime_fail("E000980", "Severian runtime invariant failed", detail); }
void __sev_runtime_fail_assertion(void) { sev_runtime_fail("E000902", "assertion failed", "the asserted condition evaluated to false"); }
void __sev_runtime_fail_division_zero(void) { sev_runtime_fail("E000920", "division by zero", "the divisor evaluated to zero"); }
"#;
