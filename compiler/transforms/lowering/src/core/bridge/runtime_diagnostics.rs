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
static volatile sig_atomic_t sev_runtime_handling_signal = 0;
static const char *sev_runtime_signal_name(int signal_number) {
  switch (signal_number) {
    case SIGABRT: return "SIGABRT";
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS: return "SIGBUS";
    case SIGILL: return "SIGILL";
    case SIGFPE: return "SIGFPE";
    default: return "unknown signal";
  }
}
static void sev_runtime_signal_handler(int signal_number) {
  if (sev_runtime_handling_signal) _Exit(128 + signal_number);
  sev_runtime_handling_signal = 1;
  void *frames[48];
  int frame_count = 0;
  char **symbols = NULL;
#ifdef __linux__
  frame_count = backtrace(frames, 48);
  symbols = backtrace_symbols(frames, frame_count);
#endif
  const char *signal_name = sev_runtime_signal_name(signal_number);
  const char *report_path = getenv("SEVERIAN_RUNTIME_DIAGNOSTIC");
  FILE *report = report_path && *report_path ? fopen(report_path, "w") : NULL;
  if (report) {
    fprintf(report, "SEVERIAN_RUNTIME_DIAGNOSTIC_V3\nE000990\nnative program terminated without a Severian runtime diagnostic\n%s\n%ld\n%ld\n%ld\nprocess received signal %d (%s)\n%d\n", sev_runtime_path, sev_runtime_line, sev_runtime_column, sev_runtime_end_column, signal_number, signal_name, symbols ? frame_count : 0);
    if (symbols) for (int index = 0; index < frame_count; ++index) fprintf(report, "%s\n", symbols[index]);
    fclose(report);
  } else {
    fprintf(stderr, "error[E000990]: native program terminated after signal %d (%s)\nstack trace:\n", signal_number, signal_name);
    if (symbols) for (int index = 0; index < frame_count; ++index) fprintf(stderr, "  %d: %s\n", index, symbols[index]);
  }
  if (symbols) free(symbols);
  _Exit(128 + signal_number);
}
__attribute__((constructor)) static void sev_runtime_install_signal_handlers(void) {
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_handler = sev_runtime_signal_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  sigaction(SIGABRT, &action, NULL);
  sigaction(SIGSEGV, &action, NULL);
  sigaction(SIGBUS, &action, NULL);
  sigaction(SIGILL, &action, NULL);
  sigaction(SIGFPE, &action, NULL);
}
"#;
