# package.profile

Timing and memory measurements for package builds and compiler stages.
`measure_command(arguments)` captures exit status, stdout, stderr, elapsed time,
user/system CPU time, and peak resident memory. CPU/RSS measurements currently
use `/usr/bin/time` on the hosted Unix system. Units are seconds and KiB.

`profile_begin` / `profile_stage` retain the compiler's `SEVERIAN_TIMINGS` TSV
hook and write JSON under `package.pkg/debug/profile`. Native compiler-step
reports include measurements and cache-hit flags, so a slow step can be
distinguished from an expensive step that was skipped.

Frontend RSS is the process high-water mark, not a per-pass allocation delta.
Native command RSS is measured for that command; these two measurements should
not be added together or interpreted as retained memory. Profiling observes
compiler stages without depending on their IR representation.
