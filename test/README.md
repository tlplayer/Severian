
# Make sure the build time is clean and not blowing up 
```sh
sudo perf record -F 99 -e cpu-clock:u --call-graph dwarf,16384 \
  -o /tmp/severian-build.perf.data \
  -- sudo -u "$(id -un)" env SEVERIAN_FORCE_REBUILD=1 \
  ./package.pkg/release/sev build sev_compiler --bin sev_compiler
```