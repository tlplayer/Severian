# Orbital space-station laboratory

This is an original, clean-room tile-based station simulation inspired by the
emergent systems pressure of games such as Space Station 13. It does not copy
game code, assets, maps, names, or protocols.

The lab composes four real Severian layers:

- `graphics` renders a tactical station map, crew, hazards, telemetry meters,
  and alerts to a deterministic SVG frame;
- three runtime tasks consume typed subsystem jobs through bounded channels and
  return results through a fan-in queue;
- the coordinator sorts concurrent results by job id before applying them, so
  replay and tests remain deterministic; and
- two concurrent observers send the same station snapshot through independent
  native TCP loopback connections using `network`.

Run it with:

```sh
cd docs/lab/space_station
sev build
./target/debug/space-station-lab
sev test main.sev
```

The application writes `/tmp/severian-space-station.svg`. It is deliberately
headless so the native executable and tests run without a display server.

## Current scenario

The station is split by a wall and door into command and engineering sections.
One engineering tile is burning and an adjacent hull tile is breached. Captain,
engineer, and doctor entities are rendered over the grid. Atmosphere, oxygen,
power, and fire jobs execute concurrently, then the main simulation applies the
results in stable order and publishes the snapshot to two observers.

## Pressure points exposed

This first milestone intentionally makes missing stack capabilities visible:

- `graphics` has no interactive window, input events, camera, sprite/image, or
  deterministic bundled-font backend yet, so the lab emits SVG;
- `network` now exposes typed TCP connections and listeners over package-owned
  opaque handles, including listener close and port-zero address inspection;
  framed message codecs, cancellation, and higher-level deadlines remain open;
- channels have no close/select/timeout protocol, so workers receive explicit
  sentinel jobs and the coordinator must know the worker count;
- integration-test syntax is recognized, but the current CLI has no integration
  selector and the native test compiler deliberately skips those tests; the
  application and `main.stdout` therefore provide native TCP acceptance;
- concurrent completion order is nondeterministic by design, requiring an
  explicit replay key and sort before state application; and
- richer simulation needs maps with typed heterogeneous components, spatial
  indexing, delta snapshots, ownership-aware entity transfer, and incremental
  rendering rather than rebuilding one complete frame.

## Next milestones

1. Add typed `TCPListener`/`TCPConnection` lifecycle and length-prefixed station
   packets, then accept actual observer clients on an ephemeral port.
2. Add window/event support to `graphics` and use input commands to move crew,
   operate doors, extinguish fire, and repair breaches.
3. Split atmosphere, power, doors, crew, and hazards into persistent subsystem
   tasks with tick barriers and cancellation.
4. Add visibility, inventory, access control, health damage, pressure diffusion,
   and deterministic replay logs.
5. Add multi-client authority tests, packet loss/reconnect scenarios, profiling,
   and `sev test --plot` telemetry.
