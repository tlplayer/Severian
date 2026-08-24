Direction or idea of how packages should work

I would document `05-building` as two layouts: the checked-in package source and the unpacked contents of the generated `package.pkg`.

```text
/home/tplayer/Documents/Severian/docs/examples/05-building/
│
├── README.md
│   # Explains this example:
│   #   sev build
│   #   sev run
│   #   sev test
│   #   package.pkg contents
│   #   compatibility resolution
│   #   container fallback
│
├── package.toml
│   # Human-authored package manifest.
│   #
│   # Defines:
│   #   package name/version/edition
│   #   source targets
│   #   binaries
│   #   build outputs
│   #   supported targets
│   #   package distribution policy
│   #   container fallback policy
│
├── package.lock
│   # Machine-generated dependency lock.
│   #
│   # Locks:
│   #   Severian dependencies
│   #   versions
│   #   package hashes
│   #   native/build-tool dependencies
│   #
│   # Does NOT describe the machine used to build the package.
│
└── src/
    │
    ├── lib.sev
    │   # Small library API used to demonstrate that source is packaged.
    │
    ├── math.sev
    │   # Secondary module showing normal multi-file package structure.
    │
    └── main.sev
        # Executable entry point.
        # Imports package code and prints a deterministic result.
```

After:

```bash
sev build
```

the build system produces something conceptually like:

```text
target/
└── package.pkg
```

`package.pkg` is an archive/container owned by Severian. Its unpacked layout should be:

```text
package.pkg/
│
├── src/
│   │
│   ├── lib.sev
│   │   # Original Severian source.
│   │
│   ├── math.sev
│   │   # Original Severian source.
│   │
│   └── main.sev
│       # Original executable entry point.
│
│
├── bin/
│   │
│   ├── x86_64-linux-gnu/
│   │   └── 05-building
│   │       # Ready-to-run native executable for this target.
│   │
│   └── aarch64-linux-gnu/
│       └── 05-building
│           # Optional executable produced for another target.
│
│
├── artifacts/
│   │
│   ├── mir/
│   │   └── 05-building.mir
│   │       # Optional Severian MIR retained for inspection/recompilation.
│   │
│   ├── mlir/
│   │   └── 05-building.mlir
│   │       # MLIR generated during compilation.
│   │
│   ├── llvm/
│   │   └── 05-building.bc
│   │       # Optional LLVM bitcode/native backend artifact.
│   │
│   └── stablehlo/
│       └── model.mlir
│           # Placeholder for packages containing StableHLO/XLA workloads.
│           # This file need not exist for ordinary CPU applications.
│
│
├── debug/
│   │
│   ├── symbols/
│   │   └── 05-building.debug
│   │       # Debug information separated from the production executable.
│   │
│   ├── tests/
│   │   └── results.toml
│   │       # Results of package tests executed during the build.
│   │
│   ├── coverage/
│   │   └── coverage.toml
│   │       # Source/test coverage data.
│   │
│   └── profile/
│       └── build.toml
│           # Compilation/runtime profiling information when requested.
│
│
├── metadata/
│   │
│   ├── package.toml
│   │   # Frozen copy of the package manifest used for this build.
│   │
│   ├── package.lock
│   │   # Exact dependency resolution used for this build.
│   │
│   ├── build.toml
│   │   # Describes this particular build.
│   │   #
│   │   # Compiler version
│   │   # build profile
│   │   # optimization level
│   │   # timestamp
│   │   # source hash
│   │   # reproducibility information
│   │
│   ├── targets.toml
│   │   # Enumerates executable/artifact targets contained in the package.
│   │   #
│   │   # Example:
│   │   # x86_64-linux-gnu
│   │   # aarch64-linux-gnu
│   │
│   ├── hardware.toml
│   │   # Hardware requirements rather than build-machine identity.
│   │   #
│   │   # CPU architecture/features
│   │   # GPU vendor/family
│   │   # required accelerators
│   │   # minimum memory
│   │   # driver/runtime requirements
│   │
│   ├── runtime.toml
│   │   # Runtime requirements.
│   │   #
│   │   # libc requirements
│   │   # Severian runtime ABI
│   │   # PJRT/runtime providers
│   │   # system libraries
│   │
│   ├── artifacts.toml
│   │   # Index of artifacts contained in package.pkg.
│   │   #
│   │   # Connects:
│   │   #   artifact
│   │   #   target
│   │   #   backend
│   │   #   entry point
│   │   #   requirements
│   │
│   ├── checksums.toml
│   │   # Cryptographic hash of every important package object.
│   │
│   └── provenance.toml
│       # Records where the package came from and how it was built.
│       #
│       # Source revision
│       # compiler identity
│       # dependency resolution
│       # build-tool identity
│       # optional signature information
│
│
└── container/
    │
    ├── container.toml
    │   # Severian container policy.
    │   #
    │   # Containers are a fallback execution mechanism.
    │   # This describes whether one may be constructed/used.
    │
    ├── runtime.toml
    │   # Host/container boundary.
    │   #
    │   # Network access
    │   # filesystem mounts
    │   # writable directories
    │   # environment
    │   # CPU/memory limits
    │   # GPU/device access
    │   # host-provided drivers
    │
    └── oci/
        │
        ├── oci-layout
        │   # Standard OCI image-layout marker.
        │
        ├── index.json
        │   # Standard OCI image index.
        │   # Can reference multiple architecture manifests.
        │
        └── blobs/
            └── sha256/
                └── ...
                    # OCI manifests/config/layers.
                    #
                    # Absent unless the package was actually built
                    # with a pre-materialized container fallback.
```

I would keep `container/` present conceptually but not require an OCI filesystem image in every package. Three useful states are:

```text
[container]
mode = "none"
```

No container support needed.

```text
[container]
mode = "recipe"
```

Metadata exists so Severian can construct a fallback container.

```text
[container]
mode = "embedded"
```

`container/oci/` contains a ready OCI image.

The resolver then has a defined order:

```text
sev run package.pkg

1. Find compatible bin/<target>/ executable
               │
               ↓ none
2. Find compatible artifacts/ representation
               │
               ↓ none
3. Rebuild from src/ using package.lock
               │
               ↓ cannot build
4. Use/build container fallback
               │
               ↓ unavailable
5. Report exact compatibility failure
```

That gives a clean conceptual boundary:

```text
package.toml    = desired package
package.lock    = resolved dependencies
src/            = authoritative program

package.pkg     = distributable realization

bin/            = immediately runnable forms
artifacts/      = compiler/backend forms
debug/          = development evidence
metadata/       = resolution/reproducibility information
container/      = portability fallback
src/            = rebuild fallback
```



General goals:
1. Package building, testing, installing, and publishing should be easy and understandable
2. Packages should be safe and configurable. Don't force everyone through a rule for a minor benefit to yourself and a detriment to all
3. Packages are portable and installing/using them should be hassle free if sev install x then sev build should always produce the program the user wants even at the expense of security while still putting security as a strong desire.

.pkg = consumable unit
target = thing the package can produce
interface = what another package may use
source = optional implementation disclosure

source                    compiled distribution

src/
├── lib.sev         ─┐
├── file.sev         │
└── main.sev         │
                     ▼
                  file.pkg
                  ├── manifest
                  ├── interface
                  ├── implementations
                  ├── artifacts
                  └── targets

file.pkg
├── package
│   name = file
│   version = 1.4.0
│
├── interface
│   File
│   File.read(path: string) -> string
│
├── implementations
│   LuaFile: File
│   JsonFile: File
│
├── targets
│   lib
│   bin:file
|   debug/ (testing, temp objects, code coverage etc useful for development)
│
└── artifacts
    native-x86_64
    native-aarch64
    xla
    ...

Later:
server.pkg

provides:
    library:
        Server
        Server.start(...)
        Server.stop(...)

    commands:
        serve(port: int = 8080)
        migrate()
        status()

requires:
    network
    filesystem:/data

artifacts:
    native linux/x86_64
    native linux/aarch64
    OCI linux/x86_64

does-not-provide:
    windows native

Severian package semantics
        ↓
.pkg
        ↓
artifact selection
   ↙      ↓       ↘
native    VM       OCI
                   ↓
           Docker / Podman /
           containerd / etc.


Yes. I would make `network/` a first-class section of `package.pkg`, separate from the Severian `network` library.

Its job is not implementing sockets. Its job is declaring the package's communication contract so `sev run`, containers, VMs, orchestration, and deployment tooling know what the process needs.

```text
package.pkg/
├── src/
├── bin/
├── artifacts/
├── debug/
├── metadata/
├── network/
└── container/
```

I would give it this structure:

```text
network/
├── network.toml
│   # Overall networking policy.
│   #
│   # offline / client / server / peer
│   # whether networking is required
│   # default deny/allow behavior
│   # startup behavior if network unavailable
│
├── endpoints.toml
│   # Named services this package communicates with.
│   #
│   # Example:
│   # model-store
│   # postgres
│   # telemetry
│   # coordinator
│
├── ingress.toml
│   # Connections allowed INTO the application.
│   #
│   # protocols
│   # ports
│   # interfaces
│   # public/private exposure
│
├── egress.toml
│   # Connections the application needs to make.
│   #
│   # destinations
│   # protocols
│   # ports
│   # DNS requirements
│
├── dns.toml
│   # Name-resolution requirements and policy.
│   #
│   # DNS needed?
│   # search domains
│   # service discovery
│   # caching behavior
│
├── tls.toml
│   # Transport security requirements.
│   #
│   # certificate requirements
│   # trust stores
│   # minimum TLS version
│   # mTLS requirements
│
├── proxy.toml
│   # Proxy behavior.
│   #
│   # inherit system proxy
│   # explicit proxy support
│   # no-proxy destinations
│
├── resilience.toml
│   # What happens when communication fails.
│   #
│   # connect timeout
│   # request timeout
│   # idle timeout
│   # retry limits
│   # backoff
│   # circuit breaking
│   # reconnect behavior
│
└── resources.toml
    # Network resource limits.
    #
    # max sockets
    # connection pools
    # max idle connections
    # buffers
    # concurrent requests
```

For example:

```toml
# network/network.toml

[network]
mode = "client"
required = true
default = "deny"

[startup]
network_required = false
degraded_mode = true
```

Then explicitly name dependencies instead of just saying "needs internet":

```toml
# network/endpoints.toml

[[endpoint]]
name = "model-store"
protocol = "https"
host = "models.example.com"
port = 443
required = true

[[endpoint]]
name = "telemetry"
protocol = "https"
host = "telemetry.example.com"
port = 443
required = false
```

Egress:

```toml
# network/egress.toml

[[allow]]
endpoint = "model-store"

[[allow]]
endpoint = "telemetry"
```

Ingress for a service:

```toml
# network/ingress.toml

[[listen]]
name = "api"
protocol = "tcp"
port = 8080
interface = "any"
required = true

[[listen]]
name = "health"
protocol = "tcp"
port = 8081
interface = "local"
```

And the part people routinely omit:

```toml
# network/resilience.toml

[connect]
timeout = "5s"

[request]
timeout = "30s"

[idle]
timeout = "60s"

[retry]
attempts = 3
backoff = "exponential"
maximum = "10s"

[pool]
maximum_connections = 128
maximum_idle = 32
idle_timeout = "30s"
```

This becomes valuable because `sev` can reason about it before launching anything.

```text
sev run package.pkg
        │
        ├── Does it require networking?
        ├── Which endpoints?
        ├── Does DNS work?
        ├── Are required ports available?
        ├── Does container policy permit egress?
        ├── Does it need inbound ports exposed?
        ├── Are TLS/runtime requirements available?
        └── What happens when communication fails?
```

Then container generation is mechanical rather than guesswork:

```text
package.pkg/network/
        ↓
Severian execution policy
        ↓
container network namespace
firewall/egress rules
port publishing
service discovery
DNS
proxy environment
health checking
```

The distinction I would enforce is:

```text
library/network/
    = how Severian programs perform networking

package.pkg/network/
    = what this particular program requires from its environment
```

And I would put `network/` alongside `container/`, not inside it. A native process, container, VM, remote job, GPU worker, or cluster task all have networking requirements. Containers are only one execution environment.

This also gives you a strong eventual `sev doctor package.pkg` story:

```text
Network
  DNS                 ok
  model-store:443     reachable
  telemetry:443       unavailable (optional)
  ingress :8080       available
  ingress :8081       available
  TLS trust store     ok
  proxy               inherited
  socket limit        4096 / required >= 128
```

That is worth designing into `.pkg` now. Network configuration is part of whether software can actually run, just like architecture, libraries, drivers, and memory.
