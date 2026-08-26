# Container database lab

This lab is a deliberately small integration target for four independent
systems: source-defined file dispatch, TCP networking, OCI runtime bundles,
and package publication/consumption.

## Run it on the host

From this directory:

```bash
sev run --bin container-database-server -- data
sev run --bin container-database-client -- LIST
sev run --bin container-database-client -- "GET 1"
```

The protocol accepts `LIST`, `GET 1` through `GET 4`, and `QUIT`. Every `GET`
passes through the standard `@file` namespace registry. The lab imports only
`file`; JSON, text, YAML, and YAML-stream providers are contributed by their
library packages and selected by their path predicates.

## Run the server as an OCI container

Build the binaries, then create and start a real OCI runtime bundle:

```bash
sev build --bin container-database-server
sev run --bin container-database-bundle -- \
  "$PWD/target/host/dev/bin/container-database-server" \
  "$PWD/target/oci/container-database" \
  "$PWD/data"
sev run --bin container-database-client -- LIST
```

The development profile uses `crun`, a read-only root filesystem, and
bind-mounted runtime libraries/data. The container service listens internally
on port `18080`; a user-space proxy applies the recorded
`127.0.0.1:8080 -> 127.0.0.1:18080` forwarding rule before the server starts.

Cleanup:

```bash
crun --root /tmp/severian-crun kill severian-container-database TERM
crun --root /tmp/severian-crun delete --force severian-container-database
```

## Publish and consume

The package is intentionally publishable. Once `sev publish` has placed
`container_database@0.1.0` in the default local registry, the unrelated
`consumer/` package resolves it by version alone:

```bash
sev publish .
sev run consumer
```

This lab does not call Docker or Podman and does not claim that a development
runtime bundle is a distributable OCI image. Image-layer construction,
registry transport, isolated network providers, and remote registry auth are
the next explicit pressure points.
