# vm

Typed virtual-machine specifications and deterministic launch plans. Image,
vCPU, memory, filesystem, virtual network, process, and storage settings are
validated in Severian. Host page-size discovery crosses the explicit
`platform` ABI.

This layer builds plans; privileged machine creation belongs to `hypervisor`.
