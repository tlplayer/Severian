# Container

`container` is the ergonomic lifecycle layer over `oci.Bundle`. It calls a
standards-compliant runtime instead of reimplementing a Linux container
runtime. The component manifest selects installed `crun` first and `runc`
second; applications do not carry a runtime-selection flag.
