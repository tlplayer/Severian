# log

Structured logging with explicit sinks and levels. Formatting and filtering
belong in Severian source; clocks and output sinks are platform capabilities.
The `info` and `error` sinks are native compile-link-execute tested. A
configurable default logger and concurrent ordering policy remain future work.
