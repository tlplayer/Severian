# log

Structured logging with explicit sinks and levels. Formatting and filtering
belong in Severian source; clocks and output sinks are runtime capabilities.
The initial `info` and `error` declarations type-check current examples. Their
runtime sinks, default logger, and concurrent ordering policy remain pending.
