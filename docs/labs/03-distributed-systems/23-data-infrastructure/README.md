# Data-to-infrastructure workflow

This executable constructs a desired workload, schedules replicas, reconciles
observed state, and prints the PQL-validated SQL used for controller history.
The application stays in Severian; native host discovery remains below the
standard-library `platform` boundary.
