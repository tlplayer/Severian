#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceAssignment {
    replicas: usize,
    partitions: usize,
    /// Indexed `[partition][replica]`, matching XLA DeviceAssignmentProto's
    /// repeated ComputationDevice representation.
    devices: Vec<Vec<i64>>,
}

impl DeviceAssignment {
    pub fn new(
        replicas: usize,
        partitions: usize,
        devices: Vec<Vec<i64>>,
    ) -> Result<Self, String> {
        let assignment = Self {
            replicas,
            partitions,
            devices,
        };
        assignment.validate(replicas as i64, partitions as i64)?;
        Ok(assignment)
    }

    pub fn single(device: i64) -> Self {
        Self {
            replicas: 1,
            partitions: 1,
            devices: vec![vec![device]],
        }
    }

    pub fn sequential(replicas: usize, partitions: usize) -> Self {
        let devices = (0..partitions)
            .map(|partition| {
                (0..replicas)
                    .map(|replica| (partition * replicas + replica) as i64)
                    .collect()
            })
            .collect();

        Self {
            replicas,
            partitions,
            devices,
        }
    }

    pub fn replicas(&self) -> usize {
        self.replicas
    }

    pub fn partitions(&self) -> usize {
        self.partitions
    }

    pub fn devices(&self) -> &[Vec<i64>] {
        &self.devices
    }

    pub fn device(&self, replica: usize, partition: usize) -> Option<i64> {
        self.devices
            .get(partition)
            .and_then(|partition_devices| partition_devices.get(replica))
            .copied()
    }

    pub fn validate(
        &self,
        expected_replicas: i64,
        expected_partitions: i64,
    ) -> Result<(), String> {
        let expected_replicas = usize::try_from(expected_replicas)
            .map_err(|_| "replica count cannot be negative".to_string())?;
        let expected_partitions = usize::try_from(expected_partitions)
            .map_err(|_| "partition count cannot be negative".to_string())?;

        if self.replicas != expected_replicas {
            return Err(format!(
                "device assignment has {} replicas but compile options request {}",
                self.replicas, expected_replicas
            ));
        }

        if self.partitions != expected_partitions {
            return Err(format!(
                "device assignment has {} partitions but compile options request {}",
                self.partitions, expected_partitions
            ));
        }

        if self.devices.len() != self.partitions {
            return Err(format!(
                "device assignment contains {} computation/partition rows; expected {}",
                self.devices.len(),
                self.partitions
            ));
        }

        for (partition, devices) in self.devices.iter().enumerate() {
            if devices.len() != self.replicas {
                return Err(format!(
                    "partition {partition} contains {} replica device ids; expected {}",
                    devices.len(),
                    self.replicas
                ));
            }
        }

        Ok(())
    }
}
