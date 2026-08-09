//! PJRT-backed default replica/partition device assignment.

use super::{api, compile::RawClient, error};
use crate::{options::DeviceAssignment, Result};

impl RawClient {
    pub fn default_device_assignment(
        &self,
        replicas: usize,
        partitions: usize,
    ) -> Result<DeviceAssignment> {
        if replicas == 0 || partitions == 0 {
            return Err(crate::XlaError::Pjrt(
                "replicas and partitions must both be greater than zero".into(),
            ));
        }

        let replicas_i32 = i32::try_from(replicas)
            .map_err(|_| crate::XlaError::Pjrt("replica count exceeds PJRT int range".into()))?;
        let partitions_i32 = i32::try_from(partitions)
            .map_err(|_| crate::XlaError::Pjrt("partition count exceeds PJRT int range".into()))?;
        let count = replicas.checked_mul(partitions)
            .ok_or_else(|| crate::XlaError::Pjrt("device assignment size overflow".into()))?;
        let mut flat = vec![-1i32; count];

        let api = self.plugin().api();
        let mut args = api::PJRT_Client_DefaultDeviceAssignment_Args {
            struct_size: api::struct_size::<api::PJRT_Client_DefaultDeviceAssignment_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            num_replicas: replicas_i32,
            num_partitions: partitions_i32,
            default_assignment_size: flat.len(),
            default_assignment: flat.as_mut_ptr(),
        };

        let result = unsafe { (api.PJRT_Client_DefaultDeviceAssignment)(&mut args) };
        unsafe { error::check(api, result)? };

        let mut by_partition = vec![vec![0i64; replicas]; partitions];
        for replica in 0..replicas {
            for partition in 0..partitions {
                let index = replica * partitions + partition;
                by_partition[partition][replica] = i64::from(flat[index]);
            }
        }

        DeviceAssignment::new(replicas, partitions, by_partition)
            .map_err(crate::XlaError::Pjrt)
    }
}
