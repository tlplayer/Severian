use super::formation::{DispatchId, DispatchPlan};
use severian_hir::Function;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchFusion {
    pub consumer: DispatchId,
    pub producers: Vec<DispatchId>,
}

pub fn fuse_dispatches(_function: &Function, plan: &DispatchPlan) -> Vec<DispatchFusion> {
    let mut result = Vec::new();

    for consumer in &plan.regions {
        let consumer_inputs = consumer.inputs.iter().cloned().collect::<HashSet<_>>();
        let mut producers = Vec::new();

        for producer in &plan.regions {
            if producer.id == consumer.id {
                continue;
            }

            if producer
                .instruction_indices
                .last()
                .zip(consumer.instruction_indices.first())
                .is_some_and(|(producer_index, consumer_index)| producer_index >= consumer_index)
            {
                continue;
            }

            let used_by_consumer = producer
                .outputs
                .iter()
                .any(|output| consumer_inputs.contains(output));

            if !used_by_consumer {
                continue;
            }

            // Do not fuse a region whose output is a public result for multiple
            // unrelated dispatches in this first planner. Multi-use fusion can
            // be added once dispatch regions carry explicit user sets.
            if producer.outputs.len() == 1 {
                producers.push(producer.id);
            }
        }

        if !producers.is_empty() {
            result.push(DispatchFusion {
                consumer: consumer.id,
                producers,
            });
        }
    }

    result
}
