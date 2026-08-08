use super::{BufferAssignment, BufferId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReuseDecision {
    pub assignment_index: usize,
    pub reuse_buffer: BufferId,
}

pub fn apply_reuse(assignments: &[BufferAssignment]) -> Vec<ReuseDecision> {
    let mut decisions = Vec::new();
    let mut active: Vec<(BufferId, usize)> = Vec::new();
    let mut free = Vec::<BufferId>::new();

    for (index, assignment) in assignments.iter().enumerate() {
        let start = assignment.lifetime.definition;

        let mut still_active = Vec::with_capacity(active.len());
        for (buffer, last_use) in active.drain(..) {
            if last_use < start {
                free.push(buffer);
            } else {
                still_active.push((buffer, last_use));
            }
        }
        active = still_active;

        let buffer = free.pop().unwrap_or(assignment.buffer);

        if buffer != assignment.buffer {
            decisions.push(ReuseDecision {
                assignment_index: index,
                reuse_buffer: buffer,
            });
        }

        active.push((buffer, assignment.lifetime.last_use));
    }

    decisions
}
