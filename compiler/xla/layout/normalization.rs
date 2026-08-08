#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Layout {
    /// XLA-style ordering from most-minor to most-major logical dimension.
    pub minor_to_major: Vec<u8>,
}

impl Layout {
    pub fn rank(&self) -> usize {
        self.minor_to_major.len()
    }

    pub fn is_valid(&self) -> bool {
        let rank = self.rank();
        let mut seen = vec![false; rank];

        for &axis in &self.minor_to_major {
            let axis = axis as usize;
            if axis >= rank || seen[axis] {
                return false;
            }
            seen[axis] = true;
        }

        seen.into_iter().all(|value| value)
    }
}

pub fn normalize_layout(layout: &Layout, rank: usize) -> Layout {
    if rank == 0 {
        return Layout::default();
    }

    let mut result = Vec::with_capacity(rank);
    let mut seen = vec![false; rank];

    for &axis in &layout.minor_to_major {
        let axis = axis as usize;
        if axis < rank && !seen[axis] {
            seen[axis] = true;
            result.push(axis as u8);
        }
    }

    for axis in (0..rank).rev() {
        if !seen[axis] {
            result.push(axis as u8);
        }
    }

    Layout {
        minor_to_major: result,
    }
}
