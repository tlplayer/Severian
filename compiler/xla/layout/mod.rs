mod assignment;
mod normalization;

pub use assignment::{assign_function_layouts, LayoutAssignment, LayoutPlan};
pub use normalization::{normalize_layout, Layout};

use severian_hir::TensorType;

pub fn default_layout(tensor: TensorType) -> Option<Layout> {
    let rank = tensor.rank? as usize;
    let minor_to_major = (0..rank).rev().map(|axis| axis as u8).collect();

    Some(Layout { minor_to_major })
}
