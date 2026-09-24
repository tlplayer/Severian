//! Shared graph mechanics. Callers own discovery, facts, and diagnostics.
#![forbid(unsafe_code)]
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Requirement {
    Symbol, Declaration, Signature, Type, Constraint, Layout, Implementation, ConstantValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency<X> {
    pub source: X,
    pub target: X,
    pub requirement: Requirement,
}

pub struct Graph<X> {
    pub nodes: Vec<X>,
    pub edges: Vec<Dependency<X>>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
}

impl<X: Clone + Ord> Graph<X> {
    pub fn new(nodes: Vec<X>, edges: Vec<Dependency<X>>) -> Result<Self, &'static str> {
        let positions: BTreeMap<_, _> = nodes.iter().cloned().enumerate().map(|(i, x)| (x, i)).collect();
        if positions.len() != nodes.len() { return Err("duplicate graph node"); }
        let mut outgoing = vec![Vec::new(); nodes.len()];
        let mut incoming = vec![Vec::new(); nodes.len()];
        for edge in &edges {
            let source = *positions.get(&edge.source).ok_or("unknown dependency source")?;
            let target = *positions.get(&edge.target).ok_or("unknown dependency target")?;
            outgoing[source].push(target);
            incoming[target].push(source);
        }
        Ok(Self { nodes, edges, outgoing, incoming })
    }

    /// Iterative Kosaraju; dependency components precede their consumers.
    pub fn components(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.nodes.len()];
        let mut finished = Vec::new();
        for root in 0..self.nodes.len() {
            if seen[root] { continue; }
            seen[root] = true;
            let mut stack = vec![(root, 0)];
            while let Some((node, edge)) = stack.last_mut() {
                if let Some(&child) = self.outgoing[*node].get(*edge) {
                    *edge += 1;
                    if !seen[child] { seen[child] = true; stack.push((child, 0)); }
                } else {
                    finished.push(*node);
                    stack.pop();
                }
            }
        }
        seen.fill(false);
        let mut components = Vec::new();
        for root in finished.into_iter().rev() {
            if seen[root] { continue; }
            seen[root] = true;
            let mut stack = vec![root];
            let mut members = Vec::new();
            while let Some(node) = stack.pop() {
                members.push(node);
                for &child in &self.incoming[node] {
                    if !seen[child] { seen[child] = true; stack.push(child); }
                }
            }
            members.sort_unstable();
            components.push(members);
        }
        components.reverse();
        components
    }

    /// Condensed DAG edges retain the requirement carried by each source edge.
    pub fn condensation(&self) -> (Vec<Vec<X>>, Vec<Dependency<usize>>) {
        let components = self.components();
        let mut owner = vec![0; self.nodes.len()];
        let positions: BTreeMap<_, _> = self.nodes.iter().enumerate().map(|(i, x)| (x, i)).collect();
        for (unit, members) in components.iter().enumerate() {
            for &member in members { owner[member] = unit; }
        }
        let mut edges = Vec::new();
        for edge in &self.edges {
            let source = owner[positions[&edge.source]];
            let target = owner[positions[&edge.target]];
            let condensed = Dependency { source, target, requirement: edge.requirement };
            if source != target && !edges.contains(&condensed) { edges.push(condensed); }
        }
        (components.into_iter().map(|members| members.into_iter().map(|i| self.nodes[i].clone()).collect()).collect(), edges)
    }
}

/// The owner joins newly discovered facts into its state and returns true only
/// when that state grows. Discovery must finish first; facts form a finite,
/// monotonic domain. A revisit alone never constitutes progress or failure.
pub fn resolve_graph<X: Clone + Ord>(graph: &Graph<X>, mut advance: impl FnMut(&X) -> bool) {
    for members in graph.components() {
        let membership: BTreeSet<_> = members.iter().copied().collect();
        let mut queue: VecDeque<_> = members.into_iter().collect();
        let mut queued = membership.clone();
        while let Some(node) = queue.pop_front() {
            queued.remove(&node);
            if advance(&graph.nodes[node]) {
                if queued.insert(node) { queue.push_back(node); }
                for &consumer in &graph.incoming[node] {
                    if membership.contains(&consumer) && queued.insert(consumer) {
                        queue.push_back(consumer);
                    }
                }
            }
        }
    }
}

/// Owners decide which facts are mandatory; unknown speculative lookups must
/// not be mistaken for semantic errors by the graph engine.
pub fn unresolved<X: Clone>(graph: &Graph<X>, mut known: impl FnMut(&X, Requirement) -> bool) -> Vec<Dependency<X>> {
    graph.edges.iter().filter(|edge| !known(&edge.target, edge.requirement)).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declaration_cycle_converges_and_condenses_before_its_consumer() {
        let graph = Graph::new(vec![0, 1, 2], vec![
            Dependency { source: 0, target: 1, requirement: Requirement::Declaration },
            Dependency { source: 1, target: 0, requirement: Requirement::Declaration },
            Dependency { source: 2, target: 0, requirement: Requirement::Declaration },
        ]).unwrap();
        assert_eq!(graph.condensation().0, vec![vec![0, 1], vec![2]]);
        let mut known = BTreeSet::from([0]);
        resolve_graph(&graph, |node| {
            let ready = graph.edges.iter().filter(|edge| edge.source == *node)
                .all(|edge| known.contains(&edge.target));
            ready && known.insert(*node)
        });
        assert_eq!(known, BTreeSet::from([0, 1, 2]));
        assert!(unresolved(&graph, |node, _| known.contains(node)).is_empty());
    }

    #[test]
    fn unseeded_layout_cycle_stalls_with_requirements_intact() {
        let graph = Graph::new(vec![0, 1], vec![
            Dependency { source: 0, target: 1, requirement: Requirement::Layout },
            Dependency { source: 1, target: 0, requirement: Requirement::Layout },
        ]).unwrap();
        let mut visits = 0;
        resolve_graph(&graph, |_| { visits += 1; false });
        assert_eq!(visits, 2);
        assert_eq!(unresolved(&graph, |_, _| false), graph.edges);
    }
}
