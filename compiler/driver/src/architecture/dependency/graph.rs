use super::ArchitectureDependency;
use std::collections::HashSet;

pub(super) fn adjacency(node_count: usize, edges: &[ArchitectureDependency]) -> Vec<Vec<usize>> {
    let mut output = vec![Vec::new(); node_count];
    for edge in edges {
        output[edge.source].push(edge.target);
    }
    for targets in &mut output {
        targets.sort_unstable();
        targets.dedup();
    }
    output
}

pub(super) fn strongly_connected_components(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct Tarjan<'a> {
        adjacency: &'a [Vec<usize>],
        next_index: usize,
        indices: Vec<Option<usize>>,
        lowlinks: Vec<usize>,
        stack: Vec<usize>,
        on_stack: Vec<bool>,
        components: Vec<Vec<usize>>,
    }

    impl Tarjan<'_> {
        fn visit(&mut self, node: usize) {
            let index = self.next_index;
            self.next_index += 1;
            self.indices[node] = Some(index);
            self.lowlinks[node] = index;
            self.stack.push(node);
            self.on_stack[node] = true;
            for &target in &self.adjacency[node] {
                if self.indices[target].is_none() {
                    self.visit(target);
                    self.lowlinks[node] = self.lowlinks[node].min(self.lowlinks[target]);
                } else if self.on_stack[target] {
                    self.lowlinks[node] = self.lowlinks[node].min(self.indices[target].unwrap());
                }
            }
            if self.lowlinks[node] != index {
                return;
            }
            let mut component = Vec::new();
            loop {
                let member = self.stack.pop().expect("Tarjan stack contains root");
                self.on_stack[member] = false;
                component.push(member);
                if member == node {
                    break;
                }
            }
            component.sort_unstable();
            self.components.push(component);
        }
    }

    let mut state = Tarjan {
        adjacency,
        next_index: 0,
        indices: vec![None; adjacency.len()],
        lowlinks: vec![0; adjacency.len()],
        stack: Vec::new(),
        on_stack: vec![false; adjacency.len()],
        components: Vec::new(),
    };
    for node in 0..adjacency.len() {
        if state.indices[node].is_none() {
            state.visit(node);
        }
    }
    state.components.sort_by_key(|component| component[0]);
    state.components
}

pub(super) fn cycle_path(component: &[usize], adjacency: &[Vec<usize>]) -> Vec<usize> {
    let members = component.iter().copied().collect::<HashSet<_>>();
    let start = component[0];
    let mut path = vec![start];
    let mut visited = HashSet::from([start]);
    if find_cycle_path(start, start, &members, adjacency, &mut visited, &mut path) {
        path
    } else {
        vec![start, start]
    }
}

fn find_cycle_path(
    node: usize,
    start: usize,
    members: &HashSet<usize>,
    adjacency: &[Vec<usize>],
    visited: &mut HashSet<usize>,
    path: &mut Vec<usize>,
) -> bool {
    for &target in &adjacency[node] {
        if !members.contains(&target) {
            continue;
        }
        if target == start {
            path.push(start);
            return true;
        }
        if visited.insert(target) {
            path.push(target);
            if find_cycle_path(target, start, members, adjacency, visited, path) {
                return true;
            }
            path.pop();
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tarjan_reports_a_real_cycle_path() {
        let adjacency = vec![vec![1], vec![2], vec![0, 3], vec![]];
        let components = strongly_connected_components(&adjacency);
        assert!(components.contains(&vec![0, 1, 2]));
        assert_eq!(cycle_path(&[0, 1, 2], &adjacency), vec![0, 1, 2, 0]);
    }
}
