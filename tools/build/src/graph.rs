use crate::node::{BuildNode, BuildNodeId};
use std::collections::{BTreeSet, HashMap, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    MissingNode(BuildNodeId),
    Cycle(Vec<BuildNodeId>),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingNode(id) => {
                write!(formatter, "build graph references missing node {}", id.0)
            }
            Self::Cycle(nodes) => write!(
                formatter,
                "build graph contains a cycle involving {}",
                nodes
                    .iter()
                    .map(|id| id.0.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl std::error::Error for GraphError {}

#[derive(Debug, Clone, Default)]
pub struct BuildGraph {
    nodes: Vec<BuildNode>,
}

impl BuildGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, mut node: BuildNode) -> BuildNodeId {
        let id = BuildNodeId(self.nodes.len());
        node.id = id;
        self.nodes.push(node);
        id
    }

    pub fn node(&self, id: BuildNodeId) -> Option<&BuildNode> {
        self.nodes.get(id.0)
    }

    pub fn node_mut(&mut self, id: BuildNodeId) -> Option<&mut BuildNode> {
        self.nodes.get_mut(id.0)
    }

    pub fn nodes(&self) -> &[BuildNode] {
        &self.nodes
    }

    pub fn add_dependency(
        &mut self,
        node: BuildNodeId,
        dependency: BuildNodeId,
    ) -> Result<(), GraphError> {
        if self.node(dependency).is_none() {
            return Err(GraphError::MissingNode(dependency));
        }
        let node_ref = self.node_mut(node).ok_or(GraphError::MissingNode(node))?;
        if !node_ref.dependencies.contains(&dependency) {
            node_ref.dependencies.push(dependency);
        }
        Ok(())
    }

    pub fn topological_order(&self) -> Result<Vec<BuildNodeId>, GraphError> {
        let mut indegree = vec![0usize; self.nodes.len()];
        let mut dependents = vec![Vec::<BuildNodeId>::new(); self.nodes.len()];

        for node in &self.nodes {
            for dependency in &node.dependencies {
                if self.node(*dependency).is_none() {
                    return Err(GraphError::MissingNode(*dependency));
                }
                indegree[node.id.0] += 1;
                dependents[dependency.0].push(node.id);
            }
        }

        let mut ready = BTreeSet::new();
        for (index, count) in indegree.iter().enumerate() {
            if *count == 0 {
                ready.insert(BuildNodeId(index));
            }
        }

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = ready.pop_first() {
            order.push(id);
            for dependent in &dependents[id.0] {
                indegree[dependent.0] -= 1;
                if indegree[dependent.0] == 0 {
                    ready.insert(*dependent);
                }
            }
        }

        if order.len() != self.nodes.len() {
            let cycle = indegree
                .iter()
                .enumerate()
                .filter_map(|(index, count)| (*count > 0).then_some(BuildNodeId(index)))
                .collect();
            return Err(GraphError::Cycle(cycle));
        }

        Ok(order)
    }

    pub fn transitive_dependents(
        &self,
        roots: &[BuildNodeId],
    ) -> Result<Vec<BuildNodeId>, GraphError> {
        let mut reverse = HashMap::<BuildNodeId, Vec<BuildNodeId>>::new();
        for node in &self.nodes {
            for dependency in &node.dependencies {
                if self.node(*dependency).is_none() {
                    return Err(GraphError::MissingNode(*dependency));
                }
                reverse.entry(*dependency).or_default().push(node.id);
            }
        }

        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from(roots.to_vec());

        while let Some(current) = queue.pop_front() {
            if !seen.insert(current) {
                continue;
            }
            if let Some(next) = reverse.get(&current) {
                queue.extend(next.iter().copied());
            }
        }

        Ok(seen.into_iter().collect())
    }
}
