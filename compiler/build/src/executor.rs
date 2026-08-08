use crate::{
    cache::{BuildCache, CacheStatus},
    fingerprint::{fingerprint_node, Fingerprint},
    graph::{BuildGraph, GraphError},
    node::{BuildNode, BuildNodeId},
};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BuildContext {
    pub compiler_version: String,
    pub target: String,
    pub profile: String,
    pub flags: Vec<String>,
}

#[derive(Debug)]
pub struct BuildFailure {
    pub node: BuildNodeId,
    pub message: String,
}

#[derive(Debug, Default)]
pub struct BuildOutcome {
    pub fresh: Vec<BuildNodeId>,
    pub built: Vec<BuildNodeId>,
    pub fingerprints: HashMap<BuildNodeId, Fingerprint>,
}

pub trait BuildExecutor {
    fn execute(&self, node: &BuildNode, context: &BuildContext) -> Result<(), String>;
}

pub struct BuildRunner<E> {
    pub graph: BuildGraph,
    pub cache: BuildCache,
    pub executor: E,
}

impl<E: BuildExecutor> BuildRunner<E> {
    pub fn run(&mut self, context: &BuildContext) -> Result<BuildOutcome, BuildFailure> {
        let order = self.graph.topological_order().map_err(|error| BuildFailure {
            node: BuildNodeId(usize::MAX),
            message: error.to_string(),
        })?;

        let mut outcome = BuildOutcome::default();

        for id in order {
            let node = self.graph.node(id).expect("topological order contains valid nodes");
            let dependencies = node
                .dependencies
                .iter()
                .filter_map(|dependency| outcome.fingerprints.get(dependency).copied())
                .collect::<Vec<_>>();

            let fingerprint = fingerprint_node(
                node,
                &context.compiler_version,
                &context.target,
                &context.profile,
                &context.flags,
                &dependencies,
            )
            .map_err(|error| BuildFailure {
                node: id,
                message: error.to_string(),
            })?;

            let status = self.cache.status(node, fingerprint).map_err(|error| BuildFailure {
                node: id,
                message: error.to_string(),
            })?;

            if status == CacheStatus::Fresh {
                outcome.fresh.push(id);
                outcome.fingerprints.insert(id, fingerprint);
                continue;
            }

            self.executor.execute(node, context).map_err(|message| BuildFailure {
                node: id,
                message,
            })?;

            self.cache.commit(node, fingerprint).map_err(|error| BuildFailure {
                node: id,
                message: error.to_string(),
            })?;

            outcome.built.push(id);
            outcome.fingerprints.insert(id, fingerprint);
        }

        Ok(outcome)
    }
}
