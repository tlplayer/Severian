use std::collections::BTreeMap;
use std::sync::Arc;

/// Lexical bindings remain in their declaring scope. Child lookup follows
/// parent edges instead of copying all visible names into every block.
#[derive(Clone, Debug)]
pub(super) struct LexicalScope<V: Clone> {
    bindings: BTreeMap<String, V>,
    parent: Option<Arc<Self>>,
}

impl<V: Clone> Default for LexicalScope<V> {
    fn default() -> Self {
        Self { bindings: BTreeMap::new(), parent: None }
    }
}

impl<V: Clone> LexicalScope<V> {
    pub fn child(&self) -> Self {
        Self { bindings: BTreeMap::new(), parent: Some(Arc::new(self.clone())) }
    }

    pub fn get(&self, name: &str) -> Option<&V> {
        self.bindings.get(name).or_else(|| self.parent.as_ref()?.get(name))
    }

    pub fn contains_key(&self, name: &str) -> bool { self.get(name).is_some() }

    pub fn insert(&mut self, name: String, value: V) -> Option<V> {
        self.bindings.insert(name, value)
    }

    pub fn remove(&mut self, name: &str) -> Option<V> {
        self.bindings.remove(name).or_else(|| {
            self.parent.as_mut().and_then(|parent| Arc::make_mut(parent).remove(name))
        })
    }

    pub fn clear(&mut self) { *self = Self::default(); }
}

impl<V: Clone> std::ops::Index<&String> for LexicalScope<V> {
    type Output = V;
    fn index(&self, name: &String) -> &V { self.get(name).expect("resolved lexical binding") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_walks_parents_and_child_declarations_do_not_escape() {
        let mut module = LexicalScope::default();
        module.insert("x".into(), 1);
        let mut outer = module.child();
        outer.insert("y".into(), 2);
        let inner = outer.child();
        assert_eq!(inner.get("x"), Some(&1));
        assert_eq!(inner.get("y"), Some(&2));
        assert_eq!(module.get("y"), None);
    }
}
