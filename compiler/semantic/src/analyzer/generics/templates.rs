use super::*;

impl Specializer {
    pub(super) fn new(module: &Module, interfaces: &[PackageInterface]) -> Self {
        let mut classes = module
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Class(class) => Some((class.name.name.clone(), class.clone())),
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let mut templates = classes
            .iter()
            .filter(|(_, class)| !class.generic_params.is_empty())
            .map(|(name, class)| {
                (
                    name.clone(),
                    GenericTemplate {
                        identity: name.clone(),
                        class: class.clone(),
                        imports: Vec::new(),
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let mut traits = module
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Trait(declaration) => {
                    Some((declaration.name.name.clone(), declaration.clone()))
                }
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let aliases = collect_imports(module);

        for interface in interfaces {
            let imports = interface
                .module
                .items
                .iter()
                .filter_map(|item| match item {
                    Item::Import(import) => Some(import.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            for item in &interface.module.items {
                match item {
                    Item::Class(class) => {
                        classes
                            .entry(class.name.name.clone())
                            .or_insert_with(|| class.clone());
                        if class.generic_params.is_empty() {
                            continue;
                        }
                        let identity = format!("{}.{}", interface.name, class.name.name);
                        let template = GenericTemplate {
                            identity: identity.clone(),
                            class: class.clone(),
                            imports: imports.clone(),
                        };
                        templates.insert(identity.clone(), template.clone());
                        if let Some(package) = &interface.export_package {
                            templates
                                .insert(format!("{package}.{}", class.name.name), template.clone());
                        }
                        for (exposed, canonical) in &aliases {
                            if canonical == &interface.name {
                                templates.insert(
                                    format!("{exposed}.{}", class.name.name),
                                    template.clone(),
                                );
                            } else if canonical == &identity {
                                templates.insert(exposed.clone(), template.clone());
                            }
                        }
                    }
                    Item::Trait(declaration) => {
                        traits
                            .entry(declaration.name.name.clone())
                            .or_insert_with(|| declaration.clone());
                    }
                    _ => {}
                }
            }
        }

        Self {
            templates,
            classes,
            traits,
            aliases,
            pending: VecDeque::new(),
            scheduled: HashSet::new(),
            required_imports: Vec::new(),
        }
    }
}
