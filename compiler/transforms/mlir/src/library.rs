#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlirLibrary {
    pub id: &'static str,
    pub abi_version: u32,
    pub pointer_bits: Option<u16>,
    pub module: &'static str,
    pub exports: &'static [&'static str],
    pub dependencies: &'static [&'static str],
}

const STRING_EXPORTS: &[&str] = &[
    "__sev_string_concat",
    "__sev_string_compare",
    "__sev_string_release",
];

const STRING_DEPENDENCIES: &[&str] = &["abort", "free", "malloc", "memcpy", "strcmp", "strlen"];

const STRING_LIBRARY: MlirLibrary = MlirLibrary {
    id: "core.text.string",
    abi_version: 1,
    pointer_bits: Some(64),
    module: include_str!("../../../../library/core/text/mlir/string_v1.mlir"),
    exports: STRING_EXPORTS,
    dependencies: STRING_DEPENDENCIES,
};

pub const fn registered_libraries() -> &'static [MlirLibrary] {
    &[STRING_LIBRARY]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_exports_are_versioned_by_the_library_contract() {
        let library = registered_libraries()[0];
        assert_eq!(library.id, "core.text.string");
        assert_eq!(library.abi_version, 1);
        assert_eq!(library.pointer_bits, Some(64));
        assert!(library.module.contains("severian.abi_version = 1"));
        assert_eq!(library.exports.len(), 3);
    }
}
