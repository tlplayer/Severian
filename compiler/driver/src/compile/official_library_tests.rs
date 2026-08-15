use super::{parse_source, EMBEDDED_OFFICIAL_PACKAGES};
use std::path::Path;

#[test]
fn embedded_distribution_contains_nested_official_packages() {
    assert!(EMBEDDED_OFFICIAL_PACKAGES
        .iter()
        .any(|package| package.name == "model.speech"));
    let module = parse_source(
        "import model.speech as speech\n",
        Path::new("embedded-consumer.sev"),
    )
    .unwrap();
    let interfaces =
        severian_package::load_embedded_official_interfaces(&module, EMBEDDED_OFFICIAL_PACKAGES)
            .unwrap();
    let names = interfaces
        .iter()
        .map(|interface| interface.name.as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"model.speech"));
    assert!(names.contains(&"math"));
    assert!(names.contains(&"random"));
}

#[test]
fn embedded_network_package_contains_its_native_provider_assets() {
    let network = EMBEDDED_OFFICIAL_PACKAGES
        .iter()
        .find(|package| package.name == "network")
        .unwrap();
    let paths = network
        .native_assets
        .iter()
        .map(|asset| asset.path)
        .collect::<Vec<_>>();
    assert!(paths.contains(&"native/include/network_abi.h"));
    assert!(paths.contains(&"native/posix/network.c"));
}
