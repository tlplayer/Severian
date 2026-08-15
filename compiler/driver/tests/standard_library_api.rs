use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn requested_standard_library_surface_executes_natively() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-standard-library-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("main.sev");
    std::fs::write(
        &source,
        r#"import core
import list
import collections
import map
import set
import string
import math
import random
import path
import file
import json
import regex
import time
import process
import environment
import tensor

def main():
    numbers := [3, 1, 2]
    list.sort(numbers)
    assert(numbers == [1, 2, 3])
    assert(list.binary_search(numbers, 2) == 1)
    assert(list.chunked(numbers, 2) == [[1, 2], [3]])
    assert(list.windowed(numbers, 2) == [[1, 2], [2, 3]])
    assert(list.flatten([[1], [2, 3]]) == numbers)
    heap = collections.MinHeap[int]([])
    heap.push(3)
    heap.push(1)
    minimum = heap.pop()
    assert(minimum == 1)
    queue = collections.Queue[int](collections.Deque[int]([]))
    queue.push(7)
    queued = queue.pop()
    assert(queued == 7)
    cache_values: map[string, int] = {}
    cache = collections.LruCache[string, int](1, cache_values, [])
    cache.put("answer", 42)
    cached = cache.get("answer")
    assert(cached == 42)
    assert(len(numbers) == 3)
    assert(size(numbers) == 3)
    assert(numbers.len() == 3)
    assert(numbers.size() == 3)
    assert(bits(numbers) == bytes(numbers) * 8)
    assert(numbers.bits() == numbers.bytes() * 8)
    assert(capacity(numbers) >= len(numbers))
    assert(core.sum(numbers) == 6)
    assert(string.strip("  Severian  ") == "Severian")
    assert("severian".starts_with("sev"))
    assert("severian".ends_with("ian"))
    assert(numbers.to_set() == {1, 2, 3})
    assert(math.floor(sqrt(9.0)) == 3)
    random.seed(7)
    assert(random.randint(1, 1) == 1)
    mapping := {"answer": 42}
    assert(map.get(mapping, "answer", 0) == 42)
    assert(mapping.set_default("missing", 7) == 7)
    entries := {1, 2}
    set.add(entries, 3)
    assert(3 in entries)
    assert(entries.symmetric_difference({2, 4}) == {1, 3, 4})
    assert(regex.search("[0-9]+", "release-42"))
    assert(path.basename("/tmp/example.txt") == "example.txt")
    assert(environment.set("SEVERIAN_LIBRARY_TEST", "ready"))
    assert(environment.get("SEVERIAN_LIBRARY_TEST") == "ready")
    assert(environment.remove("SEVERIAN_LIBRARY_TEST"))
    json_path = "/tmp/severian-standard-library-api.json"
    _written = file.write(json_path, "{\"answer\":42}")
    loaded_json = file.read("/tmp/severian-standard-library-api.json")
    assert(loaded_json.kind() == "json")
    assert(loaded_json.get("answer") == 42)
    assert(json.dumps([1, 2, 3]) == "[1,2,3]")
    value = tensor.ones([2, 2])
    assert(len(value) == 4)
    assert(value.size() == 4)
    assert(value.bytes() == 32)
    assert(value.bits() == 256)
    assert(tensor.mean(value) == 1.0)
    assert(time.monotonic() > 0.0)
    assert(process.run("true") == 0)
"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn implicit_receivers_execute_across_a_package_boundary() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-implicit-receiver-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("main.sev");
    std::fs::write(
        &source,
        r#"import io

def main():
    source = io.MemoryStream([1, 2, 3])
    destination = io.MemoryStream([])
    switch io.copy(source, destination, 2):
        ok count:
            assert(count == 3)
            assert(destination.snapshot() == [1, 2, 3])
        failure _error:
            assert(false)
    limited = io.LimitedWriter(io.MemoryStream([]), 1)
    switch io.copy(io.MemoryStream([1, 2]), limited, 2):
        ok _count:
            assert(false)
        failure error:
            assert(error.message == "writer accepted only part of a stream chunk")
"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn file_read_dispatches_formats_and_accepts_trait_decoders() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-file-formats-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let text = root.join("note.txt");
    let csv = root.join("people.csv");
    let mp3 = root.join("sound.mp3");
    let wav = root.join("sound.wav");
    let playlist = root.join("music.m3u");
    std::fs::write(&text, "hello").unwrap();
    std::fs::write(&csv, "name,note\nAda,\"compiler, author\"\n").unwrap();
    std::fs::write(&mp3, b"ID3payload").unwrap();
    let mut wav_header = vec![0_u8; 44];
    wav_header[0..4].copy_from_slice(b"RIFF");
    wav_header[8..12].copy_from_slice(b"WAVE");
    wav_header[22..24].copy_from_slice(&2_u16.to_le_bytes());
    wav_header[24..28].copy_from_slice(&48_000_u32.to_le_bytes());
    wav_header[34..36].copy_from_slice(&16_u16.to_le_bytes());
    std::fs::write(&wav, wav_header).unwrap();
    std::fs::write(&playlist, "one.mp3\ntwo.mp3").unwrap();

    let source = root.join("main.sev");
    std::fs::write(
        &source,
        format!(
            r#"import file
import platform

class Playlist: file.File
    source_path: string
    tracks: list[string]

    def path() -> string:
        return source_path

    def name() -> string:
        return platform.path_basename(source_path)

    def extension() -> string:
        return ".m3u"

    def kind() -> string:
        return "playlist"

    def media_type() -> string:
        return "audio/x-playlist"

    def size() -> int:
        return size(platform.string_encode(tracks.join("\n")))

    def bytes() -> int:
        return size(platform.string_encode(tracks.join("\n")))

    def raw() -> list[int]:
        return platform.string_encode(tracks.join("\n"))

    def exists() -> bool:
        return platform.file_exists(source_path)

    def text() -> Result[string, file.FormatError]:
        return tracks.join("\n")

    def value() -> Any:
        return tracks

    def write() -> Result[unit, IOError]:
        return platform.file_write(source_path, tracks.join("\n"))

class PlaylistReader: file.Reader
    def extensions() -> list[string]:
        return [".m3u"]

    def read(path: string) -> Result[file.File, IOError | file.FormatError]:
        content = platform.file_read(path)
        return Playlist(path, content.split("\n"))

def main():
    file.register("m3u", PlaylistReader())
    switch file.read("{}"):
        ok document:
            assert(document.kind() == "text")
            assert(document.extension() == ".txt")
            assert(document.bytes() == 5)
            assert(document.content == "hello")
        failure error:
            assert(false, error)
    switch file.read("{}"):
        ok document:
            assert(document.kind() == "csv")
            assert(document.records() == [["Ada", "compiler, author"]])
        failure error:
            assert(false, error)
    switch file.read("{}"):
        ok document:
            assert(document.kind() == "mp3")
            assert(document.extension() == ".mp3")
            assert(document.has_id3)
        failure error:
            assert(false, error)
    switch file.read("{}"):
        ok document:
            assert(document.kind() == "wav")
            assert(document.bytes() == 44)
            assert(document.channels == 2)
            assert(document.sample_rate == 48000)
            assert(document.bits_per_sample == 16)
        failure error:
            assert(false, error)
    switch file.read("{}"):
        ok document:
            assert(document.kind() == "playlist")
            assert(document.bytes() == 15)
            assert(document.get("tracks") == ["one.mp3", "two.mp3"])
        failure error:
            assert(false, error)
"#,
            text.display(),
            csv.display(),
            mp3.display(),
            wav.display(),
            playlist.display(),
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn all_requested_packages_are_workspace_members() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library/package.toml");
    let manifest = std::fs::read_to_string(workspace).unwrap();
    for package in [
        "core",
        "list",
        "collections",
        "map",
        "set",
        "string",
        "math",
        "random",
        "file",
        "path",
        "json",
        "regex",
        "time",
        "process",
        "environment",
        "http",
        "network",
        "logging",
        "io",
        "tensor",
    ] {
        assert!(
            manifest.contains(&format!("\"{package}\"")),
            "missing {package}"
        );
    }
}

#[test]
fn nested_official_package_needs_no_manifest_dependency() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-nested-standard-library-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"stdlib-consumer\"\nversion = \"0.1.0\"\nedition = \"2026\"\n\n[[bin]]\nname = \"stdlib-consumer\"\npath = \"src/main.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/main.sev"),
        "import model.speech as speech\n\ndef main():\n    assert(speech.estimate_audio_tokens(\"hello\", \"reference\", 25, 1.0) > 0)\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(root.join("sev.lock")).unwrap(),
        "version = 1\n"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn declared_dependency_cannot_shadow_official_package() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-standard-library-shadow-{}-{nonce}",
        std::process::id()
    ));
    let application = root.join("application");
    let impostor = root.join("impostor");
    std::fs::create_dir_all(application.join("src")).unwrap();
    std::fs::create_dir_all(impostor.join("src")).unwrap();
    std::fs::write(
        impostor.join("package.toml"),
        "[package]\nname = \"file\"\nversion = \"0.1.0\"\nedition = \"2026\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        impostor.join("src/lib.sev"),
        "def impostor() -> bool:\n    return true\n",
    )
    .unwrap();
    std::fs::write(
        application.join("package.toml"),
        "[package]\nname = \"shadow-consumer\"\nversion = \"0.1.0\"\nedition = \"2026\"\n\n[[bin]]\nname = \"shadow-consumer\"\npath = \"src/main.sev\"\n\n[dependencies]\nfile = { path = \"../impostor\", version = \"0.1.0\" }\n",
    )
    .unwrap();
    std::fs::write(
        application.join("src/main.sev"),
        "import file\n\ndef main():\n    print(file.impostor())\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&application)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("cannot shadow the reserved Severian standard-library package"),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(root).unwrap();
}
