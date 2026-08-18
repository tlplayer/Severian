Direction or idea of how packages should work

General goals:
1. Package building, testing, installing, and publishing should be easy and understandable
2. Packages should be safe and configurable. Don't force everyone through a rule for a minor benefit to yourself and a detriment to all
3. Packages are portable and installing/using them should be hassle free if sev install x then sev build should always produce the program the user wants even at the expense of security while still putting security as a strong desire.

.pkg = consumable unit
target = thing the package can produce
interface = what another package may use
source = optional implementation disclosure

source                    compiled distribution

src/
├── lib.sev         ─┐
├── file.sev         │
└── main.sev         │
                     ▼
                  file.pkg
                  ├── manifest
                  ├── interface
                  ├── implementations
                  ├── artifacts
                  └── targets

file.pkg
├── package
│   name = file
│   version = 1.4.0
│
├── interface
│   File
│   File.read(path: string) -> string
│
├── implementations
│   LuaFile: File
│   JsonFile: File
│
├── targets
│   lib
│   bin:file
|   debug/ (testing, temp objects, code coverage etc useful for development)
│
└── artifacts
    native-x86_64
    native-aarch64
    xla
    ...
