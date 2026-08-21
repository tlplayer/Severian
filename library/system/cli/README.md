# cli

`cli` is Severian's declarative command-line interface library. It keeps an
application's command schema separate from process state and uses that one
schema for parsing, generated help, completion candidates, and Clippy-style
interface diagnostics.

```sev
import cli

application = cli.command(
    "betterquest",
    "Local speech generation",
    "0.1.0",
    commands=[
        cli.command(
            "generate",
            "Generate a WAV from text",
            options=[
                cli.option("text", "t", "Text to synthesize", "TEXT", required=true),
                cli.option("output", "o", "Destination WAV", "WAV", kind="path", default_value="speech.wav"),
                cli.flag("verbose", "v", "Show model-loading progress"),
            ],
        ),
    ],
)
```

Use `cli.parse_process(application)` at the executable boundary. That function
explicitly obtains the raw argument vector from `process.arguments()` and
removes the executable name before parsing. Parsing returns `Matches` or a
structured `CliError`; it never prints or exits on the application's behalf.
Tests use `cli.parse(application, values)` with an explicit value list. This
makes the same command definition deterministic and lets BetterQuest decide
whether errors go to a terminal, log, GUI, or service response.

`cli.check(application)` performs schema checks such as duplicate flags,
invalid kebab-case names, missing help, contradictory required/default values,
invalid positional ordering, and variadic arguments that are not last.

See [examples/betterquest.sev](examples/betterquest.sev) for a complete
BetterQuest-oriented command tree.
