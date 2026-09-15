#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-sev_compiler}"
OUT="${2:-/tmp/sev-prune}"

rm -rf "$OUT"
mkdir -p "$OUT"

FILES="$OUT/files.txt"
SYMBOLS="$OUT/symbols.tsv"
DUPES="$OUT/duplicate-symbols.txt"
UNREFERENCED="$OUT/unreferenced-symbols.tsv"
LOW_FANIN="$OUT/low-fanin-symbols.tsv"
ORPHAN_FILES="$OUT/orphan-files.tsv"
DUPLICATE_FILES="$OUT/duplicate-files.tsv"

echo "Scanning $ROOT..."

# 1. Sources
git ls-files "$ROOT" \
    | grep -E '\.sev$' \
    > "$FILES" || true

echo "Severian files: $(wc -l < "$FILES")"

# 2. Declarations
while IFS= read -r file; do
    {
        rg -n \
            '^[[:space:]]*(class|trait|enum|def|operator)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' \
            "$file" 2>/dev/null || true
    } | sed -E \
        "s#^([0-9]+):[[:space:]]*(class|trait|enum|def|operator)[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*#\3\t\2\t$file:\1#"
done < "$FILES" \
    | sort \
    > "$SYMBOLS"

echo "Declarations: $(wc -l < "$SYMBOLS")"

# 3. Duplicate declarations
cut -f1 "$SYMBOLS" \
    | sort \
    | uniq -d \
    > "$OUT/duplicate-names.txt"

while IFS= read -r symbol; do
    echo "===== $symbol ====="
    grep -F "${symbol}"$'\t' "$SYMBOLS" || true
    echo
done < "$OUT/duplicate-names.txt" \
    > "$DUPES"

# 4. Fan-in
while IFS=$'\t' read -r symbol kind location; do

    count="$(
        rg -w --glob '*.sev' --glob '*.rs' \
            "$symbol" "$ROOT" 2>/dev/null \
        | wc -l || true
    )"

    count="${count:-0}"

    printf "%s\t%s\t%s\t%s\n" \
        "$count" "$kind" "$symbol" "$location"

done < "$SYMBOLS" \
    | sort -n \
    > "$LOW_FANIN"

awk -F '\t' '$1 <= 1' "$LOW_FANIN" \
    > "$UNREFERENCED"

# 5. Orphan files
while IFS= read -r file; do
    base="$(basename "$file" .sev)"

    refs="$(
        {
            rg -w \
                --glob '*.sev' \
                --glob '*.toml' \
                --glob '*.pkg' \
                "$base" "$ROOT" 2>/dev/null || true
        } \
        | grep -v "^${file}:" \
        | wc -l || true
    )"

    refs="${refs:-0}"

    if [[ "$refs" -eq 0 ]]; then
        printf "%s\t%s\n" "$refs" "$file"
    fi
done < "$FILES" \
    > "$ORPHAN_FILES"

# 6. Identical source files
while IFS= read -r file; do
    sha256sum "$file"
done < "$FILES" \
    | sort \
    | awk '
        $1 == previous_hash {
            if (!printed_previous) {
                print previous_file
                printed_previous=1
            }
            print $2
        }
        $1 != previous_hash {
            printed_previous=0
        }
        {
            previous_hash=$1
            previous_file=$2
        }
    ' \
    > "$DUPLICATE_FILES"

# 7. Large files
while IFS= read -r file; do
    printf "%7d %s\n" "$(wc -l < "$file")" "$file"
done < "$FILES" \
    | sort -rn \
    > "$OUT/largest-files.txt"

# 8. Churn
git log -n 50 --format='' --numstat -- "$ROOT" \
    | awk '
        NF == 3 && $1 != "-" {
            added[$3]+=$1
            deleted[$3]+=$2
        }
        END {
            for (f in added)
                printf "%8d %8d %8d %s\n",
                    added[f]+deleted[f], added[f], deleted[f], f
        }
    ' \
    | sort -rn \
    > "$OUT/churn.txt"

echo
echo "Reports:"
echo "  unreferenced: $UNREFERENCED"
echo "  duplicates:   $DUPES"
echo "  orphan files: $ORPHAN_FILES"
echo "  identical:    $DUPLICATE_FILES"
echo "  large files:  $OUT/largest-files.txt"
echo "  churn:        $OUT/churn.txt"
echo "  fan-in:       $LOW_FANIN"

echo
echo "Top unreferenced:"
head -20 "$UNREFERENCED" || true

echo
echo "Top large files:"
head -20 "$OUT/largest-files.txt" || true

echo
echo "Top churn:"
head -20 "$OUT/churn.txt" || true