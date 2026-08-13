# String

Strings are a core Severian type. Common operations require no import:

```sev
name = "  Jaina Proudmoore  ".strip().title()
words = "one  two\nthree".words()
safe = "Ada-1".filter(|character| character.is_alnum() or character == "-")
line = "_".join(["high", "overlord", "saurfang"])
```

The built-in API includes:

- `strip`, `lstrip`, `rstrip`, `lower`, `upper`, `capitalize`, `title`, and
  `swapcase`;
- `split`, `rsplit`, Python-compatible `splitlines`, `split_lines`, `lines`,
  `words`, `characters`, and both
  separator-style and collection-style `join`;
- `starts_with`, `ends_with`, `contains`, `find`, `rfind`, `index`, `rindex`,
  and `count`;
- `is_empty`, `is_space`, `is_alpha`, `is_digit`, `is_alnum`, `is_ascii`,
  `is_lower`, `is_upper`, `is_ascii_alnum`, `is_word`, and `is_punctuation`;
- `remove_prefix`, `remove_suffix`, repeated `trim_prefix`/`trim_suffix`,
  `translate`, `replace_many`, and `filter`; `remove("()[]{}")` removes any
  listed character, while `remove(["<pause>", "[bell]"])` removes exact
  strings (longest match first), and `remove_all("literal")` removes one exact
  substring everywhere;
- `collapse_space`, `collapse_horizontal_space`, `normalize_space`, `repeat`,
  padding, slicing, extraction, and partition helpers.

`string` remains available as a namespace package for function-oriented code,
but it delegates to the same core methods. Regex remains a separate package.

String indexing and `characters()` are Unicode-scalar aware. Character
classification and case conversion currently use the native runtime's ASCII
classification; a distinct Unicode `char` type and Unicode data tables remain
future work.
