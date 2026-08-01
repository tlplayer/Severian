def parse_count(text):
    if text == "":
        return (False, "empty count")
    return (True, int(text))

ok, value = parse_count("42")
print(value)
