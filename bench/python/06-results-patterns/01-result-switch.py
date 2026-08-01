def load(_path):
    return (True, "settings")

ok, value = load("settings.toml")
print(value)
