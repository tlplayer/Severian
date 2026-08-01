def find_name(identifier):
    return "ada" if identifier == 1 else None

name = find_name(1)
if name:
    print(name)
elif name is not None:
    print("blank")
else:
    print("missing")
