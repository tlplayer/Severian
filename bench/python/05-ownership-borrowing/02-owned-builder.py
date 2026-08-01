class Buffer:
    def __init__(self, values):
        self.bytes = values

    def push(self, byte):
        self.bytes.append(byte)

def freeze(buffer):
    return buffer

buffer = Buffer([])
buffer.push(65)
print(freeze(buffer).bytes)
