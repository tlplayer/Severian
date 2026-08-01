class Button:
    def __init__(self, label):
        self.label = label

    def name(self):
        return self.label

    def draw(self):
        print(self.label)

def render(item):
    item.draw()

render(Button("Save"))
