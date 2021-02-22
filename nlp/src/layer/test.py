class GrandParent:
    def __init__(self):
        self._name = "grand parent"

    @property
    def name(self):
        return self._name


class Parent(GrandParent):
    def __init__(self):
        super().__init__()
        self._name = "parent"

    @property
    def name(self):
        return super().name


class Child(Parent):
    def __init__(self):
        super().__init__()
        self._name = "child"

    @property
    def name(self):
        return super(Parent).name


print(Child().name)