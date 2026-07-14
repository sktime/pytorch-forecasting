"""A class property decorator."""


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)
