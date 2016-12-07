# stacks.py - contain config, create and insert subclasses

from ._compat import integer_types

__all__ = ['ConfigStack']


class ConfigStack(object):
    """Ordered and filename-indexed collection of Config classes."""

    def __init__(self, config):
        self._base = config
        self._map = {config.filename: config}
        self._classes = [config]

    def insert(self, index, filename):
        """Insert a new subclass with filename at index, mockup __module__."""
        base = self._base
        dct = {'__module__': base.__module__, 'filename': filename, '_stack': self}
        cls = type(base.__name__, (base,), dct)

        self._map[cls.filename] = cls
        self._classes.insert(index, cls)

    def __getitem__(self, filename):
        if isinstance(filename, integer_types):
            return self._classes[filename]

        return self._map[filename]

    def __iter__(self):
        return iter(self._classes)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self._classes)
