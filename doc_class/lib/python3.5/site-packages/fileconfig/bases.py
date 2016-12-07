# bases.py - to be subclassed by client code

from ._compat import iteritems, with_metaclass

from . import meta

__all__ = ['Config', 'Stacked']


class Config(with_metaclass(meta.ConfigMeta, object)):
    """Return section by name from filename as instance."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        items = ('  %r: %r' % (k, v) for k, v in sorted(iteritems(self.__dict__)))
        return '{\n%s\n}' % ',\n'.join(items)

    def __repr__(self):
        if getattr(self, 'key', None) is None:
            return '<%s.%s object at %#x>' % (self.__module__,
                self.__class__.__name__, id(self))
        return '%s.%s(%r)' % (self.__module__, self.__class__.__name__, self.key)

    @property
    def names(self):
        """Names, by which the instance can be retrieved."""
        if getattr(self, 'key', None) is None:
            result = []
        else:
            result = [self.key]
        if hasattr(self, 'aliases'):
            result.extend(self.aliases)
        return result


class Stacked(with_metaclass(meta.StackedMeta, Config)):
    """Return section by name from first matching file as instance."""

    def __repr__(self):
        if getattr(self, 'key', None) is None:
            return '<%s.%s[%r] object at %#x>' % (self.__module__,
                self.__class__.__name__, self.__class__.filename, id(self))
        return '%s.%s[%r](%r)' % (self.__module__, self.__class__.__name__,
            self.__class__.filename, self.key)
