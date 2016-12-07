# meta.py - parse config, collect arguments, create instances

import os
import io

from ._compat import PY2, try_encode, ConfigParser

from . import stack, tools

__all__ = ['ConfigMeta', 'StackedMeta']

DEFAULT = 'default'


class ConfigMeta(type):
    """Parse file, create instance for each section, return by section or alias."""

    filename = None

    _pass_notfound = False

    _parser = ConfigParser

    _encoding = None

    _enc = staticmethod(lambda s: s)

    @staticmethod
    def _split_aliases(aliases):
        return aliases.replace(',', ' ').split()

    def __init__(self, name, bases, dct):
        if self.filename is None:
            return

        # work around nose doctest issue
        if self.__module__ in ('__builtin__', 'builtins'):
            self.__module__ = '__main__'

        if not os.path.isabs(self.filename):
            self.filename = os.path.join(tools.class_path(self), self.filename)

        self.filename = os.path.realpath(self.filename)

        if not self._pass_notfound and not os.path.exists(self.filename):
            open(self.filename)

        parser = self._parser()
        enc = self._enc

        if PY2:
            if self._encoding is None:
                parser.read(self.filename)
            else:
                with io.open(self.filename, encoding=self._encoding) as fd:
                    parser.readfp(fd)
                enc = try_encode
        else:
            with io.open(self.filename, encoding=self._encoding) as fd:
                parser.readfp(fd)

        self._keys = []
        self._kwargs = {}
        self._aliases = {}

        for key in parser.sections():
            key = enc(key)
            items = ((enc(k), enc(v)) for k, v in parser.items(key))
            kwargs = dict(items, key=key)

            if 'aliases' in kwargs:
                aliases = kwargs.pop('aliases')
                if aliases.strip():
                    aliases = self._split_aliases(aliases)
                    self._aliases.update((a, key) for a in aliases)
                    kwargs['aliases'] = aliases

            if 'inherits' in kwargs:
                kwargs = dict(((k, v)
                    for k, v in parser.items(kwargs['inherits'])
                    if k != 'aliases'), **kwargs)

            self._keys.append(key)
            self._kwargs[key] = kwargs

        self._cache = {}

    def __call__(self, key=DEFAULT):
        if isinstance(key, self):
            return key

        key = self._aliases.get(key, key)
        if key in self._cache:
            inst = self._cache[key]
        else:
            kwargs = self._kwargs.pop(key)
            inst = self.create(**kwargs)
        return inst

    def create(self, key=None, **kwargs):
        inst = super(ConfigMeta, self).__call__(key=key, **kwargs)

        if key is not None:
            self._cache[key] = inst

        return inst

    def __iter__(self):
        for key in self._keys:
            yield self(key)

    def pprint_all(self):
        for c in self:
            print('%s\n' % c)


class StackedMeta(ConfigMeta):
    """Can register multiple filenames and returns the first match."""

    stack = None

    def __init__(self, name, bases, dct):
        super(StackedMeta, self).__init__(name, bases, dct)

        if self.filename is not None:
            self.stack = stack.ConfigStack(self)

    def add(self, filename, position=0, caller_steps=1):
        if not os.path.isabs(filename):
            filename = os.path.join(tools.caller_path(caller_steps), filename)

        self.stack.insert(position, filename)

    def __getitem__(self, filename):
        return self.stack[filename]

    def __call__(self, key=DEFAULT):
        if isinstance(key, self):
            return key

        for cls in self.stack:
            try:
                return super(StackedMeta, cls).__call__(key)
            except KeyError:
                pass
        else:
            raise KeyError(key)

    def __iter__(self):
        seen = set()
        for cls in self.stack:
            for inst in super(StackedMeta, cls).__iter__():
                if inst.key not in seen:
                    yield inst
                    seen.add(inst.key)

    def __repr__(self):
        if self.stack is None:
            return super(StackedMeta, self).__repr__()
        return '<class %s.%s[%r]>' % (self.__module__, self.__name__, self.filename)
