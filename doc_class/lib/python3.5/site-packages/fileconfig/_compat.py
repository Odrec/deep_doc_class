# _compat.py - Python 2/3 compatibility

import sys

PY2 = sys.version_info[0] == 2


if PY2:  # pragma: no cover
    integer_types = (int, long)

    def iteritems(d):
        return d.iteritems()

    def try_encode(chars, encoding='ascii'):
        """Return encoded chars, leave unchanged if encoding fails.

        >>> try_encode(u'spam')
        'spam'

        >>> assert try_encode(u'm\xf8\xf8se') == u'm\xf8\xf8se'
        """
        try:
            return chars.encode(encoding)
        except UnicodeEncodeError:
            return chars

    from ConfigParser import SafeConfigParser as ConfigParser


else:  # pragma: no cover
    integer_types = (int,)

    def iteritems(d):
        return iter(d.items())

    def try_encode(chars, encoding='ascii'):
        return chars

    from configparser import ConfigParser


def with_metaclass(meta, *bases):
    """From Jinja2 (BSD licensed).

    http://github.com/mitsuhiko/jinja2/blob/master/jinja2/_compat.py
    """
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__
        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)
    return metaclass('temporary_class', None, {})
