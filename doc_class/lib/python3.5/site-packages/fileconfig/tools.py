# tools.py - runtime path inspection

import sys
import os
import inspect

__all__ = ['class_path', 'caller_path']


def class_path(cls):
    """Return the path to the source file of the given class."""
    if cls.__module__ == '__main__':
        path = None
    else:
        path = os.path.dirname(inspect.getfile(cls))

    if not path:
        path = os.getcwd()

    return os.path.realpath(path)


def caller_path(steps=1):
    """Return the path to the source file of the current frames' caller."""
    frame = sys._getframe(steps + 1)

    try:
        path = os.path.dirname(frame.f_code.co_filename)
    finally:
        del frame

    if not path:
        path = os.getcwd()

    return os.path.realpath(path)
