from __future__ import absolute_import, division, print_function

import os
from builtins import bytes, dict, int, range, str, super

from future.utils import PY2, PY3, native

__all__ = ['generate_base32_token']

def generate_base32_token(length):
    base32_lut = 'ybndrfg8ejkmcpqxot1uwisza345h769'
    return ''.join(base32_lut[i&31] for i in bytes(os.urandom(length)))
