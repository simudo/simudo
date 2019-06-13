
from collections import OrderedDict

COLOR = { # thanks to emily for color scheme
    'blue' : '#817aab',
    'red'  : '#d33d3c',
    'green': '#78c583'}

BANDS = OrderedDict([
    ('CB', dict(color=COLOR['blue' ], sym='C')),
    ('IB', dict(color=COLOR['red'  ], sym='I')),
    ('VB', dict(color=COLOR['green'], sym='V'))])

ABSORPTIONS = OrderedDict([
    ('cv', dict(color='blue')),
    ('ci', dict(color='red')),
    ('iv', dict(color='orange'))])

