
import functools
from collections import defaultdict
from itertools import chain as ichain
from itertools import combinations

from suffix_trees.STree import STree

__all__ = [
    'make_string_system',
    'format_string_system']

def iterjoin(separator, iterable):
    iterable = iter(iterable)
    yield next(iterable)
    for x in iterable:
        yield separator
        yield x

def all_lcs(stree, minimum_length=0):
    num_strings = len(stree.word_starts)
    seen = set()
    def f(x):
        # wondering about the 'list'?
        # upstream bug, extraneous typecheck:
        #     if stringIdxs == -1 or not isinstance(stringIdxs, list):
        y = stree.lcs(list(x))

        if len(y) >= minimum_length and y not in seen:
            seen.add(y)
            for i in range(num_strings):
                if i not in x:
                    f(x.union((i,)))
    for x in combinations(range(num_strings), 2):
        f(frozenset(x))
    return seen

def default_score(num_occurrences, len_lcs):
    return len_lcs*num_occurrences

def make_string_system(
        strings,
        min_length=6,
        score=default_score):

    num_strings = len(strings)
    strings = [(x,) for x in strings]
    def_index = 0

    while True:
        sstrings = [x for s in strings for x in s if isinstance(x, str)]
        stree = STree(sstrings)

        lcss = all_lcs(stree, minimum_length=min_length)
        if not lcss:
            break
        best = max(lcss, key=lambda sub:
                   score(sum(s.count(sub) for s in sstrings), len(sub)))

        strings = [list(ichain.from_iterable(
            (x,) if not isinstance(x, str)
            else iterjoin(def_index, x.split(best))
            for x in s))
                   for s in strings]
        strings.append((best,))
        def_index += 1

    return (strings[:num_strings], strings[num_strings:])

def format_string_system(string_system_output, replacements=None):
    strings, defs = string_system_output
    if replacements is None:
        replacements = list('{{{}}}'.format(chr(i+65)) for i in range(len(defs)))

    def repl(x):
        return replacements[x] if not isinstance(x, str) else x

    strings = [''.join(map(repl, s)) for s in strings]
    defs    = [''.join(map(repl, s)) for s in defs]

    defs = list(zip(replacements, defs))

    return (strings, defs)

# print(format_string_system(string_system([
#     "function arguments of different types will be cached separately",
#     "bound function is periodically called with the same arguments",
#     'Apply function of two arguments cumulatively to the items of sequence',
#     'argumentative individual'], 5)))
