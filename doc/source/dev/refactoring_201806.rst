
Rambling on about refactoring 2018-06
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

This document is mostly obsolete.

Remove pyaml
############

Code execution
==============

The ``pyaml`` implementation is a mess, and the amount of nasty code
in ``simudo/solution/fragment.py`` is appalling.


Object paths
------------

A useful feature in the Pyaml approach is the ability to override
methods in different classes, as a way of wiring things together. We
need a convenient replacement for that.

``/VB``: ``ValenceBand`` class instance

Restructuring
#############

``/$band/$quantity``
example: /VB/j

``/u`` unit_registry

``/optical/field/wavelength=23.3um,phi=45deg/@optical_field``

Memoization
###########

Currently, computing a value stores it in a dictionary
``solution.cdict`` that is specific to the solution object.

However, value lifetime may not match solution object lifetime. If
value lifetime is shorter, then it needs to be manually reset. If it
is longer, then it must be manually set in ``solution.dict``.

There is a better way.

Each object that is the result of a computation records its inputs.

Problem: symbolic expression versus value
=========================================

We don't need to rebuild symbolic expressions (e.g., dolfin, sympy)
every time, even if a terminal changes.

However, we do need to recompute evaluations (e.g., dolfin
projections) if one of the terminals in the expression changes.

Normally the relationship between a symbolic expression and its
evaluation is:

  evaluated = evaluate(symbolic_expression, value_for_each_terminal)

Dolfin blurs the distinction by having the terminals themselves have
values.

Problem: Combinatorial explosion in checking cached value staleness
===================================================================

To check whether a value is stale, we need to check whether its
dependencies are stale. This needs to be re-checked as soon any
mutation occurs. This is expensive. There are two solutions.

#. Mutation of a value invalidates cached values that depend on
   it. This requires setting up a complex system to invalidate only
   what is necessary.

#. Live with it. Mutation happens infrequently, and we can just
   invalidate the entire cache at once. If invalidated, each value
   decides whether it is actually stale (and if it isn't, re-uses the
   current value).

Note that either way, invalidation does *not* imply that the value is
recomputed.

Problem: verbosity
==================

To implement a cached computation, one would need to write something
like::

  class MyValue(CachedComputation):
    def cache_key():
      pass
    def compute_value():
      pass

  cache.register(MyValue)
  cache[MyValue]

::

   /expr:
     _v["/CB/j"]

   v = _._new(cache_policy="dolfin_value")

   a = v[CB/"j"]

   CB('j', cache="dolfin_value")
   CB('expr', cache="dolfin_expr")

Kinds of things
---------------

Plumbing values::

  /VB/j: _.some_j + _.other_j

Plumbing functions::

  /VB/w_to_density: !e lambda w: dolfin.exp(w * _.beta)

Projections and evaluations::

  /VB/density_proj: dolfin.project(VB.w_to_density(_.w), space.CG1)

Problem: plumbing functions, as defined above, fail to establish
dependency. And even more importantly, they do not cache their
results. These can be seen as separate issues.

Fix example::

  /CB/w_to_density: |
    _ = no_track(_)
    @caching_track_v1()
    def w_to_density(cache, w):
      yield (w, _.beta)
      yield dolfin.exp(w * _.beta)
    return w_to_density

Solution
========

Each memoized function call is stored in a table. The table is indexed
by the function identifier and the cache key. A table entry is called
a ``MemoizedEvaluation``.

Consider definitions::

  /dir:
    A: create_variable()
    B: _.A*2
    C: evaluate(_.B)

The evaluation of ``B`` must look up the value of ``A`` in ``_``; this
translates to a call to ``retrieve_path(solution_dictionary,
"/dir/A")``. The function ``retrieve_path`` is also a memoized
function, and its cache key is ``(id(solution_dictionary),
solution_dictionary["/mtime"], path)``. That is, it
does *not* track any further changes. To indicate that mutation has
occurred in the dictionary, all cache entries for ``retrieve_path``
are removed.

Cache clearing
--------------

For most objects, regenerating the value is not a big deal (a waste of
CPU at most). However, certain objects *must* have their identity
preserved. In particular, ``dolfin.Function`` and ``dolfin.Mesh``
instances. If these are regenerated, the UFL expressions depending on
them won't make sense anymore, and will also need to be regenerated.

Easiest hack for now is to clear based on last access time, and to
re-access ("touch") the memoized values that should stay alive right
before "garbage collecting" the cache. A simple way to mark these
values (in a solution dictionary for example) is::

  /path/:
    func: dolfin.Function(...)
    func/@keep: True

In this case, the solution dictionary searches for properties ending
in ``/@keep``. If the value is ``True``, then the part before the "/"
is touched to prevent clearing. If the value is an iterable (which
must then contain ``MemoizedEvaluation`` instances), the iterable is
traversed and the instances are touched.

Request servicing process
-------------------------

0. User code calls ``problemdata["/some/path"]``.
1. A Request object is produced with the wanted path.
2. The path is matched against a regex made up of all responders'
   ``responder_get_path_regex()``. The relevant responders are
   filtered using that regex.
3. ``responder.responder_rejects_request(request)`` is called for each
   responder. Those that answered ``False`` are kept.
4. The list of relevant responders is stored in
   ``request.responders``, and ``request.responder_index`` is set to
   zero.
5. The 


Lookup algorithm
----------------

1. Break path into subpaths, e.g. "/a/b/c" becomes ``["/", "/a/",
   "/a/b/", "/a/b/c/"]``.

2. For each subpath ``subpath``:

   a. If ``$subpath/@mount`` exists (call it ``mount``), then call ``mount(subpath)``.


Problem with mount system: what if the following all exist?

1. ``/a``
1. ``/a/@mount``
1. ``/a/``

Use dependency (topological) sort to establish resolution
order. Hooray for reinventing C3 linearization.

project(ufl_expr, space) -> dolfin.Function

tunneling_recombination: |
  E = poisson.E
  ...
  return dolfin.Function

Things to consider
==================

- unit system
- function subspace registry
- band
- band transport form
- poisson
- poisson form
- material
   - material parameter interdependency
- optical fields
- recombination/generation models
- tunnelling
- interpolated data loaded from disk
- refinement by independent mesh generation
- degenerate bands require a more complicated relationship between qfl and carrier concentration
   - in particular, it uses an approximation for both ways qfl->density and density->qfl
   - the inverse is NOT exactly the identity
   - need to avoid going qfl->density->qfl
      - this currently does not happen in the code, so just watch that it doesn't happen as a result of the refactor


- "In the current code, it seems quite awkward to set up a series of
  runs with different values of a material parameter. so maybe moving
  to python code will make it easier to vary parameters
  programmatically."

pyaml rehaul implement:
- symlink
- textual code gen
- ...
