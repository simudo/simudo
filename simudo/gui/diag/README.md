# Diagnostic and benchmark scripts

One-off scripts from debugging sessions: Jacobian dumps, linear-solver probes,
MUMPS parameter sweeps, solver-reuse benchmarks, and a J(V) comparison helper.

**This directory deliberately has no `__init__.py`**, so `find_packages()` does
not pick it up and none of it goes into the wheel. It was previously in
`simudo/gui/`, where all of it shipped.

The scripts hardcode container paths (`/home/user/simudo/code`) and load the
runner with `exec(open(...).read())`. They are kept because they encode real
debugging approaches, not because they are ready to run as-is.
