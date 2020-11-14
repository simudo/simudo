# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.6.4.0] - 2020-11-13
### Added
- Physics: Thermionic heterojunction boundary condition.
- FEM: Utility method `MeshUtil.region_oriented_dS()` for getting the
  same-side and opposite-side dS facet measures from a FacetRegion.
- Mesh: Null cell and facet regions.
- Added changelog file.

### Changed
- Stepper: If the solver raises a RuntimeError, treat it as a failure
  rather than letting the exception bubble up.
- Stepper: Reduce step size more dramatically on failure.
