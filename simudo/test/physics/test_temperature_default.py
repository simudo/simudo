"""A project that names no temperature is solved at 300 K, not at 0 K.

Every material reads `temperature` inside `get_dict()` while computing band
gaps, densities of states and thermal velocities, and an unset spatial key
resolves to zero.  So a project that simply forgets it used to fail deep in the
material rather than at the point of the omission -- or, worse, produce a device
at absolute zero.  The runner now registers 300 K at `PRIORITY_MATERIAL`, below
anything the project itself can say, and logs that it did.

Created by Claude Opus 5 (Anthropic), 2026-08-26.
"""

import pytest

from simudo.gui import simudo_1d_runner as runner

from .conftest import MATERIALS_DIR, _q, make_isotype_config


def _spatial(config):
    builder, _ = runner.build_problems(
        config, library_dir=MATERIALS_DIR,
        goals=('local charge neutrality',))
    return builder.problem('local charge neutrality').pdd.spatial


def _without_temperature():
    config = make_isotype_config()
    config['regions'][0]['properties'].pop('temperature', None)
    return config


def test_missing_temperature_defaults_to_300K():
    rules = _spatial(_without_temperature()).value_rules['temperature']
    assert len(rules) == 1, "expected exactly the fallback rule"
    (rule,) = rules
    assert rule.priority == runner.PRIORITY_MATERIAL, (
        "the fallback must sit below anything the project can set")
    assert float(rule.value.m_as('K')) == pytest.approx(
        runner.DEFAULT_TEMPERATURE_K)


def test_explicit_temperature_is_not_overridden():
    """A project that does set a temperature keeps it, and gains no fallback."""
    config = make_isotype_config()
    config['regions'][0]['properties']['temperature'] = _q(77.0, 'K')

    rules = _spatial(config).value_rules['temperature']
    assert len(rules) == 2, (
        "the domain rule is registered twice, in the early and late passes; "
        "no fallback should join them")
    assert all(r.priority == runner.PRIORITY_DOMAIN for r in rules)
    assert all(float(r.value.m_as('K')) == pytest.approx(77.0) for r in rules)
