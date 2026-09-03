"""The `domain` overlay must sit *below* layers in the rule priority order.

`domain` is how a project states a device-wide default.  A value set on one
layer therefore has to override it, or the Layers panel's most common gesture --
"this everywhere, except here" -- does nothing.  It did nothing for a while:
`domain` shared `PRIORITY_OVERLAY` with named overlay regions, which sit *above*
layers, so `Spatial.get` (first rule to claim a cell wins) always took the
device-wide value and the layer-level override was silently inert.  Measured on
the GUI tutorial's silicon cell, a 10^6 contrast in SRH lifetime between
`domain` and one layer changed the current by not one bit.  See
`doc/release0_7/06_known_issues.md` A62.

Created by Claude Opus 5 (Anthropic), 2026-08-26.
"""

import pytest

from simudo.gui import simudo_1d_runner as runner
from simudo.util import make_unit_registry

from .conftest import MATERIALS_DIR, _q, make_isotype_config


def test_priority_constants_are_ordered():
    """Named overlay > layer > domain > material, lower number winning."""
    assert (runner.PRIORITY_OVERLAY
            < runner.PRIORITY_LAYER
            < runner.PRIORITY_DOMAIN
            < runner.PRIORITY_MATERIAL)


class _RecordingSpatial:
    """Stands in for `Spatial`, capturing each rule's priority."""

    def __init__(self):
        self.calls = []

    def add_callable_rule(self, key, region, func, priority=0):
        self.calls.append((key, region, priority))


class _Regions:
    """Stands in for `CellRegions`, which makes regions on attribute access."""

    domain = 'domain-region'

    def __getattr__(self, name):
        return f'{name}-region'


def _top_hat(value, unit):
    return {'top_hat': {'E_low': _q(1.0, 'eV'), 'E_high': _q(9.0, 'eV'),
                        'value': value, 'unit': unit}}


@pytest.mark.parametrize('register, spec_unit, values', [
    (runner._register_beer_lambert_alpha, '1/cm', (1e4, 2e4)),
    (runner._register_ib_beer_lambert_sigma, 'cm^2', (1e-14, 2e-14)),
])
def test_callable_rules_put_domain_below_layers(register, spec_unit, values):
    """`alpha_function` / `sigma_function` on `domain` must lose to a layer."""
    config = {'domain': _top_hat(values[0], spec_unit),
              'narrow': _top_hat(values[1], spec_unit)}
    spatial = _RecordingSpatial()
    register('opt', config, spatial, _Regions(), make_unit_registry(),
             layer_names={'wide', 'narrow'})

    got = {region: priority for _, region, priority in spatial.calls}
    assert got['domain-region'] == runner.PRIORITY_DOMAIN
    assert got['narrow-region'] == runner.PRIORITY_LAYER
    assert got['narrow-region'] < got['domain-region']


def test_layer_rule_outranks_domain_rule():
    """The scalar path, on a real assembled `Spatial` and with no solve.

    `Spatial.value_rules[key]` is a `SortedList` ordered by priority, and
    `Spatial.get` gives each cell to the first rule that claims it.  So the
    assertion that matters is an ordering one: the layer's rule has to come
    first.
    """
    DOMAIN_TAU, LAYER_TAU = 1.0e-9, 1.0e-3

    config = make_isotype_config()
    config['regions'][0]['properties']['SRH/CB/tau'] = _q(DOMAIN_TAU, 's')
    config['layers'][1]['properties']['SRH/CB/tau'] = _q(LAYER_TAU, 's')

    builder, _ = runner.build_problems(
        config, library_dir=MATERIALS_DIR,
        goals=('local charge neutrality',))
    rules = builder.problem(
        'local charge neutrality').pdd.spatial.value_rules['SRH/CB/tau']

    priorities = [r.priority for r in rules]
    assert priorities == sorted(priorities), "value_rules must stay sorted"
    assert priorities[0] == runner.PRIORITY_LAYER, (
        f"the layer rule must come first, got priorities {priorities}")
    assert runner.PRIORITY_DOMAIN in priorities, (
        "the domain default should still be registered, just outranked")
