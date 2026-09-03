"""Tests for deprecated spatial-key aliasing.

See :py:data:`simudo.fem.spatial.KEY_ALIASES`. Thermal velocity is a band
property spelled ``vth``; ``v_th`` is the deprecated alias kept so that older
material files and project YAML keep working.

The important negative case is that aliasing applies to the *final* path
component only, so a key such as ``v_th/tau`` -- where ``v_th`` names a
process or region rather than the quantity -- must pass through untouched.
"""
import warnings

import pytest

import simudo.fem.spatial as spatial_module
from simudo.fem.spatial import KEY_ALIASES, canonicalize_key


@pytest.fixture(autouse=True)
def reset_warned_keys():
    '''Deprecation warnings fire once per key per session; reset between tests.'''
    spatial_module._warned_legacy_keys.clear()
    yield
    spatial_module._warned_legacy_keys.clear()


@pytest.mark.parametrize("key,expected", [
    # --- deprecated spelling in the final component: rewritten -------------
    ("nr_top/v_th",           "nr_top/vth"),
    ("nr_bottom/v_th",        "nr_bottom/vth"),
    ("nr_ci/v_th",            "nr_ci/vth"),
    ("v_th",                  "vth"),

    # --- already canonical: unchanged --------------------------------------
    ("CB/vth",                "CB/vth"),
    ("VB/vth",                "VB/vth"),
    ("nr_top/vth",            "nr_top/vth"),

    # --- 'v_th' NOT in the final component: must be left alone -------------
    ("v_th/tau",              "v_th/tau"),
    ("v_th/sigma_th",         "v_th/sigma_th"),
    ("foo/v_th/tau",          "foo/v_th/tau"),
    ("v_th/v_th/tau",         "v_th/v_th/tau"),

    # --- final component merely *contains* 'v_th': must not match ----------
    ("nr_top/v_theta",        "nr_top/v_theta"),
    ("nr_top/xv_th",          "nr_top/xv_th"),
    ("nr_top/v_th_eff",       "nr_top/v_th_eff"),

    # --- unrelated keys -----------------------------------------------------
    ("temperature",           "temperature"),
    ("poisson/static_rho",    "poisson/static_rho"),
    ("opt_cv/alpha_function", "opt_cv/alpha_function"),
    ("nr_top/sigma_th",       "nr_top/sigma_th"),
    ("",                      ""),
])
def test_canonicalize_key(key, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert canonicalize_key(key) == expected


@pytest.mark.parametrize("key", [
    "nr_top/v_th", "v_th", "v_th/tau", "CB/vth", "temperature",
])
def test_canonicalize_key_is_idempotent(key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        once = canonicalize_key(key)
        assert canonicalize_key(once) == once


def test_deprecated_key_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        canonicalize_key("nr_top/v_th")
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    message = str(caught[0].message)
    assert "nr_top/v_th" in message and "nr_top/vth" in message


def test_deprecated_key_warns_only_once_per_key():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            canonicalize_key("nr_top/v_th")
        canonicalize_key("nr_bottom/v_th")
    # once for each distinct key, not once per call
    assert len(caught) == 2


@pytest.mark.parametrize("key", [
    "v_th/tau", "CB/vth", "temperature", "nr_top/v_theta",
])
def test_untouched_keys_do_not_warn(key):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        canonicalize_key(key)
    assert caught == []


def test_add_rule_stores_canonical_key():
    '''Writing with the old spelling must land under the canonical key.'''
    spatial = spatial_module.Spatial()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        spatial.add_rule("nr_top/v_th", None, 2.0e7)

    assert "nr_top/vth" in spatial.value_rules
    assert "nr_top/v_th" not in spatial.value_rules


def test_add_rule_does_not_rewrite_non_final_component():
    spatial = spatial_module.Spatial()
    spatial.add_rule("v_th/tau", None, 1.0e-9)

    assert "v_th/tau" in spatial.value_rules
    assert "vth/tau" not in spatial.value_rules


def test_old_and_new_spellings_share_one_rule_list():
    '''A value written as ``v_th`` must be visible to a reader asking ``vth``.'''
    spatial = spatial_module.Spatial()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        spatial.add_rule("nr_top/v_th", None, 2.0e7)
        spatial.add_rule("nr_top/vth", None, 3.0e7)

    assert len(spatial.value_rules["nr_top/vth"]) == 2


def test_alias_table_targets_are_themselves_canonical():
    '''No alias may point at another alias, or canonicalisation is order-dependent.'''
    for alias, canonical in KEY_ALIASES.items():
        assert canonical not in KEY_ALIASES
