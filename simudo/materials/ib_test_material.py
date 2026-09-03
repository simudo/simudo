"""Test material that exercises the v1 callable convention for material
``get_dict()`` entries.

The material
provides scalar layer-properties for the I-region of a p-IB-n device AND
a callable σ(E) for an ``IBBeerLambert`` process named ``opt_ci``.
"""
from simudo.physics import Material


class IBTestMaterial(Material):
    """Minimal IB-region material with a top-hat IB→CB cross-section."""

    name = 'IBTestMaterial'

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        # --- Scalar properties (existing convention; pint quantities) -------
        d.update({
            # Make sure non-radiative trapping has the parameters it needs.
            # sigma_th is a property of the trapping *process*, so it stays
            # under the process name.  vth is a property of the *band* -- the
            # capture coefficient uses the regular (non-trap) band's thermal
            # velocity -- so it belongs under the band name.
            'nr_top/sigma_th':    U('5e-18 cm^2'),
            'nr_bottom/sigma_th': U('5e-18 cm^2'),
            'CB/vth':             U('2e7 cm/s'),
            'VB/vth':             U('2e7 cm/s'),

            # IB band properties.
            'IB/number_of_states': U('2.5e17 cm^-3'),
            'IB/mobility0':        U('500 cm^2/V/s'),

            # Doping for the I region — moderate p-type.
            'poisson/static_rho':  U('1.25e17 elementary_charge/cm^3'),
        })

        # --- Callable property: σ(E) for the opt_ci IBBeerLambert process ---
        # Top-hat: σ = 3e-14 cm² for E ≥ 0.8 eV (E_high = ∞).  The dict
        # shape is the v1 convention defined in
        # gui/notes/TODO-MANY-FIELDS.md §B.  The runner's
        # _register_material_dict dispatches this to add_callable_rule;
        # the GUI's mat_ast scanner recognises it via the 'type' tag.
        d['opt_ci/sigma_function'] = {
            'type':   'top_hat',
            'E_low':  U('0.8 eV'),
            'E_high': float('inf') * U('eV'),
            'value':  U('3e-14 cm^2'),
        }

        return d
