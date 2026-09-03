# Robin (Surface Recombination Velocity) Boundary Conditions

## Physics

The surface recombination velocity (SRV) boundary condition on a carrier band at a surface is:

```
j·n = s * q * (u - u_eq)
```

where `s` is the SRV [cm/s], `q` is the elementary charge, `u` is the local carrier density,
and `u_eq` is the thermal-equilibrium carrier density. This is a Robin-type (mixed) boundary
condition: the normal current flux is proportional to the deviation of carrier density from
equilibrium. The two existing BCs are limiting cases:

- **ohmic**: `s → ∞` — enforces `u = u_eq` as an essential BC
- **zero_current**: `s = 0` — enforces `j·n = 0` as an essential BC on `j`

---

## Relation to the standard Robin BC formulation

The standard Robin BC for a diffusion problem (see Langtangen & Dokken, FEniCSx tutorial
ch. 3) is:

```
-κ ∂u/∂n = r(u - s)   on Γ_R
```

After integration by parts and substituting the BC, the weak form acquires:

```
F(u,v) += ∫_{Γ_R} r(u - s) v ds
```

Split into bilinear and linear contributions:
```
a(u,v) += ∫_{Γ_R} r·u·v ds       ← u-dependent: stiffens the boundary row in the Jacobian
L(v)   += ∫_{Γ_R} r·s·v ds       ← source term (does not depend on the solution)
```

In FEniCS this is implemented by adding `r * inner(u - s, v) * ds(marker)` directly to the
residual form `F` — **not** as a DirichletBC.

The SRV condition `j·n = s_srv * q * (u - u_eq)` is exactly this structure with
`r = s_srv * q` and `s = u_eq`. By identifying `-κ ∂u/∂n ↔ j·n` and `u ↔` carrier density,
the surface integral to add to the residual is:

```
∫_{Γ_SRV} s_srv * q * (u - u_eq) * v ds
```

This belongs in the **continuity** (`v`-tested) row, since that is the equation where `∇·j`
(and hence the boundary flux `j·n`) appears.

### Where the BC enters Simudo's mixed weak form

The `MixedQflBand` weak form uses two sets of test functions:

1. **Continuity row** (tested with `v`): `∫ v ∇·j dx = ∫ v g dx`
   — `j` is the BDM unknown; `∇·j` is exact within each cell.
   — The SRV surface flux belongs here as `∫_{Γ_SRV} v * s_srv * q * (u - u_eq) ds`.

2. **Drift-diffusion row** (tested with `ξ`): after integration by parts,
   `∫ ξ·j/μu dx + ∫ ∇·ξ δw dx = ∫_{∂Ω} δw (ξ·n) ds`
   — This is the natural BC slot for specifying `δw` (quasi-Fermi level).
   — The SRV condition does **not** go here.

For SRV, the condition `j·n = s_srv*q*(u - u_eq)` does **not** fit the existing framework as:
- an essential BC on `u` (would require knowing `j·n` in advance), or
- a natural BC specifying `δw` (would require inverting the nonlinear `u(δw)` relation).

The full residual form with the SRV term is:

```
∫_Ω v ∇·j dx + ∫_{Γ_SRV} v * s_srv * q * (u - u_eq) ds = ∫ v g dx
```

### Nonlinearity and the Newton Jacobian

In the tutorial's standard CG Lagrange setting, `u` is the trial function directly, so the
split into `a(u,v)` (bilinear) and `L(v)` (linear) is clean. In Simudo's mixed method, `u`
is a **nonlinear function of `delta_w`** (the DG trial variable):

```
u = qfl_to_u(base_w + delta_w)
```

So `∫ s_srv * q * u * v ds` is a nonlinear boundary term. Simudo's Newton solver applies
`dolfin.derivative(F, mixed_function)` to obtain the Jacobian automatically. The
`u(delta_w)` dependence is differentiated correctly, giving a Jacobian contribution:

```
∫_{Γ_SRV} s_srv * q * (∂u/∂(delta_w)) * δ(delta_w) * v ds
```

The `u_eq` term is constant (does not depend on the solution) and contributes only to
the residual (right-hand side), just like `L(v)` in the tutorial.

---

## Analogy with ThermionicHeterojunction

`ThermionicHeterojunction` in `simudo/physics/heterojunction.py` is the closest existing
pattern. Its `register()` method adds a penalty/residual term to
`band.mixedqfl_drift_diffusion_heterojunction_bc_term`, which goes into the
**drift-diffusion (ξ-tested) row**, enforcing a target `Δw` weakly via `dS * (Δw - Δw_BC) *
dot(ξ, n)`.

For SRV the cleanest approach is different: add the Robin flux directly to the
**continuity (v-tested) row**. This is more natural because the Robin condition directly
constrains `j·n`, which appears in the continuity equation.

---

## Implementation plan

### 1. New class in `simudo/physics/heterojunction.py`

```python
@attr.s
class SurfaceRecombination:
    """
    Surface recombination velocity (SRV) boundary condition.

    Imposes  j·n = s * q * (u - u_eq)  on the given boundary facet,
    where s is the SRV [cm/s], u is the carrier density, and u_eq is
    the thermal-equilibrium carrier density.

    Parameters
    ----------
    band : Band
        The carrier band (MixedQflBand) to apply the BC to.
    boundary : FacetRegion
        The boundary facet region where SRV is applied.
    srv : pint Quantity
        Surface recombination velocity, e.g. 1000 cm/s.
    """
    band: Band = attr.ib()
    boundary: FacetRegion = attr.ib()
    srv: object = attr.ib()   # pint Quantity with velocity units

    def register(self):
        band = self.band
        mu = band.pdd.mesh_util
        U = mu.unit_registry

        s   = self.srv
        q   = U.elementary_charge
        u   = band.u
        u_eq = band.thermal_equilibrium_u

        # Robin flux: j·n = s * q * (u - u_eq)
        srv_flux = s * q * (u - u_eq)   # units: A / mesh_unit^2

        # Build ds measure restricted to the SRV boundary
        fvss = mu.mesh_data.evaluate_topology(self.boundary)
        ds_srv = sum(mu.ds(subdomain_id=fv) for fv, sign in fvss)

        v = band.mixedqfl_v

        # Add to continuity row.  band.sign converts flux to a source/sink
        # consistent with the sign convention of the continuity equation.
        band.mixedqfl_surface_recombination_term += (
            ds_srv * v * band.sign * srv_flux
        )
```

### 2. `simudo/physics/poisson_drift_diffusion.py`

Add a new `cached_property` to `MixedQflBand` (near the existing
`mixedqfl_drift_diffusion_heterojunction_bc_term` at line 763):

```python
@cached_property
def mixedqfl_surface_recombination_term(self):
    return DelayedForm()
```

### 3. `simudo/physics/poisson_drift_diffusion1.py1`

Add `surface_recombination_term` to the `MixedQflBand.mixedqfl_` block:

```yaml
surface_recombination_term: !c 'band.mixedqfl_surface_recombination_term'
```

Then include it in `weak_form`:

```yaml
weak_form: !c |
  + _.continuity_j_term
  + _.continuity_g_term
  + _.surface_recombination_term          # ← new
  + _.drift_diffusion_bc_term
  + _.drift_diffusion_heterojunction_bc_term
  + _.drift_diffusion_j_term
  + _.drift_diffusion_qfl_term
  + _.drift_diffusion_base_w_jump_term
  + _.external_qfl_term
  + _.external_j_term
```

### 4. `gui/simudo_1d_runner.py`

In the BC dispatch block (around line 824), add a new branch:

```python
elif bc_type == 'surface_recombination':
    srv_quantity = parse_quantity(bc_type_config['srv'], U)
    SurfaceRecombination(band_obj, face, srv=srv_quantity).register()
```

The `bc_type_config` would be the dict value under the band key in the YAML (see schema below).
Unlike ohmic/zero_current, this does **not** call `spatial.add_BC` — it only registers
the weak-form term directly on the band object.

The import at the top of `simudo_1d_runner.py` would need:
```python
from simudo.physics.heterojunction import ThermionicHeterojunction, SurfaceRecombination
```

### 5. YAML schema

The band BC value can be either a plain string (`"ohmic"`, `"zero_current"`) or a dict for
parameterised BCs. The dispatch in `simudo_1d_runner.py` already reads
`bc_type = bands_resolved[bname]` — this would be extended to also accept dicts:

```python
if isinstance(bc_type_raw, dict):
    bc_type = bc_type_raw['type']
    bc_type_config = bc_type_raw
else:
    bc_type = bc_type_raw
    bc_type_config = {}
```

Example YAML:

```yaml
contacts:
  left:
    role: swept
    bands:
      CB:
        type: surface_recombination
        srv:
          value: 1000.0
          unit: cm/s
      VB: ohmic
  right:
    role: reference
    bands:
      CB: ohmic
      VB:
        type: surface_recombination
        srv:
          value: 100.0
          unit: cm/s
```

---

## Critical constraint: no conflicting essential BCs

The `bcpair` mechanism (lines 443–451 of `poisson_drift_diffusion1.py1`) constructs
DirichletBCs on the BDM `j` space whenever `{bname}/j` is registered via `spatial.add_BC`.
If an essential BC on `j` (or `u`) is registered on the same facet as an SRV condition,
it will override the weak-form Robin term. Therefore:

- **Do not call `spatial.add_BC(f'{bname}/j', face, ...)` for an SRV face.**
- **Do not call `spatial.add_BC(f'{bname}/u', face, ...)` for an SRV face.**

The weak-form term alone is sufficient. The BDM normal flux on an exterior face with no
essential BC remains a free variable that the Robin residual will correctly drive.

---

## Files to change (summary)

| File | Change |
|---|---|
| `simudo/physics/heterojunction.py` | Add `SurfaceRecombination` class |
| `simudo/physics/poisson_drift_diffusion.py` | Add `mixedqfl_surface_recombination_term` cached property to `MixedQflBand` |
| `simudo/physics/poisson_drift_diffusion1.py1` | Add `surface_recombination_term` alias and include in `weak_form` |
| `gui/simudo_1d_runner.py` | Parse dict-valued BC configs; add `surface_recombination` dispatch branch |

No changes to `spatial.py`, the mesh topology, the Poisson equation, or the GUI panels
are required for a first working implementation.

## Alternative: natural BC in the ξ-tested (drift-diffusion) row

A structurally different implementation is possible that does **not** constrain the j·n DOF as an essential BC. Instead, the SRV condition is enforced *weakly* by adding a surface term to the ξ-tested drift-diffusion row — the same slot used by `ThermionicHeterojunction` for interface currents.

### Background: the ξ-tested row and its boundary term

In Simudo's BDM mixed formulation, each carrier band has two blocks of equations per cell K:

| Row | Test fn | Enforces |
|---|---|---|
| (v) continuity | v ∈ DG | ∇·j = R (generation–recombination) |
| (ξ) drift-diffusion | ξ ∈ BDM | constitutive law: j = −κ(u) ∇(Δw) |

The drift-diffusion constitutive law is rewritten in weak form. After integrating by parts the ∇(Δw) term over element K:

```
∫_K (ξ/κ)·j dx  −  ∫_K Δw ∇·ξ dx  +  ∫_{∂K} Δw_bdry (ξ·n) ds  =  0
```

where κ = μu (effective conductivity, nonlinear in u). The boundary integral `∫_{∂K} Δw_bdry (ξ·n) ds` supplies the *natural* boundary value of Δw in the mixed sense — it is the analogue of a Dirichlet condition on Δw in the primal sense.

On an **exterior SRV face** Γ ⊂ ∂Ω, the standard formulation would impose Δw = 0 (ohmic) or leave j·n = 0 (zero-current). For a finite SRV we instead want

```
j·n_out = band.sign · s · q · (u − u_eq)         (★)
```

with u depending on Δw through the band equation of state.

### Replacing the essential BC with a surface term

Leave the j·n DOF **free** (no DirichletBC). Remove Γ from the set of Dirichlet faces in both the v-row and the ξ-row. Add to the ξ-tested row the surface residual

```
∫_Γ (ξ·n) · [band.sign · s · q · (u(Δw) − u_eq)] ds
```

This weakly enforces (★): if the residual is zero, the j·n DOF equals the right-hand side of (★) at every quadrature point on Γ. No essential BC on j is needed; the coupling between j·n and Δw lives entirely inside the ξ-tested row.

In Simudo this term would be returned by the `mixedqfl_drift_diffusion_heterojunction_bc_term` cached property (currently returning an empty `DelayedForm()`), analogously to the `ThermionicHeterojunction` pattern.

### The band equation of state u(Δw)

The carrier density u is a nonlinear function of Δw; the explicit form depends on the band model:

- **Boltzmann**: `u = u_eq · exp(band.sign · q · Δw / kT)`, so
  `u − u_eq = u_eq · (exp(band.sign · q · Δw / kT) − 1)`
- **Fermi–Dirac**: u is given by a Fermi–Dirac integral of the quasi-Fermi level; no closed-form inversion exists.

The weak-form surface term is therefore

```
∫_Γ (ξ·n) · band.sign · s · q · (u(Δw) − u_eq) ds
```

with u(Δw) left in this generic form; the assembly substitutes whichever equation of state the band uses.

### Linearisation

Because u(Δw) is nonlinear, the surface term contributes to the Newton Jacobian. Two options:

**Picard** (simpler): freeze u at the previous iterate before each Newton step, producing the constant surface term

```
∫_Γ (ξ·n) · band.sign · s · q · (u_prev − u_eq) ds
```

This is equivalent in structure to the Picard essential-BC approach (option 1) but avoids touching the essential-BC list entirely.

**Full Newton** (more accurate): differentiate the surface term with respect to Δw analytically. For Boltzmann bands the Jacobian contribution is

```
d/d(Δw) [band.sign · s · q · (u − u_eq)]  =  s · q · (q / kT) · u
```

(using du/d(Δw) = band.sign · (q/kT) · u for Boltzmann). For Fermi–Dirac bands the derivative involves the Fermi–Dirac integral of order −1/2 and is more involved.

### Comparison with option 1

| | Option 1 (essential BC on j) | Option 2 (ξ-row surface term) |
|---|---|---|
| j·n DOF | constrained (DirichletBC) | free, determined by system |
| Picard update | in `rebase_w` hook | assembled each Newton step |
| Full Newton | not available | available (Boltzmann easy; Fermi harder) |
| Implementation | `DirichletBC` + updater list | `DelayedForm` surface term |
| Simudo hook | `get_essential_bcs()` | `mixedqfl_drift_diffusion_heterojunction_bc_term` |

Option 2 is more natural from the variational standpoint and enables a proper Newton Jacobian for Boltzmann bands. The trade-off is that the Jacobian entry for Fermi–Dirac bands is more complex, and the implementation requires understanding the exact normalisation of the ξ-tested row in Simudo.

---

## Implementation + empirical results (2026-05-22)

Three placements are now implemented in `simudo/physics/boundary_conditions.py`,
selectable via `SurfaceRecombination.method` and from YAML via a `method:` key
on the `surface_recombination` BC:

- `'drift_diffusion'` (option 2, **default**; ξ-row natural BC — see Finding 3)
- `'continuity_corrected'` (option 1 with the double-count removed —
  see Finding 4)
- `'continuity'` (option 1, original; **does not converge** — see Finding 1;
  retained only to document the failure mode)

### Finding 1 — option 1 (continuity-row term) does not converge

The current continuity-row form adds `∫_Γ band.sign·s·q·(u−u_eq)·v ds` to the
**v-tested** row while leaving `j·n` free. This **double-counts the boundary
flux**: for the DG component of `v`, `∫_K v ∇·j dx = v·∫_∂K j·n ds` already
contains the boundary flux, so adding a second surface term makes the row
enforce `j·n + sign·srv_flux = source` rather than `j·n = sign·srv_flux`.
The `j·n` DOF ends up weakly/under-constrained → a near-null direction.
Empirically the solver hangs at I≈1e-25 (dark): `|b|`~1e-12 but
`|du|/(r|u|+a)`~1.4 never drops below 1, step size collapses to ~1e-27.
This is the cause of the `fourlayer_ibBeerLambert.yaml` failure.

### Finding 2 — the ξ-row term as written above is dimensionally invalid

The "Alternative" section proposes adding `∫_Γ (ξ·n)·band.sign·s·q·(u−u_eq) ds`
to the ξ-row. But the MixedQfl ξ-row's natural boundary term imposes a
**quasi-Fermi level (potential, eV)**, not a flux — confirmed by both
`drift_diffusion_bc_term` (`−δw_bc·(ξ·n)`) and the heterojunction term
(`BC_term·U("eV")·(ξ·n)`). Multiplying `(ξ·n)` by a flux `[A/area]` gives a
term that does not match the row's `[eV]·(ξ·n)` units; Simudo raises a
`pint.DimensionalityError` on the first Newton assembly.

### Finding 3 — corrected option 2 (QFL-inversion natural BC) works

The fix: impose the **equivalent boundary QFL** instead of the flux. Invert the
band equation of state so the carrier density the QFL produces yields the
required Robin flux:

```
j·n = band.sign·s·q·(u − u_eq)
  ⟹  u_target = u_eq + (j·n)/(band.sign·s·q)
  ⟹  w_target = band.u_to_qfl(u_target)        # FD inverse — valid for degenerate bands
      δw_Γ     = w_target − base_w
ξ-row += −∫_Γ δw_Γ · (ξ·n) ds                   # same sign convention as drift_diffusion_bc_term
```

`j·n = mu.dot(band.mixedqfl_j, mu.n)` is the live BDM trial flux, so the term is
nonlinear in the solution and dolfin's autodiff supplies the full Newton
Jacobian (no Picard freeze). This is dimensionally correct (`δw_Γ` is eV) and
uses `band.u_to_qfl`, which is the Fermi–Dirac inverse — so the "Fermi harder"
caveat does not apply.

### Finding 4 — fixing option 1's double-count also works (v-row, corrected)

Removing the double-counted boundary flux from option 1 fully repairs it. The
v-row already supplies `∫_Γ v (j·n)` (via `∫_K v∇·j`); subtract that and
substitute the Robin flux:

```
v-row += ∫_Γ v · (srv_flux − j·n) ds                # srv_flux = band.sign·s·q·(u − u_eq)
                                                    # j·n      = mu.dot(band.mixedqfl_j, mu.n)
```

The conservation row then enforces `j·n|_Γ = srv_flux` directly, without an
essential BC.  Both terms are nonlinear in the solution (`u` via δw, and the
live trial flux `j·n`), so dolfin's autodiff supplies the full Newton
Jacobian — no Picard.  Exposed as `method='continuity_corrected'`.

### Head-to-head

On `fourlayer_ibBeerLambert.yaml` (SRV 1000 cm/s on left/CB, 1-sun AM1.5G,
V-sweep to 1.0 V):

| | `continuity` (opt 1) | `continuity_corrected` (opt 1 fixed) | `drift_diffusion` (opt 2) |
|---|---|---|---|
| Result | hangs at I≈1e-25 (dark) | **Run complete**: ramp + V→1.0 V | **Run complete**: ramp + V→1.0 V |
| Wall time | does not finish | ~16 min | ~5.7 min |
| Jsc (V=0) | — | −12.14 mA/cm² | −12.23 mA/cm² |

The fact that two independent weak-form placements (`continuity_corrected` and
`drift_diffusion`) converge to Jsc values within ~1% of each other is a useful
cross-check that both are physically correct.

**Conclusion: option 2 (`drift_diffusion`) is ~3× faster and the recommended
default.** `continuity_corrected` works fully and is useful as a validation
cross-check.  The original `continuity` is retained behind the selector solely
for documentation of the failure mode — it should not be used in practice.
The literal ξ-row × flux form from the "Alternative" section above is wrong
(dimensional); the working ξ-row form is the QFL-inversion in Finding 3.

---