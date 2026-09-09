"""
Simudo GUI — project data model.

Plain Python dataclasses that map 1-to-1 with the project YAML schema.
No Panel / param dependencies here; this module must be importable without
any GUI framework installed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import datetime


def _default_output_folder() -> str:
    return "out/" + datetime.date.today().strftime("%Y%b%d") + "/a"


# ── Colours assigned to layers in the schematic ──────────────────────────────

LAYER_COLORS = [
    '#e07b39',  # orange
    '#5b7fcc',  # blue
    '#9b59b6',  # purple
    '#27ae60',  # green
    '#e74c3c',  # red
    '#1abc9c',  # teal
    '#f39c12',  # amber
    '#2980b9',  # dark blue
    '#8e44ad',  # violet
    '#16a085',  # dark teal
]


def default_layer_color(index: int) -> str:
    return LAYER_COLORS[index % len(LAYER_COLORS)]


# ── Value + unit pair ─────────────────────────────────────────────────────────

@dataclass
class ValueUnit:
    value: float = 0.0
    unit: str = ""

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit}

    @classmethod
    def from_dict(cls, d) -> "ValueUnit":
        if isinstance(d, dict):
            return cls(value=d.get("value", 0.0), unit=d.get("unit", ""))
        return cls(value=float(d), unit="")


# ── Mesh parameters ───────────────────────────────────────────────────────────

@dataclass
class MeshParams:
    start: Optional[float] = None   # starting mesh spacing
    factor: Optional[float] = None  # mesh spacing expansion factor


# ── Layer ─────────────────────────────────────────────────────────────────────

@dataclass
class Layer:
    name: str = "layer"
    thickness: float = 1.0
    thickness_unit: str = "um"
    material: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    mesh: MeshParams = field(default_factory=MeshParams)
    # GUI-only: display colour (not saved to YAML)
    color: str = "#4a9eff"


# ── Overlay region ────────────────────────────────────────────────────────────

@dataclass
class OverlayRegion:
    name: str = "overlay"
    # Extent: either coordinate range or layer-name list.
    # None = not yet specified.
    start: Optional[float] = None
    end: Optional[float] = None
    extent_unit: str = "um"
    layer_names: List[str] = field(default_factory=list)  # alternative to coords
    properties: Dict[str, Any] = field(default_factory=dict)
    is_domain: bool = False  # True for the built-in 'domain' overlay


def make_domain_overlay() -> OverlayRegion:
    return OverlayRegion(
        name="domain", is_domain=True,
        properties={"temperature": {"value": 300.0, "unit": "K"}},
    )


# ── Band ──────────────────────────────────────────────────────────────────────

BAND_TYPE_YAML_TO_GUI = {
    "nondegenerate": "Boltzmann",
    "degenerate":    "Fermi (parabolic)",
    "intermediate":  "Sharp Intermediate Band",
}
BAND_TYPE_GUI_TO_YAML = {v: k for k, v in BAND_TYPE_YAML_TO_GUI.items()}


@dataclass
class Band:
    name: str = "CB"
    type: str = "nondegenerate"   # YAML value
    sign: int = -1                # -1 electrons, +1 holes
    extent: Any = "all"           # 'all' or list of region names
    extent_boundary_bc_left: str = "zero_current"
    extent_boundary_bc_right: str = "zero_current"


def default_bands() -> List[Band]:
    """Standard two-band semiconductor starting point."""
    return [
        Band(name="CB", type="nondegenerate", sign=-1, extent="all"),
        Band(name="VB", type="nondegenerate", sign=+1, extent="all"),
    ]


# ── Interband process ─────────────────────────────────────────────────────────

@dataclass
class CallableTopHat:
    """One top-hat callable spec attached to a Process for a given region.

    ``region`` is either ``'domain'`` or a layer name.  ``E_low``/``E_high``
    are stored as floats with explicit units; ``E_high`` of ``None`` means
    +infinity.  Multiple ``CallableTopHat`` rows for the same callable
    shorthand are allowed (one per region scope).
    """
    region:     str   = "domain"
    E_low:      float = 0.0
    E_low_unit: str   = "eV"
    E_high:     Optional[float] = None
    E_high_unit: str  = "eV"
    value:      float = 0.0
    unit:       str   = "1/cm"


@dataclass
class Process:
    name: str = ""
    cls: str = "SRHRecombination"
    src_band: str = "VB"
    dst_band: str = "CB"
    trap_band: Optional[str] = None
    radiative_recombination: Optional[bool] = None
    # Shockley-van Roosbroeck quadrature window, used only when
    # radiative_recombination is on.  None means "let the runner default it".
    # The integrand is weighted by exp(-E/kT), which peaks at the absorption
    # threshold: E_min belongs *at* that threshold, or the bin straddling it is
    # counted or dropped whole and the rate is wrong by tens of percent.  E_max
    # ~23 kT higher (0.6 eV at 300 K) then keeps the bins narrow.
    svr_E_min: Optional[float] = None
    svr_E_min_unit: str = "eV"
    svr_E_max: Optional[float] = None
    svr_E_max_unit: str = "eV"
    # Per-callable list of top-hat specs the user has entered in the GUI.
    # Keyed by callable shorthand (e.g. ``'alpha_function'``,
    # ``'sigma_function'``).  Entries are persisted to YAML under the
    # process's ``alpha:`` / ``sigma:`` block as one region→top_hat entry
    # per CallableTopHat instance.
    callable_top_hats: Dict[str, List[CallableTopHat]] = field(default_factory=dict)


# ── Optical field ─────────────────────────────────────────────────────────────

@dataclass
class BlackbodyIntensity:
    T_source: float = 6000.0
    T_source_unit: str = "K"
    concentration: float = 1.0
    E_min: float = 1.42
    E_min_unit: str = "eV"
    E_max: Optional[float] = None   # None = infinity
    E_max_unit: str = "eV"


@dataclass
class ExplicitIntensity:
    value: float = 100.0
    unit: str = "mW/cm^2"
    input_type: str = "power_density"   # "flux" | "power_density"


@dataclass
class OpticalField:
    name: str = "field"
    direction: str = "+x"
    photon_energy: float = 1.42
    photon_energy_unit: str = "eV"
    # intensity is either ExplicitIntensity or BlackbodyIntensity
    intensity: Any = field(default_factory=ExplicitIntensity)


@dataclass
class OpticalFieldSet:
    """A set of N optical fields generated from a spectrum.

    Persisted to / from the YAML ``optical_field_sets:`` block; the runner
    expands it into N monochromatic fields at simulation time (item 7).
    """
    name_prefix: str = "sun"
    direction: str = "+x"
    spectrum: str = "am15g"        # 'am15g' | 'am15d' | 'blackbody'
    N_bins: int = 5
    E_min: float = 0.8
    E_min_unit: str = "eV"
    E_max: float = 4.0
    E_max_unit: str = "eV"
    concentration: float = 1.0
    # Only meaningful for spectrum == 'blackbody'.
    T_source: float = 6000.0
    T_source_unit: str = "K"
    # 'equal_flux' (default) | 'uniform_energy' (runner item 13).
    bin_edges: str = "equal_flux"


# ── Boundary conditions ───────────────────────────────────────────────────────

@dataclass
class Contact:
    role: str = "reference"           # 'swept' or 'reference'
    band_bcs: Dict[str, str] = field(default_factory=dict)  # band_name -> 'ohmic'|'zero_current'


def default_contacts() -> Dict[str, Contact]:
    return {
        "left": Contact(role="swept"),
        "right": Contact(role="reference"),
    }


# ── Interface BCs ─────────────────────────────────────────────────────────────

@dataclass
class InterfaceBCSpec:
    """One BC type applied at an internal interface."""
    type: str = "ThermionicHeterojunction"
    bands: List[str] = field(default_factory=list)
    # Per-band enhancement: band_name → value.  Missing keys default to 1.0.
    HJBC_enhancement: Dict[str, float] = field(default_factory=dict)


@dataclass
class Interface:
    """Internal interface between two adjacent layers.

    `left` and `right` are layer names (Python identifiers) matching the
    Simudo region attribute names: facet = R.<left>.boundary(R.<right>).
    """
    left: str = ""
    right: str = ""
    bcs: List[InterfaceBCSpec] = field(default_factory=list)


# ── Material ──────────────────────────────────────────────────────────────────

@dataclass
class Material:
    name: str = "material"
    source: Optional[str] = None        # None = inline; "library://ClassName" etc.
    properties: Dict[str, Any] = field(default_factory=dict)  # inline key→{value,unit}


# ── Execution profile ────────────────────────────────────────────────────────

@dataclass
class ExecutionProfile:
    """
    Describes how to launch simudo_1d_runner.py.

    type: "local" | "docker" | "ssh" | "ssh+docker"

    The user-level default lives in ~/.simudo_gui.yaml under the key
    "execution".  A per-project override can be stored in the project YAML
    under the same key; it is merged on top of the default at runtime.
    """
    type: str = "docker"

    # ── Docker ────────────────────────────────────────────────────────────────
    docker_container: str = ""
    # Absolute host path that is bind-mounted into the container
    docker_host_root: str = ""
    # The path inside the container that corresponds to docker_host_root
    docker_container_root: str = ""
    # Output of: docker exec <container> python3 -c "import simudo; print(simudo.__file__)"
    # If set, the runner is located relative to this path rather than via the
    # bind-mount translation.  Required when Simudo is pip-installed inside the
    # container rather than mounted from the host.
    docker_simudo_path: str = ""

    # ── SSH (also used as base for ssh+docker) ────────────────────────────────
    ssh_host: str = ""            # .ssh/config alias, or hostname / user@hostname
    ssh_user: str = ""            # leave blank if host spec already includes user
    ssh_identity_file: str = ""   # path to private key; "" = agent / config default
    ssh_remote_work_dir: str = "~/simudo_runs"
    # Output of: python3 -c "import simudo; print(simudo.__file__)" on the remote.
    # Used to derive the remote runner path and bundled-materials path.
    ssh_simudo_path: str = ""
    # If True, the remote run directory is deleted after a successful output sync.
    ssh_delete_remote_after_sync: bool = False

    # ── Common ────────────────────────────────────────────────────────────────
    python_cmd: str = "python3"


# ── Physics ───────────────────────────────────────────────────────────────────

@dataclass
class PhysicsSettings:
    temperature: float = 300.0
    temperature_unit: str = "K"


# ── Simulation ────────────────────────────────────────────────────────────────

@dataclass
class IntensityRampSettings:
    enabled: bool = True
    selfconsistent_optics: bool = True
    step_size: Optional[float] = None


@dataclass
class VoltageSweepSettings:
    enabled: bool = True
    values: List[float] = field(default_factory=list)
    selfconsistent_optics: bool = True
    step_size: Optional[float] = None


@dataclass
class CheckpointStageSettings:
    checkpoint_at_end: bool = True
    at_values: List[float] = field(default_factory=list)


@dataclass
class CheckpointsSettings:
    intensity_ramp: CheckpointStageSettings = field(default_factory=CheckpointStageSettings)
    voltage_sweep: CheckpointStageSettings = field(default_factory=CheckpointStageSettings)
    directory: str = "checkpoints"


@dataclass
class OutputSettings:
    xdmf_mesh: bool = True        # spatially resolved output the GUI can plot
    spatial_profiles: bool = False  # optional CSV copy of the same quantities


@dataclass
class SimulationSettings:
    intensity_ramp: IntensityRampSettings = field(default_factory=IntensityRampSettings)
    voltage_sweep: VoltageSweepSettings = field(default_factory=VoltageSweepSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    checkpoints: CheckpointsSettings = field(default_factory=CheckpointsSettings)
    resume_from: Optional[str] = None


# ── Top-level project ─────────────────────────────────────────────────────────

@dataclass
class Project:
    filepath: Optional[str] = None          # path to the saved YAML file
    description: str = ""
    layers: List[Layer] = field(default_factory=list)
    overlay_regions: List[OverlayRegion] = field(
        default_factory=lambda: [make_domain_overlay()]
    )
    bands: List[Band] = field(default_factory=default_bands)
    processes: List[Process] = field(default_factory=list)
    optical_fields: List[OpticalField] = field(default_factory=list)
    optical_field_sets: List[OpticalFieldSet] = field(default_factory=list)
    contacts: Dict[str, Contact] = field(default_factory=default_contacts)
    physics: PhysicsSettings = field(default_factory=PhysicsSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    custom_eop_files: List[str] = field(default_factory=list)
    output_folder: str = field(default_factory=_default_output_folder)
    mesh_max_edge_length: float = 0.02   # global maximum mesh spacing (µm)
    # 1 = true-1D interval mesh (default); 2 = the 2D one-cell-tall strip that
    # Simudo used for 1D problems before the 1D route existed.
    mesh_dimension: int = 1

    materials: List[Material] = field(default_factory=list)
    interfaces: List[Interface] = field(default_factory=list)
    # Per-project execution profile override (None = use user-level default)
    execution: Optional[ExecutionProfile] = None

    @property
    def band_names(self) -> List[str]:
        return [b.name for b in self.bands]

    @property
    def layer_names(self) -> List[str]:
        return [l.name for l in self.layers]
