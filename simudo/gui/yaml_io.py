"""
Simudo GUI — YAML serialization / deserialization.

Converts between the Project data model and the YAML format consumed by
simudo_1d_runner.py.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional

import yaml

from simudo.gui.model import (
    Band, BlackbodyIntensity, CallableTopHat, CheckpointStageSettings,
    CheckpointsSettings, Contact, ExecutionProfile, ExplicitIntensity,
    Interface, InterfaceBCSpec, IntensityRampSettings,
    Layer, Material, MeshParams, OpticalField, OpticalFieldSet, OverlayRegion,
    PhysicsSettings,
    Process, Project, SimulationSettings, OutputSettings, VoltageSweepSettings,
    default_contacts, default_layer_color, make_domain_overlay,
)


# Map: callable shorthand → YAML block key on the process entry.
# (Convention: strip a trailing '_function' suffix.)
def _callable_yaml_key(shorthand: str) -> str:
    if shorthand.endswith('_function'):
        return shorthand[:-len('_function')]
    return shorthand


# ── Unit-type helpers ─────────────────────────────────────────────────────────

_POWER_DENSITY_UNITS = {
    "mw/cm^2", "w/m^2", "w/cm^2", "mw/m^2", "kw/m^2",
    "suns", "sun", "w cm^-2", "mw cm^-2",
}

def _is_power_density_unit(unit: str) -> bool:
    """Return True if *unit* looks like a power-density (energy-flux) unit."""
    return unit.lower().replace(" ", "") in _POWER_DENSITY_UNITS


# ── Load ──────────────────────────────────────────────────────────────────────

def load_project(path: str) -> Project:
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    proj = Project(filepath=os.path.abspath(path))
    proj.description = data.get("description", "")
    proj.custom_eop_files = data.get("custom_eop_files", [])

    # Layers
    _LAYER_RESERVED = {"name", "thickness", "material", "mesh", "properties"}

    proj.layers = []
    for i, ld in enumerate(data.get("layers", [])):
        mesh_d = ld.get("mesh", {})
        t = ld.get("thickness", {})
        # Merge nested 'properties:' dict with any legacy top-level param keys.
        props: dict = dict(ld.get("properties") or {})
        for k, v in ld.items():
            if k not in _LAYER_RESERVED:
                props[k] = v
        layer = Layer(
            name=ld["name"],
            thickness=t.get("value", 1.0) if isinstance(t, dict) else float(t),
            thickness_unit=t.get("unit", "um") if isinstance(t, dict) else "um",
            material=ld.get("material", ""),
            properties=props,
            mesh=MeshParams(
                start=mesh_d.get("start"),
                factor=mesh_d.get("factor"),
            ),
            color=default_layer_color(i),
        )
        proj.layers.append(layer)

    _OV_RESERVED = {"name", "extent", "type", "properties"}

    # Overlay regions — always start with the domain overlay
    domain_ov = make_domain_overlay()
    proj.overlay_regions = [domain_ov]
    for od in data.get("regions", []):
        extent = od.get("extent", {})
        ov_props: dict = dict(od.get("properties") or {})
        for k, v in od.items():
            if k not in _OV_RESERVED:
                ov_props[k] = v
        if od.get("name") == "domain":
            # Load properties into the built-in domain overlay.  Merge rather
            # than replace, so a project file that omits `temperature` still
            # arrives with the 300 K default that a new project gets -- and
            # shows it in the Layers panel, where it can be changed.  Anything
            # the file does say wins.
            merged = dict(domain_ov.properties)
            merged.update(ov_props)
            domain_ov.properties = merged
            continue
        ov = OverlayRegion(
            name=od["name"],
            properties=ov_props,
        )
        if isinstance(extent, dict):
            ov.start = extent.get("start")
            ov.end = extent.get("end")
            ov.extent_unit = extent.get("unit", "um")
        elif isinstance(extent, list):
            ov.layer_names = extent
        proj.overlay_regions.append(ov)

    # Bands
    proj.bands = []
    for bd in data.get("bands", []):
        proj.bands.append(Band(
            name=bd["name"],
            type=bd.get("type", "nondegenerate"),
            sign=bd.get("sign", -1),
            extent=bd.get("extent", "all"),
        ))

    # Processes
    proj.processes = []
    # Known callable shorthand → YAML block keys.  When more callable types
    # come online we just list their shorthands here; the loop below
    # inspects each process for any of the corresponding YAML keys.
    _CALLABLE_SHORTHANDS = ("alpha_function", "sigma_function")
    for pd in data.get("processes", []):
        proc = Process(
            name=pd["name"],
            cls=pd["class"],
            src_band=pd.get("src_band", ""),
            dst_band=pd.get("dst_band", ""),
            trap_band=pd.get("trap_band"),
            radiative_recombination=pd.get("radiative_recombination"),
        )
        # radiative_recombination may be a bool, or a dict carrying the SVR
        # quadrature window (the runner accepts both forms).
        _rr = pd.get("radiative_recombination")
        if isinstance(_rr, dict):
            proc.radiative_recombination = True
            for _key, _attr in (("E_min", "svr_E_min"), ("E_max", "svr_E_max")):
                _q = _rr.get(_key)
                if isinstance(_q, dict) and _q.get("value") is not None:
                    setattr(proc, _attr, float(_q["value"]))
                    setattr(proc, _attr + "_unit", _q.get("unit", "eV"))
        # Callable top-hat specs (alpha:/sigma: blocks).
        for shorthand in _CALLABLE_SHORTHANDS:
            yml_key = _callable_yaml_key(shorthand)
            block = pd.get(yml_key)
            if not isinstance(block, dict):
                continue
            rows: list = []
            for region_name, spec in block.items():
                if not isinstance(spec, dict) or "top_hat" not in spec:
                    # Non-top-hat specs (constant, energies_eV, ...) are
                    # not yet GUI-editable; preserved as a model annotation
                    # but not surfaced in the process card.
                    continue
                th = spec["top_hat"]
                e_low_raw  = th.get("E_low",  {"value": 0.0, "unit": "eV"})
                e_high_raw = th.get("E_high", "infinity")
                _v = _val(e_high_raw)
                if (e_high_raw in (None, "infinity", ".inf")
                        or (isinstance(_v, float) and _v == float("inf"))):
                    e_high_val: Optional[float] = None
                    e_high_unit = "eV"
                else:
                    e_high_val  = _v
                    e_high_unit = _unit(e_high_raw, "eV")
                rows.append(CallableTopHat(
                    region=region_name,
                    E_low=_val(e_low_raw),
                    E_low_unit=_unit(e_low_raw, "eV"),
                    E_high=e_high_val,
                    E_high_unit=e_high_unit,
                    value=float(th.get("value", 0.0)),
                    unit=str(th.get("unit", "1/cm")),
                ))
            if rows:
                proc.callable_top_hats[shorthand] = rows
        proj.processes.append(proc)

    # Optical fields
    proj.optical_fields = []
    for ofd in data.get("optical_fields", []):
        intensity_d = ofd.get("intensity", {})
        if isinstance(intensity_d, dict) and intensity_d.get("mode") == "blackbody_band":
            e_max_raw = intensity_d.get("E_max")
            e_max = None if e_max_raw in (None, "infinity", float("inf")) else _val(e_max_raw)
            intensity = BlackbodyIntensity(
                T_source=_val(intensity_d.get("T_source", 6000)),
                T_source_unit=_unit(intensity_d.get("T_source", {}), "K"),
                concentration=intensity_d.get("concentration", 1.0),
                E_min=_val(intensity_d.get("E_min", 0)),
                E_min_unit=_unit(intensity_d.get("E_min", {}), "eV"),
                E_max=e_max,
                E_max_unit=_unit(intensity_d.get("E_max", {}), "eV") if e_max is not None else "eV",
            )
        else:
            raw_unit = (_unit(intensity_d, "mW/cm^2")
                        if isinstance(intensity_d, dict) else "mW/cm^2")
            raw_mode = (intensity_d.get("mode", "")
                        if isinstance(intensity_d, dict) else "")
            # "explicit_power_density" is explicit in YAML; also auto-detect
            # from unit for backwards-compat with old files that had mW/cm^2.
            if raw_mode == "explicit_power_density":
                itype = "power_density"
            elif raw_mode in ("explicit", ""):
                itype = ("power_density"
                         if _is_power_density_unit(raw_unit) else "flux")
            else:
                itype = "flux"
            intensity = ExplicitIntensity(
                value=_val(intensity_d) if intensity_d else 0.0,
                unit=raw_unit,
                input_type=itype,
            )
        pe = ofd.get("photon_energy", {})
        proj.optical_fields.append(OpticalField(
            name=ofd["name"],
            direction=ofd.get("direction", "+x"),
            photon_energy=_val(pe),
            photon_energy_unit=_unit(pe, "eV"),
            intensity=intensity,
        ))

    # Optical field sets (spectrum-expanded)
    proj.optical_field_sets = []
    for fsd in data.get("optical_field_sets", []):
        spec = fsd.get("spectrum", "am15g")
        if isinstance(spec, dict):
            spec_type = spec.get("type", "am15g")
            T_raw = spec.get("T_source", {"value": 6000.0, "unit": "K"})
        else:
            spec_type = spec
            T_raw = {"value": 6000.0, "unit": "K"}
        e_min = fsd.get("E_min", {"value": 0.8, "unit": "eV"})
        e_max = fsd.get("E_max", {"value": 4.0, "unit": "eV"})
        conc = fsd.get("concentration",
                       spec.get("concentration", 1.0) if isinstance(spec, dict) else 1.0)
        proj.optical_field_sets.append(OpticalFieldSet(
            name_prefix=fsd.get("name_prefix", "sun"),
            direction=fsd.get("direction", "+x"),
            spectrum=spec_type,
            N_bins=int(fsd.get("N_bins", 5)),
            E_min=_val(e_min), E_min_unit=_unit(e_min, "eV"),
            E_max=_val(e_max), E_max_unit=_unit(e_max, "eV"),
            concentration=float(conc),
            T_source=_val(T_raw), T_source_unit=_unit(T_raw, "K"),
            bin_edges=fsd.get("bin_edges", "equal_flux"),
        ))

    # Contacts
    proj.contacts = default_contacts()
    for side, cd in data.get("contacts", {}).items():
        c = Contact(role=cd.get("role", "reference"))
        c.band_bcs = dict(cd.get("bands", {}))
        proj.contacts[side] = c

    # Materials
    proj.materials = []
    for mat_name, mat_def in data.get("materials", {}).items():
        if isinstance(mat_def, dict) and "source" in mat_def:
            # Library-ref or derived (derived also has "properties")
            overrides = dict(mat_def.get("properties") or {})
            proj.materials.append(Material(
                name=mat_name,
                source=mat_def["source"],
                properties=overrides,
            ))
        else:
            proj.materials.append(Material(name=mat_name, properties=dict(mat_def or {})))

    # Interfaces
    proj.interfaces = []
    for iface_def in data.get("interfaces", []):
        bcs = []
        for bc_def in iface_def.get("bcs", []):
            bands_list = list(bc_def.get("bands", []))
            enh_raw = bc_def.get("HJBC_enhancement", {})
            if isinstance(enh_raw, dict):
                enh_dict = {k: float(v) for k, v in enh_raw.items()}
            else:
                # Backwards compat: scalar → same value for every listed band
                scalar = float(enh_raw) if enh_raw else 1.0
                enh_dict = {b: scalar for b in bands_list}
            bcs.append(InterfaceBCSpec(
                type=bc_def.get("type", "ThermionicHeterojunction"),
                bands=bands_list,
                HJBC_enhancement=enh_dict,
            ))
        proj.interfaces.append(Interface(
            left=iface_def.get("left", ""),
            right=iface_def.get("right", ""),
            bcs=bcs,
        ))

    # Physics (temperature is now a spatial domain property, not a physics field)
    proj.physics = PhysicsSettings()

    # Simulation
    sim = data.get("simulation", {})
    ir = sim.get("intensity_ramp", {})
    vs = sim.get("voltage_sweep", {})
    out = sim.get("output", {})
    ck = sim.get("checkpoints", {})
    proj.simulation = SimulationSettings(
        intensity_ramp=IntensityRampSettings(
            enabled=ir.get("enabled", True),
            selfconsistent_optics=ir.get("selfconsistent_optics", True),
            step_size=ir.get("step_size"),
        ),
        voltage_sweep=VoltageSweepSettings(
            enabled=vs.get("enabled", True),
            values=list(vs.get("values", [])),
            selfconsistent_optics=vs.get("selfconsistent_optics", True),
            step_size=vs.get("step_size"),
        ),
        output=OutputSettings(
            spatial_profiles=out.get("spatial_profiles", True),
            xdmf_mesh=out.get("xdmf_mesh", True),
        ),
        checkpoints=CheckpointsSettings(
            intensity_ramp=_ck_stage(ck.get("intensity_ramp", {})),
            voltage_sweep=_ck_stage(ck.get("voltage_sweep", {})),
            directory=ck.get("directory", "checkpoints"),
        ),
        resume_from=sim.get("resume_from"),
    )

    proj.output_folder = data.get("output", {}).get("folder", "out")
    proj.mesh_max_edge_length = float(data.get("mesh", {}).get("max_edge_length", 0.02))

    # Execution profile (per-project override — merged with user default in app.py)
    if "execution" in data:
        proj.execution = _load_execution_profile(data["execution"])

    return proj


def _val(x) -> float:
    if isinstance(x, dict):
        return float(x.get("value", 0.0))
    if x is None:
        return 0.0
    return float(x)


def _unit(x, default: str = "") -> str:
    if isinstance(x, dict):
        return x.get("unit", default)
    return default


def _ck_stage(d: dict) -> CheckpointStageSettings:
    return CheckpointStageSettings(
        checkpoint_at_end=d.get("checkpoint_at_end", True),
        at_values=list(d.get("at_values", [])),
    )


# ── Save ──────────────────────────────────────────────────────────────────────

def save_project(proj: Project, path: Optional[str] = None) -> None:
    path = path or proj.filepath
    if not path:
        raise ValueError("No file path set for project")
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = project_to_dict(proj)
    with open(path, "w", encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    proj.filepath = path


def project_to_dict(proj: Project) -> dict:
    d: Dict[str, Any] = {}

    if proj.description:
        d["description"] = proj.description

    if proj.custom_eop_files:
        d["custom_eop_files"] = proj.custom_eop_files

    # Layers
    d["layers"] = []
    for layer in proj.layers:
        ld: Dict[str, Any] = {
            "name": layer.name,
            "thickness": {"value": layer.thickness, "unit": layer.thickness_unit},
        }
        if layer.material:
            ld["material"] = layer.material
        mesh_d: Dict[str, Any] = {}
        if layer.mesh.start is not None:
            mesh_d["start"] = layer.mesh.start
        if layer.mesh.factor is not None:
            mesh_d["factor"] = layer.mesh.factor
        if mesh_d:
            ld["mesh"] = mesh_d
        if layer.properties:
            ld["properties"] = layer.properties
        d["layers"].append(ld)

    # Overlay regions — domain is written only when it has properties
    regions_out = []
    for ov in proj.overlay_regions:
        if ov.is_domain:
            if ov.properties:
                regions_out.append({"name": "domain", "properties": ov.properties})
        else:
            od: Dict[str, Any] = {"name": ov.name, "type": "overlay"}
            if ov.start is not None and ov.end is not None:
                od["extent"] = {"start": ov.start, "end": ov.end, "unit": ov.extent_unit}
            elif ov.layer_names:
                od["extent"] = ov.layer_names
            if ov.properties:
                od["properties"] = ov.properties
            regions_out.append(od)
    if regions_out:
        d["regions"] = regions_out

    # Bands
    if proj.bands:
        d["bands"] = []
        for band in proj.bands:
            bd: Dict[str, Any] = {
                "name": band.name,
                "type": band.type,
                "sign": band.sign,
            }
            if band.extent != "all":
                bd["extent"] = band.extent
            else:
                bd["extent"] = "all"
            d["bands"].append(bd)

    # Processes
    if proj.processes:
        d["processes"] = []
        for proc in proj.processes:
            pd: Dict[str, Any] = {
                "class": proc.cls,
                "name": proc.name,
                "dst_band": proc.dst_band,
                "src_band": proc.src_band,
            }
            if proc.trap_band:
                pd["trap_band"] = proc.trap_band
            if proc.radiative_recombination is not None:
                if (proc.radiative_recombination
                        and (proc.svr_E_min is not None
                             or proc.svr_E_max is not None)):
                    _rr = {}
                    if proc.svr_E_min is not None:
                        _rr["E_min"] = {"value": float(proc.svr_E_min),
                                        "unit": proc.svr_E_min_unit or "eV"}
                    if proc.svr_E_max is not None:
                        _rr["E_max"] = {"value": float(proc.svr_E_max),
                                        "unit": proc.svr_E_max_unit or "eV"}
                    pd["radiative_recombination"] = _rr
                else:
                    pd["radiative_recombination"] = proc.radiative_recombination
            # Callable top-hat specs → alpha:/sigma: YAML blocks.
            for shorthand, rows in (proc.callable_top_hats or {}).items():
                if not rows:
                    continue
                yml_key = _callable_yaml_key(shorthand)
                block: Dict[str, Any] = {}
                for row in rows:
                    e_low = {"value": row.E_low, "unit": row.E_low_unit}
                    # E_high=None means +∞.  Emit the {value: .inf, unit} dict
                    # form the runner's parse_quantity understands (a bare
                    # 'infinity' string would not parse).
                    if row.E_high is None:
                        e_high: Any = {"value": float("inf"), "unit": row.E_high_unit}
                    else:
                        e_high = {"value": row.E_high, "unit": row.E_high_unit}
                    block[row.region] = {
                        "top_hat": {
                            "E_low":  e_low,
                            "E_high": e_high,
                            "value":  row.value,
                            "unit":   row.unit,
                        }
                    }
                pd[yml_key] = block
            d["processes"].append(pd)

    # Optical fields
    if proj.optical_fields:
        d["optical_fields"] = []
        for of in proj.optical_fields:
            ofd: Dict[str, Any] = {
                "name": of.name,
                "direction": of.direction,
                "photon_energy": {"value": of.photon_energy, "unit": of.photon_energy_unit},
            }
            if isinstance(of.intensity, BlackbodyIntensity):
                i = of.intensity
                e_max = "infinity" if i.E_max is None else {"value": i.E_max, "unit": i.E_max_unit}
                ofd["intensity"] = {
                    "mode": "blackbody_band",
                    "T_source": {"value": i.T_source, "unit": i.T_source_unit},
                    "concentration": i.concentration,
                    "E_min": {"value": i.E_min, "unit": i.E_min_unit},
                    "E_max": e_max,
                }
            else:
                i = of.intensity
                idict: Dict[str, Any] = {"value": i.value, "unit": i.unit}
                if i.input_type == "power_density":
                    idict["mode"] = "explicit_power_density"
                ofd["intensity"] = idict
            d["optical_fields"].append(ofd)

    # Optical field sets
    if proj.optical_field_sets:
        d["optical_field_sets"] = []
        for fs in proj.optical_field_sets:
            fsd: Dict[str, Any] = {
                "name_prefix": fs.name_prefix,
                "direction": fs.direction,
                "N_bins": fs.N_bins,
                "E_min": {"value": fs.E_min, "unit": fs.E_min_unit},
                "E_max": {"value": fs.E_max, "unit": fs.E_max_unit},
                "concentration": fs.concentration,
            }
            if fs.spectrum == "blackbody":
                fsd["spectrum"] = {
                    "type": "blackbody",
                    "T_source": {"value": fs.T_source, "unit": fs.T_source_unit},
                }
            else:
                fsd["spectrum"] = fs.spectrum
            if fs.bin_edges and fs.bin_edges != "equal_flux":
                fsd["bin_edges"] = fs.bin_edges
            d["optical_field_sets"].append(fsd)

    # Materials
    if proj.materials:
        d["materials"] = {}
        for mat in proj.materials:
            if mat.source:
                entry: Dict[str, Any] = {"source": mat.source}
                if mat.properties:  # derived material — include overrides
                    entry["properties"] = dict(mat.properties)
                d["materials"][mat.name] = entry
            else:
                d["materials"][mat.name] = dict(mat.properties)

    # Interfaces
    if proj.interfaces:
        d["interfaces"] = []
        for iface in proj.interfaces:
            bcs_out = []
            for bc in iface.bcs:
                if bc.bands:  # only write specs that have ≥1 band
                    # Write per-band enhancement dict; omit bands with default (1.0)
                    # to keep YAML compact, but always write if any differ.
                    enh_dict = {b: bc.HJBC_enhancement.get(b, 1.0) for b in bc.bands}
                    bcs_out.append({
                        "type": bc.type,
                        "bands": list(bc.bands),
                        "HJBC_enhancement": enh_dict,
                    })
            if bcs_out:
                d["interfaces"].append({
                    "left": iface.left,
                    "right": iface.right,
                    "bcs": bcs_out,
                })

    # Contacts — always write all band BCs explicitly so the runner
    # doesn't have to guess defaults.
    d["contacts"] = {}
    for side, contact in proj.contacts.items():
        cd: Dict[str, Any] = {"role": contact.role}
        bands_out = {}
        for band in proj.bands:
            bc = contact.band_bcs.get(band.name, "ohmic")
            bands_out[band.name] = bc
        if bands_out:
            cd["bands"] = bands_out
        d["contacts"][side] = cd

    # Simulation
    ir = proj.simulation.intensity_ramp
    vs = proj.simulation.voltage_sweep
    out = proj.simulation.output
    ck = proj.simulation.checkpoints
    sim: Dict[str, Any] = {
        "intensity_ramp": _filter_none({
            "enabled": ir.enabled,
            "selfconsistent_optics": ir.selfconsistent_optics,
            "step_size": ir.step_size,
        }),
        "voltage_sweep": _filter_none({
            "enabled": vs.enabled,
            "values": vs.values,
            "selfconsistent_optics": vs.selfconsistent_optics,
            "step_size": vs.step_size,
        }),
        "output": {
            "spatial_profiles": out.spatial_profiles,
            "xdmf_mesh": out.xdmf_mesh,
        },
        "checkpoints": {
            "intensity_ramp": _ck_stage_to_dict(ck.intensity_ramp),
            "voltage_sweep": _ck_stage_to_dict(ck.voltage_sweep),
            "directory": ck.directory,
        },
    }
    if proj.simulation.resume_from:
        sim["resume_from"] = proj.simulation.resume_from
    d["simulation"] = sim

    d["output"] = {"folder": proj.output_folder or "out"}
    d["mesh"] = {"max_edge_length": proj.mesh_max_edge_length}

    # Per-project execution override (only saved if set)
    if proj.execution is not None:
        d["execution"] = _execution_profile_to_dict(proj.execution)

    return d


def _load_execution_profile(d: dict) -> ExecutionProfile:
    return ExecutionProfile(
        type=d.get("type", "docker"),
        docker_container=d.get("docker_container", ""),
        docker_host_root=d.get("docker_host_root", ""),
        docker_container_root=d.get("docker_container_root", ""),
        docker_simudo_path=d.get("docker_simudo_path", ""),
        ssh_host=d.get("ssh_host", ""),
        ssh_user=d.get("ssh_user", ""),
        ssh_identity_file=d.get("ssh_identity_file", ""),
        ssh_remote_work_dir=d.get("ssh_remote_work_dir", "~/simudo_runs"),
        ssh_simudo_path=d.get("ssh_simudo_path", ""),
        ssh_delete_remote_after_sync=d.get("ssh_delete_remote_after_sync", False),
        python_cmd=d.get("python_cmd", "python3"),
    )


def _execution_profile_to_dict(ep: ExecutionProfile) -> dict:
    d: Dict[str, Any] = {"type": ep.type, "python_cmd": ep.python_cmd}
    if ep.type in ("docker", "ssh+docker"):
        d["docker_container"] = ep.docker_container
        d["docker_host_root"] = ep.docker_host_root
        d["docker_container_root"] = ep.docker_container_root
        if ep.docker_simudo_path:
            d["docker_simudo_path"] = ep.docker_simudo_path
    if ep.type in ("ssh", "ssh+docker"):
        d["ssh_host"] = ep.ssh_host
        if ep.ssh_user:
            d["ssh_user"] = ep.ssh_user
        if ep.ssh_identity_file:
            d["ssh_identity_file"] = ep.ssh_identity_file
        d["ssh_remote_work_dir"] = ep.ssh_remote_work_dir
        if ep.ssh_simudo_path:
            d["ssh_simudo_path"] = ep.ssh_simudo_path
        if ep.ssh_delete_remote_after_sync:
            d["ssh_delete_remote_after_sync"] = True
    return d


def _filter_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _ck_stage_to_dict(s: CheckpointStageSettings) -> dict:
    return {
        "checkpoint_at_end": s.checkpoint_at_end,
        "at_values": s.at_values,
    }
