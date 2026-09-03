"""
Simudo GUI — batch parameter sweep.

Data model, sidecar I/O, sub-YAML generation, and parallel launcher for
running a set of sub-simulations that vary one or more project parameters.

Sidecar file: ``<project_stem>.batch.yaml`` (next to the project YAML).

Sub-YAML layout::

    <output_folder>/0/<project>.yaml   (output.folder = ".")
    <output_folder>/1/<project>.yaml
    ...
"""

from __future__ import annotations

import copy
import itertools
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import yaml


# ── Directory helpers (mirrors simudo_1d_runner._increment_dir / ensure_new_dir) ──

def _increment_dir(cur_dir: str) -> str:
    """Increment the last alphanumeric character of a path string."""
    last_pos = -1
    for i, c in reversed(list(enumerate(cur_dir))):
        if c.isalnum():
            last_pos = i
            last_c = c
            break
    if last_pos == -1:
        return cur_dir + "_1"
    if last_c == "z":
        return cur_dir[:last_pos] + "za" + cur_dir[last_pos + 1:]
    elif last_c == "Z":
        return cur_dir[:last_pos] + "ZA" + cur_dir[last_pos + 1:]
    elif last_c.isdigit():
        numbers = re.findall(r"\d+", cur_dir)
        if numbers:
            last_num = int(numbers[-1])
            last_num_pos = cur_dir.rfind(numbers[-1])
            num_len = len(numbers[-1])
            return (cur_dir[:last_num_pos]
                    + str(last_num + 1)
                    + cur_dir[last_num_pos + num_len:])
        return cur_dir
    else:
        return cur_dir[:last_pos] + chr(ord(last_c) + 1) + cur_dir[last_pos + 1:]


def ensure_new_dir(cur_dir: str) -> str:
    """Return cur_dir (or an incremented variant) that does not yet exist."""
    while os.path.isdir(cur_dir):
        cur_dir = _increment_dir(cur_dir)
    return cur_dir


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class BatchTarget:
    """Identifies a single numeric parameter in the project YAML tree."""
    kind: str           # "layer_thickness" | "layer_property" | "overlay_property"
                        # | "optical_field_photon_energy" | "optical_field_intensity_value"
                        # | "optical_field_bb_T_source" | "optical_field_bb_concentration"
                        # | "optical_field_bb_E_min" | "optical_field_bb_E_max"
    container_name: str # layer name, overlay region name, or optical field name
    key: str = ""       # property key (empty for scalar targets)
    unit: str = ""      # default unit used when creating a property that does not exist yet

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "container_name": self.container_name, "key": self.key}
        if self.unit:
            d["unit"] = self.unit
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BatchTarget":
        return cls(
            kind=d["kind"],
            container_name=d["container_name"],
            key=d.get("key", ""),
            unit=d.get("unit", ""),
        )


@dataclass
class BatchVariation:
    """One dimension of the parameter sweep: which parameter and what values."""
    label: str
    target: BatchTarget
    values: List[float]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "target": self.target.to_dict(),
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BatchVariation":
        return cls(
            label=d.get("label", ""),
            target=BatchTarget.from_dict(d["target"]),
            values=[float(v) for v in d.get("values", [])],
        )


@dataclass
class BatchSpec:
    mode: str = "zip"           # "zip" (paired) or "cartesian" (product)
    max_parallel: int = 1
    variations: List[BatchVariation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "max_parallel": self.max_parallel,
            "variations": [v.to_dict() for v in self.variations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BatchSpec":
        return cls(
            mode=d.get("mode", "zip"),
            max_parallel=int(d.get("max_parallel", 1)),
            variations=[BatchVariation.from_dict(vd) for vd in d.get("variations", [])],
        )

    def run_count(self) -> int:
        """Number of sub-runs this spec will produce."""
        if not self.variations:
            return 0
        if self.mode == "zip":
            return min(len(v.values) for v in self.variations)
        else:
            total = 1
            for v in self.variations:
                total *= len(v.values)
            return total


# ── Sidecar I/O ───────────────────────────────────────────────────────────────

def sidecar_path(project_yaml_path: str) -> str:
    stem = os.path.splitext(os.path.abspath(project_yaml_path))[0]
    return stem + ".batch.yaml"


def load_batch_spec(project_yaml_path: str) -> Optional[BatchSpec]:
    path = sidecar_path(project_yaml_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return BatchSpec.from_dict(data)
    except Exception:
        return None


def save_batch_spec(project_yaml_path: str, spec: BatchSpec) -> None:
    path = sidecar_path(project_yaml_path)
    with open(path, "w", encoding='utf-8') as f:
        yaml.dump(
            spec.to_dict(), f,
            default_flow_style=False, allow_unicode=True, sort_keys=False,
        )


# ── Parameter enumeration ─────────────────────────────────────────────────────

def enumerate_sweep_params(project) -> List[Tuple[str, BatchTarget]]:
    """Return all numeric parameters in the project that can be swept.

    Returns a list of (display_label, BatchTarget) pairs.
    """
    result: List[Tuple[str, BatchTarget]] = []

    for layer in project.layers:
        result.append((
            f"{layer.name} / thickness ({layer.thickness_unit})",
            BatchTarget(kind="layer_thickness", container_name=layer.name, key=""),
        ))
        for key, val in layer.properties.items():
            if isinstance(val, dict) and "value" in val:
                result.append((
                    f"{layer.name} / {key}",
                    BatchTarget(kind="layer_property", container_name=layer.name, key=key),
                ))
        # Per-layer mesh parameters
        result.append((
            f"{layer.name} / mesh / start spacing (µm)",
            BatchTarget(kind="layer_mesh_start", container_name=layer.name, key=""),
        ))
        result.append((
            f"{layer.name} / mesh / mesh growth factor",
            BatchTarget(kind="layer_mesh_factor", container_name=layer.name, key=""),
        ))

    for ov in project.overlay_regions:
        for key, val in ov.properties.items():
            if isinstance(val, dict) and "value" in val:
                result.append((
                    f"{ov.name} / {key}",
                    BatchTarget(kind="overlay_property", container_name=ov.name, key=key),
                ))

    # Global mesh parameter
    result.append((
        "global / mesh max edge length (µm)",
        BatchTarget(kind="mesh_max_edge_length", container_name="", key=""),
    ))

    # ── Optical fields ────────────────────────────────────────────────────────
    for of in project.optical_fields:
        fn = of.name
        result.append((
            f"optical / {fn} / photon energy ({of.photon_energy_unit})",
            BatchTarget(kind="optical_field_photon_energy",
                        container_name=fn, unit=of.photon_energy_unit),
        ))
        from simudo.gui.model import BlackbodyIntensity, ExplicitIntensity
        if isinstance(of.intensity, ExplicitIntensity):
            i = of.intensity
            result.append((
                f"optical / {fn} / intensity value ({i.unit})",
                BatchTarget(kind="optical_field_intensity_value",
                            container_name=fn, unit=i.unit),
            ))
        elif isinstance(of.intensity, BlackbodyIntensity):
            i = of.intensity
            result.append((
                f"optical / {fn} / T_source ({i.T_source_unit})",
                BatchTarget(kind="optical_field_bb_T_source",
                            container_name=fn, unit=i.T_source_unit),
            ))
            result.append((
                f"optical / {fn} / concentration (× f_s)",
                BatchTarget(kind="optical_field_bb_concentration",
                            container_name=fn),
            ))
            result.append((
                f"optical / {fn} / E_min ({i.E_min_unit})",
                BatchTarget(kind="optical_field_bb_E_min",
                            container_name=fn, unit=i.E_min_unit),
            ))
            if i.E_max is not None:
                result.append((
                    f"optical / {fn} / E_max ({i.E_max_unit})",
                    BatchTarget(kind="optical_field_bb_E_max",
                                container_name=fn, unit=i.E_max_unit),
                ))

    # ── Contact SRV parameters ────────────────────────────────────────────────
    for side, contact in project.contacts.items():
        for band_name, bc_spec in contact.band_bcs.items():
            if isinstance(bc_spec, dict) and bc_spec.get("type") == "surface_recombination":
                srv = bc_spec.get("srv", {})
                unit = srv.get("unit", "cm/s") if isinstance(srv, dict) else "cm/s"
                result.append((
                    f"contact / {side} / {band_name} / SRV ({unit})",
                    BatchTarget(kind="contact_srv",
                                container_name=f"{side}/{band_name}",
                                unit=unit),
                ))

    return result


def get_current_param_value(project, target: BatchTarget) -> str:
    """Return a human-readable 'current value + unit' string for a BatchTarget.

    Returns an empty string if the parameter has no value set.
    """
    kind = target.kind

    if kind == "layer_thickness":
        for lyr in project.layers:
            if lyr.name == target.container_name:
                return f"{lyr.thickness} {lyr.thickness_unit}"

    elif kind == "layer_property":
        for lyr in project.layers:
            if lyr.name == target.container_name:
                v = lyr.properties.get(target.key)
                if isinstance(v, dict):
                    unit = v.get("unit", "")
                    return (f"{v.get('value', '')} {unit}".strip())

    elif kind == "overlay_property":
        for ov in project.overlay_regions:
            if ov.name == target.container_name:
                v = ov.properties.get(target.key)
                if isinstance(v, dict):
                    unit = v.get("unit", "")
                    return (f"{v.get('value', '')} {unit}".strip())

    elif kind == "layer_mesh_start":
        for lyr in project.layers:
            if lyr.name == target.container_name:
                if lyr.mesh.start is not None:
                    return f"{lyr.mesh.start} µm"
        return "(not set)"

    elif kind == "layer_mesh_factor":
        for lyr in project.layers:
            if lyr.name == target.container_name:
                if lyr.mesh.factor is not None:
                    return f"{lyr.mesh.factor}"
        return "(not set)"

    elif kind == "mesh_max_edge_length":
        return f"{project.mesh_max_edge_length} µm"

    elif kind in ("optical_field_photon_energy", "optical_field_intensity_value",
                  "optical_field_bb_T_source", "optical_field_bb_concentration",
                  "optical_field_bb_E_min", "optical_field_bb_E_max"):
        from simudo.gui.model import BlackbodyIntensity, ExplicitIntensity
        for of in project.optical_fields:
            if of.name != target.container_name:
                continue
            if kind == "optical_field_photon_energy":
                return f"{of.photon_energy} {of.photon_energy_unit}"
            i = of.intensity
            if kind == "optical_field_intensity_value" and isinstance(i, ExplicitIntensity):
                return f"{i.value} {i.unit}"
            if kind == "optical_field_bb_T_source" and isinstance(i, BlackbodyIntensity):
                return f"{i.T_source} {i.T_source_unit}"
            if kind == "optical_field_bb_concentration" and isinstance(i, BlackbodyIntensity):
                return f"{i.concentration}"
            if kind == "optical_field_bb_E_min" and isinstance(i, BlackbodyIntensity):
                return f"{i.E_min} {i.E_min_unit}"
            if kind == "optical_field_bb_E_max" and isinstance(i, BlackbodyIntensity):
                return f"{i.E_max} {i.E_max_unit}" if i.E_max is not None else "∞"

    elif kind == "contact_srv":
        side, band_name = target.container_name.split("/", 1)
        contact = project.contacts.get(side)
        if contact:
            bc_spec = contact.band_bcs.get(band_name, {})
            if isinstance(bc_spec, dict):
                srv = bc_spec.get("srv", {})
                if isinstance(srv, dict):
                    val = srv.get("value", "")
                    unit = srv.get("unit", "cm/s")
                    return f"{val} {unit}".strip()

    return ""


# ── YAML manipulation ─────────────────────────────────────────────────────────

def _apply_variation_to_dict(yaml_dict: dict, target: BatchTarget, value: float) -> None:
    """Modify yaml_dict in-place to update a single parameter."""
    if target.kind == "layer_thickness":
        for layer in yaml_dict.get("layers", []):
            if layer.get("name") == target.container_name:
                t = layer.get("thickness", {})
                unit = t.get("unit", "um") if isinstance(t, dict) else "um"
                layer["thickness"] = {"value": value, "unit": unit}
                return

    elif target.kind == "layer_property":
        for layer in yaml_dict.get("layers", []):
            if layer.get("name") == target.container_name:
                props = layer.setdefault("properties", {})
                existing = props.get(target.key, {})
                unit = existing.get("unit", "") if isinstance(existing, dict) else ""
                props[target.key] = {"value": value, "unit": unit}
                return

    elif target.kind == "overlay_property":
        regions = yaml_dict.setdefault("regions", [])
        for region in regions:
            if region.get("name") == target.container_name:
                props = region.setdefault("properties", {})
                existing = props.get(target.key)
                if isinstance(existing, dict) and existing.get("unit"):
                    unit = existing["unit"]
                else:
                    unit = target.unit
                props[target.key] = {"value": value, "unit": unit}
                return
        # Region not found — create it (handles domain overlay and new overlays)
        new_region: dict = {"name": target.container_name, "properties": {}}
        if target.container_name != "domain":
            new_region["type"] = "overlay"
        new_region["properties"][target.key] = {"value": value, "unit": target.unit}
        regions.append(new_region)

    elif target.kind == "layer_mesh_start":
        for layer in yaml_dict.get("layers", []):
            if layer.get("name") == target.container_name:
                layer.setdefault("mesh", {})["start"] = value
                return

    elif target.kind == "layer_mesh_factor":
        for layer in yaml_dict.get("layers", []):
            if layer.get("name") == target.container_name:
                layer.setdefault("mesh", {})["factor"] = value
                return

    elif target.kind == "mesh_max_edge_length":
        yaml_dict.setdefault("mesh", {})["max_edge_length"] = value

    elif target.kind == "contact_srv":
        side, band_name = target.container_name.split("/", 1)
        contacts = yaml_dict.setdefault("contacts", {})
        contact = contacts.setdefault(side, {})
        bands = contact.setdefault("bands", {})
        existing = bands.get(band_name, {})
        if isinstance(existing, dict) and existing.get("type") == "surface_recombination":
            unit = existing.get("srv", {}).get("unit", target.unit or "cm/s")
        else:
            unit = target.unit or "cm/s"
        bands[band_name] = {
            "type": "surface_recombination",
            "srv": {"value": value, "unit": unit},
        }

    elif target.kind in ("optical_field_photon_energy", "optical_field_intensity_value",
                         "optical_field_bb_T_source", "optical_field_bb_concentration",
                         "optical_field_bb_E_min", "optical_field_bb_E_max"):
        for field in yaml_dict.get("optical_fields", []):
            if field.get("name") != target.container_name:
                continue
            k = target.kind
            if k == "optical_field_photon_energy":
                pe = field.setdefault("photon_energy", {})
                if isinstance(pe, dict):
                    pe["value"] = value
                else:
                    field["photon_energy"] = {"value": value, "unit": target.unit or "eV"}
            elif k == "optical_field_intensity_value":
                i = field.setdefault("intensity", {})
                if isinstance(i, dict):
                    i["value"] = value
                else:
                    field["intensity"] = {"value": value, "unit": target.unit or "mW/cm^2"}
            elif k == "optical_field_bb_T_source":
                i = field.setdefault("intensity", {})
                ts = i.get("T_source", {})
                if isinstance(ts, dict):
                    ts["value"] = value
                    i["T_source"] = ts
                else:
                    i["T_source"] = {"value": value, "unit": target.unit or "K"}
            elif k == "optical_field_bb_concentration":
                i = field.setdefault("intensity", {})
                i["concentration"] = value
            elif k == "optical_field_bb_E_min":
                i = field.setdefault("intensity", {})
                em = i.get("E_min", {})
                if isinstance(em, dict):
                    em["value"] = value
                    i["E_min"] = em
                else:
                    i["E_min"] = {"value": value, "unit": target.unit or "eV"}
            elif k == "optical_field_bb_E_max":
                i = field.setdefault("intensity", {})
                em = i.get("E_max", {})
                if isinstance(em, dict):
                    em["value"] = value
                    i["E_max"] = em
                elif em in (None, "infinity"):
                    i["E_max"] = {"value": value, "unit": target.unit or "eV"}
                else:
                    i["E_max"] = {"value": value, "unit": target.unit or "eV"}
            return


def build_combinations(spec: BatchSpec) -> List[List[Tuple[BatchTarget, float]]]:
    """Return one list of (target, value) pairs per sub-run."""
    if not spec.variations:
        return []
    if spec.mode == "zip":
        n = min(len(v.values) for v in spec.variations)
        return [
            [(var.target, var.values[i]) for var in spec.variations]
            for i in range(n)
        ]
    else:  # cartesian
        value_lists = [var.values for var in spec.variations]
        targets = [var.target for var in spec.variations]
        combos = []
        for vals in itertools.product(*value_lists):
            combos.append(list(zip(targets, vals)))
        return combos


# ── Sub-YAML generation ───────────────────────────────────────────────────────


def compact_param_label(target: BatchTarget, value: float) -> str:
    """Return a short 'name=value' string for a single (target, value) pair."""
    kind = target.kind
    if kind == "layer_thickness":
        return f"{target.container_name}={value:.4g}"
    elif kind == "layer_property":
        key_short = target.key.split("/")[-1]
        return f"{target.container_name}/{key_short}={value:.4g}"
    elif kind == "overlay_property":
        key_short = target.key.split("/")[-1]
        return f"{target.container_name}/{key_short}={value:.4g}"
    elif kind == "layer_mesh_start":
        return f"{target.container_name}/mesh.start={value:.4g}"
    elif kind == "layer_mesh_factor":
        return f"{target.container_name}/mesh.factor={value:.4g}"
    elif kind == "mesh_max_edge_length":
        return f"mesh.max={value:.4g}"
    elif kind == "contact_srv":
        side, band_name = target.container_name.split("/", 1)
        return f"{side}/{band_name}/srv={value:.4g}"
    elif kind == "optical_field_photon_energy":
        return f"{target.container_name}/E_ph={value:.4g}"
    elif kind == "optical_field_intensity_value":
        return f"{target.container_name}/I={value:.4g}"
    elif kind == "optical_field_bb_T_source":
        return f"{target.container_name}/T={value:.4g}"
    elif kind == "optical_field_bb_concentration":
        return f"{target.container_name}/conc={value:.4g}"
    elif kind == "optical_field_bb_E_min":
        return f"{target.container_name}/E_min={value:.4g}"
    elif kind == "optical_field_bb_E_max":
        return f"{target.container_name}/E_max={value:.4g}"
    return f"{value:.4g}"

def generate_sub_yamls(project_yaml_path: str, spec: BatchSpec) -> List[str]:
    """Create numbered sub-directories and write a modified project YAML into each.

    Sub-YAMLs use output.folder = "." so runner output goes alongside them.
    Returns the list of absolute sub-YAML paths.
    """
    with open(project_yaml_path, encoding='utf-8') as f:
        base_dict = yaml.safe_load(f)

    proj_dir = os.path.dirname(os.path.abspath(project_yaml_path))
    yaml_stem = os.path.splitext(os.path.basename(project_yaml_path))[0]

    output_folder = base_dict.get("output", {}).get("folder", "out")
    if not os.path.isabs(output_folder):
        base_out_dir = os.path.join(proj_dir, output_folder)
    else:
        base_out_dir = output_folder

    # Find the next unused directory so batch runs never overwrite existing output.
    base_out_dir = ensure_new_dir(base_out_dir)

    combinations = build_combinations(spec)
    sub_yaml_paths: List[str] = []

    for i, combo in enumerate(combinations):
        sub_dir = os.path.join(base_out_dir, str(i))
        os.makedirs(sub_dir, exist_ok=True)

        sub_dict = copy.deepcopy(base_dict)
        sub_dict.setdefault("output", {})["folder"] = "."

        for target, value in combo:
            _apply_variation_to_dict(sub_dict, target, value)

        sub_yaml_path = os.path.join(sub_dir, f"{yaml_stem}.yaml")
        with open(sub_yaml_path, "w", encoding='utf-8') as f:
            yaml.dump(
                sub_dict, f,
                default_flow_style=False, allow_unicode=True, sort_keys=False,
            )
        sub_yaml_paths.append(sub_yaml_path)

    return sub_yaml_paths


# ── Batch launcher ─────────────────────────────────────────────────────────────

class BatchLauncher:
    """Launches sub-runs in parallel, bounded by a semaphore."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def launch_all(
        self,
        sub_yaml_paths: List[str],
        spec: BatchSpec,
        backend,
        runner_host: str,
        lib_dirs: List[str],
        log_cb: Callable[[str], None],
        status_cb: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        """Start a background supervisor thread that manages all sub-runs.

        status_cb(idx, status) is called from background threads whenever a
        run changes state.  status is one of: "queued", "running", "done",
        "failed", "skipped".
        """
        self._stop_event.clear()
        t = threading.Thread(
            target=self._supervise,
            args=(sub_yaml_paths, spec, backend, runner_host, lib_dirs,
                  log_cb, status_cb),
            daemon=True,
        )
        t.start()

    def _supervise(
        self,
        sub_yaml_paths: List[str],
        spec: BatchSpec,
        backend,
        runner_host: str,
        lib_dirs: List[str],
        log_cb: Callable[[str], None],
        status_cb: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        sem = threading.BoundedSemaphore(max(1, spec.max_parallel))
        threads = []

        for i, yaml_path in enumerate(sub_yaml_paths):
            if self._stop_event.is_set():
                _safe_log(f"[Sweep] Run {i} skipped (stopped).\n", log_cb)
                _safe_status(i, "skipped", status_cb)
                continue

            def _run_one(idx: int = i, yp: str = yaml_path) -> None:
                sem.acquire()
                try:
                    if self._stop_event.is_set():
                        _safe_log(f"[Sweep] Run {idx}: Skipped.\n", log_cb)
                        _safe_status(idx, "skipped", status_cb)
                        return
                    _safe_log(f"[Sweep] Run {idx}: Starting…\n", log_cb)
                    _safe_status(idx, "running", status_cb)
                    try:
                        handle = backend.launch(
                            runner_host_path=runner_host,
                            project_yaml_host_path=yp,
                            lib_dirs_host=lib_dirs,
                            extra_args=["--resume"],
                        )
                        for line in handle.stdout_lines():
                            _safe_log(f"[{idx}] {line}", log_cb)
                        rc = handle.returncode
                        if rc == 0:
                            _safe_log(f"[Sweep] Run {idx}: Done (exit 0).\n", log_cb)
                            _safe_status(idx, "done", status_cb)
                        else:
                            _safe_log(f"[Sweep] Run {idx}: Failed (exit {rc}).\n", log_cb)
                            _safe_status(idx, "failed", status_cb)
                    except Exception as exc:
                        _safe_log(f"[Sweep] Run {idx}: Error — {exc}\n", log_cb)
                        _safe_status(idx, "failed", status_cb)
                finally:
                    sem.release()

            t = threading.Thread(target=_run_one, daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        if self._stop_event.is_set():
            _safe_log("[Sweep] Sweep interrupted by user.\n", log_cb)
        else:
            _safe_log("[Sweep] All runs complete.\n", log_cb)


def _safe_log(line: str, log_cb: Callable[[str], None]) -> None:
    try:
        log_cb(line)
    except Exception:
        pass


def _safe_status(idx: int, status: str,
                 status_cb: Optional[Callable[[int, str], None]]) -> None:
    if status_cb is None:
        return
    try:
        status_cb(idx, status)
    except Exception:
        pass
