"""
Simudo GUI — shared widget stylesheets and small utilities.

Import from here rather than duplicating definitions across panel modules.
"""

from __future__ import annotations
import glob
import math
import os
import re
import panel as pn
import yaml


# ── Widget stylesheets (dark theme) ───────────────────────────────────────────

INPUT_SS = ["""
:host { font-size: 13px !important; }
.bk-input { background: #1a2435 !important; color: #c8d6e5 !important;
    border: 1px solid #2d3748 !important; border-radius: 3px !important;
    padding: 2px 6px !important; font-size: 13px !important; }
.bk-input:focus { border-color: #4a9eff !important; outline: none !important; }
"""]

SELECT_SS = ["""
:host { font-size: 12px !important; }
select { background: #1a2435 !important; color: #c8d6e5 !important;
    border: 1px solid #2d3748 !important; border-radius: 3px !important;
    padding: 1px 2px !important; font-size: 12px !important; }
"""]

BTN_LIGHT_SS = ["""
.bk-btn { background: #1a2435 !important; color: #8a9bb5 !important;
    border: 1px solid #2d3748 !important; border-radius: 3px !important;
    font-size: 12px !important; padding: 0 !important; cursor: pointer !important; }
.bk-btn:hover { color: #e74c3c !important; border-color: #e74c3c !important; }
"""]

BTN_MOVE_SS = ["""
.bk-btn { background: transparent !important; color: #556070 !important;
    border: none !important; font-size: 11px !important;
    padding: 0 !important; cursor: pointer !important; line-height: 1 !important; }
.bk-btn:hover { color: #c8d6e5 !important; }
"""]

RADIO_SS = ["""
.bk-btn-group .bk-btn {
    background: #1a2435 !important; color: #8a9bb5 !important;
    border: 1px solid #2d3748 !important;
    font-size: 12px !important; padding: 2px 8px !important;
}
.bk-btn-group .bk-btn.bk-active {
    background: #2d4a6e !important; color: #4a9eff !important;
    border-color: #4a9eff !important;
}
"""]

ADD_BTN_SS = ["""
.bk-btn { background: transparent !important; color: #7a8caa !important;
    border: 1px dashed #2d3748 !important; border-radius: 4px !important;
    font-size: 13px !important; cursor: pointer !important;
    padding: 4px 12px !important; }
.bk-btn:hover { color: #4a9eff !important; border-color: #4a9eff !important; }
"""]

MULTICHOICE_SS = ["""
:host { font-size: 12px !important; }
.choices { background: #1a2435 !important; border: 1px solid #2d3748 !important;
    border-radius: 3px !important; }
.choices__inner { background: transparent !important; border: none !important;
    padding: 2px 4px !important; min-height: 28px !important; }
.choices__input { background: transparent !important; color: #c8d6e5 !important;
    font-size: 12px !important; }
.choices__item--selectable { color: #c8d6e5 !important;
    background: #2d4a6e !important; font-size: 11px !important; }
.choices__list--dropdown { background: #1a2435 !important;
    border: 1px solid #4a9eff !important; z-index: 200 !important; }
.choices__list--dropdown .choices__item--selectable {
    background: transparent !important; color: #c8d6e5 !important; }
.choices__list--dropdown .choices__item--selectable.is-highlighted {
    background: #2d4a6e !important; }
"""]

CHECKBOX_SS = ["""
:host { font-size: 12px !important; }
.bk-input-group { display: flex; align-items: center; gap: 6px; }
.bk-input-group label { color: #c8d6e5 !important; font-size: 12px !important;
    cursor: pointer !important; }
input[type=checkbox] { accent-color: #4a9eff; width: 13px; height: 13px; cursor: pointer; }
"""]

TOGGLE_SS = ["""
:host { font-size: 12px !important; }
.bk-btn { font-size: 12px !important; padding: 2px 10px !important;
    border-radius: 3px !important; }
"""]

# ── Section header HTML ────────────────────────────────────────────────────────

_SEC_STYLE = (
    "font-size:11px;font-weight:700;letter-spacing:0.10em;"
    "color:#8a9bb5;border-left:2px solid #4a9eff;padding-left:6px;"
)

def sec_header(title: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f'<div style="{_SEC_STYLE}">{title}</div>',
        sizing_mode="stretch_width", height=20, margin=(8, 0, 4, 0),
    )


def label(text: str, width: int = 70) -> pn.pane.HTML:
    return pn.pane.HTML(
        f'<span style="font-size:11px;color:#8a9bb5;white-space:nowrap;">{text}</span>',
        width=width, height=26, margin=0,
    )


def row_gap(items, gap: str = "6px") -> pn.Row:
    return pn.Row(
        *items, margin=0,
        styles={"gap": gap, "align-items": "center"},
        sizing_mode="stretch_width",
    )


def card(title: str, *content, collapsed: bool = False) -> pn.Card:
    return pn.Card(
        *content,
        title=title,
        collapsed=collapsed,
        sizing_mode="stretch_width",
        styles={"background": "#1a2435", "border": "1px solid #2d3748",
                "border-radius": "4px"},
        header_css_classes=[],
    )


# ── Checkpoint scanning ─────────────────────────────────────────────────────
# Shared by the Simulation panel's (secondary/advanced) resume picker and the
# app-header "Checkpoint" entry point — see gui/TODO.md "Checkpoint / Resume
# UX" for the overall design.

_CKPT_STAGE_LABELS = {
    "I": "optical ramp",
    "V": "voltage sweep",
}
_CKPT_NAME_RE = re.compile(r'_([A-Za-z]+)=([^=/\\]+)\.yaml$')


def scan_checkpoint_dir(abs_dir: str) -> list[tuple[str, str]]:
    """Return [(label, relative_filename), ...] for checkpoints in *abs_dir*.

    Labels are human-readable (e.g. "V = 0.4  (voltage sweep)"); the value
    stored in ``resume_from`` is just the bare filename, matching what
    ``simudo_1d_runner.py`` expects (a path relative to the output folder —
    checkpoints live in a subdirectory of it, so we include that segment).
    """
    if not abs_dir or not os.path.isdir(abs_dir):
        return []
    found = []
    for path in glob.glob(os.path.join(abs_dir, "*.yaml")):
        m = _CKPT_NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        param_name, value_str = m.group(1), m.group(2)
        try:
            value = float(value_str)
        except ValueError:
            continue
        stage = _CKPT_STAGE_LABELS.get(param_name, param_name)
        label_str = f"{param_name} = {value:g}  ({stage})"
        found.append((param_name, value, label_str, os.path.basename(path)))
    # Sort by parameter name, then by value (so V branches walk out from 0).
    found.sort(key=lambda t: (t[0], t[1]))
    return [(label_str, fname) for (_, _, label_str, fname) in found]


def read_checkpoint_metadata(checkpoint_yaml_path: str):
    """Return ``(param_name, param_value)`` for a checkpoint yaml, or
    ``None`` if it can't be read.

    Deliberately independent of ``simudo.io.h5yaml`` / ``simudo_1d_runner``
    (both import ``dolfin`` at module level, which the GUI's own conda
    environment does not have) -- the metadata file itself is plain
    key/string pairs (see ``AdaptiveStepper.write_xdmf_checkpoint``), so a
    bare ``yaml.safe_load`` is sufficient here.
    """
    m = _CKPT_NAME_RE.search(os.path.basename(checkpoint_yaml_path))
    if not m:
        return None
    param_name = m.group(1)
    try:
        with open(checkpoint_yaml_path, encoding='utf-8') as fh:
            metadata = yaml.safe_load(fh)
        return param_name, float(metadata['parameter'])
    except Exception:
        return None


def already_computed_values(run_folder: str, parameter_name: str) -> set:
    """Return the set of *parameter_name* values already recorded in
    ``<run_folder>/sim_<parameter_name>.csv`` (mirrors
    ``simudo_1d_runner._already_computed_values``, independently
    implemented here for the same dolfin-free-environment reason as
    :py:func:`read_checkpoint_metadata`).
    """
    csv_path = os.path.join(run_folder, f'sim_{parameter_name}.csv')
    if not os.path.isfile(csv_path):
        return set()
    try:
        with open(csv_path, newline='', encoding='utf-8') as fh:
            lines = fh.readlines()
    except OSError:
        return set()
    if not lines:
        return set()
    headers = [h.strip() for h in lines[0].split(',')]
    col = f'sweep_parameter:{parameter_name}'
    if col not in headers:
        return set()
    idx = headers.index(col)
    values = set()
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if idx < len(parts):
            try:
                values.add(float(parts[idx]))
            except ValueError:
                pass
    return values


def value_already_computed(value: float, computed: set) -> bool:
    return any(math.isclose(value, c, rel_tol=1e-9, abs_tol=1e-9) for c in computed)
