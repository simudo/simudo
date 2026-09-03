"""
AST-based introspection of Python Material subclass files.

Imports nothing from dolfin/simudo so it runs on the host machine without
the simulation environment installed.
"""

from __future__ import annotations
import ast, os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PropertyInfo:
    key: str
    display: str        # string to show in the UI (U-arg string or "(computed…)")
    copyable: bool      # True if we extracted a numeric value + unit
    value: Optional[float] = None
    unit: str = ""
    # When the get_dict entry is a callable definition (dict literal with a
    # 'type' key), is_callable=True and callable_type holds the type tag
    # (one of 'top_hat', 'energy_table', 'wavelength_table', 'formula').
    is_callable: bool = False
    callable_type: str = ""


# Recognised callable type tags in material get_dict() entries (mirrors the
# table in gui/notes/TODO-MANY-FIELDS.md §B).  Must stay in sync with the
# runner-side dispatcher in simudo_1d_runner.py.
_CALLABLE_TYPES = frozenset({
    "top_hat", "energy_table", "wavelength_table", "formula",
})


# ── Which classes count as materials ───────────────────────────────────────────
#
# A materials file may define helper classes that are not materials at all —
# `Alloy` (helpers.py) and `OpticsInterpolator` (alloy_optics.py) are the two in
# the shipped library.  Offering those in the GUI's material list is a bug: they
# have no get_dict() and cannot be assigned to a layer.  A material is
# recognised by its base class, not by the file it lives in.

def _base_name(node: ast.expr) -> Optional[str]:
    """Terminal name of a base-class expression: `Material` or `mod.Material`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_material_base(name: str) -> bool:
    """True for base names that mark their subclass as a material.

    `Material` itself, plus the library's concrete classes, which a user file
    may subclass to tweak one property (`SiliconMaterial`,
    `AluminumGalliumArsenideAlloy`, ...).  The bare helper `Alloy` is not a
    material base: it interpolates between two parents by composition.
    """
    return name == "Material" or (
        name != "Alloy" and (name.endswith("Material") or name.endswith("Alloy")))


@dataclass
class ClassInfo:
    class_name: str
    file_path: str
    material_name: Optional[str] = None   # value of class-level `name = "..."` if present
    properties: List[PropertyInfo] = field(default_factory=list)

    @property
    def copyable_properties(self) -> List[PropertyInfo]:
        return [p for p in self.properties if p.copyable]

    @property
    def computed_properties(self) -> List[PropertyInfo]:
        return [p for p in self.properties if not p.copyable]


# ── Value parsing ──────────────────────────────────────────────────────────────

def _parse_quantity_string(s: str) -> Tuple[Optional[float], str]:
    """Split 'value unit' into (float, unit_str), or return (None, s) if unparseable."""
    parts = s.strip().split(None, 1)
    if not parts:
        return None, ""
    try:
        v = float(parts[0])
        u = parts[1].strip() if len(parts) > 1 else ""
        return v, u
    except ValueError:
        return None, s


def _parse_value_node(key: str, node: ast.expr) -> PropertyInfo:
    """Extract a PropertyInfo from a value AST node."""
    # Callable definition: dict literal whose entries include a 'type' key
    # whose value is one of the recognised callable type tags.
    if isinstance(node, ast.Dict):
        type_tag = None
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and k.value == "type"
                    and isinstance(v, ast.Constant)
                    and isinstance(v.value, str)
                    and v.value in _CALLABLE_TYPES):
                type_tag = v.value
                break
        if type_tag is not None:
            return PropertyInfo(
                key=key,
                display=f"(callable: {type_tag})",
                copyable=False,
                is_callable=True,
                callable_type=type_tag,
            )

    # Single-argument call like U("1.42 eV") or ureg("300 K")
    if (isinstance(node, ast.Call)
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        arg = node.args[0].value
        v, u = _parse_quantity_string(arg)
        if v is not None:
            return PropertyInfo(key=key, display=arg, copyable=True, value=v, unit=u)
        # String arg but not numeric — still show it, mark not copyable
        return PropertyInfo(key=key, display=f'"{arg}"', copyable=False)

    # Plain numeric constant
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return PropertyInfo(key=key, display=str(node.value),
                            copyable=True, value=float(node.value), unit="")

    # Unary minus on a constant, e.g. -0.712
    if (isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))):
        v = -float(node.operand.value)
        return PropertyInfo(key=key, display=str(v), copyable=True, value=v, unit="")

    # Anything else — computed at runtime
    return PropertyInfo(key=key, display="(computed at runtime)", copyable=False)


# ── Function body extraction ───────────────────────────────────────────────────

def _extract_props_from_function(func_node: ast.FunctionDef) -> List[PropertyInfo]:
    """
    Walk a function body and collect key→value pairs from:
      - d.update({k: v, ...})
      - d[k] = v
    """
    props: List[PropertyInfo] = []
    for node in ast.walk(func_node):
        # d.update({...})
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and node.args
                and isinstance(node.args[0], ast.Dict)):
            for k, v in zip(node.args[0].keys, node.args[0].values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    props.append(_parse_value_node(k.value, v))

        # d[k] = v  (ast.Assign with Subscript target)
        if (isinstance(node, ast.Assign)
                and node.targets
                and isinstance(node.targets[0], ast.Subscript)):
            sl = node.targets[0].slice
            # Python 3.9+: slice is the node directly; 3.8-: wrapped in Index
            if isinstance(sl, ast.Index):
                sl = sl.value  # type: ignore[attr-defined]
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                props.append(_parse_value_node(sl.value, node.value))

    return props


# ── File-level extraction ──────────────────────────────────────────────────────

def extract_classes_from_file(file_path: str) -> List[ClassInfo]:
    """Parse a .py file and return a ClassInfo for every *material* class in it.

    Non-material helper classes are skipped — see `_is_material_base`.
    """
    try:
        source = open(file_path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(source)
    except Exception:
        return []

    # Two passes: a class also counts as a material if it subclasses another
    # material defined earlier in the same file.
    local_materials: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names = [n for n in (_base_name(b) for b in node.bases) if n]
            if any(_is_material_base(n) or n in local_materials for n in names):
                local_materials.add(node.name)

    results: List[ClassInfo] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name not in local_materials:
            continue    # helper class, not a material — keep it out of the GUI
        props: List[PropertyInfo] = []
        material_name: Optional[str] = None
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "get_dict":
                props = _extract_props_from_function(item)
            # class-level: name = "SomeName"
            if (isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == "name"
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)):
                material_name = item.value.value
        results.append(ClassInfo(
            class_name=node.name,
            file_path=file_path,
            material_name=material_name,
            properties=props,
        ))
    return results


def scan_library_dirs(library_dirs: list) -> List[ClassInfo]:
    """Scan all .py files in *library_dirs* and return all ClassInfo objects found."""
    results: List[ClassInfo] = []
    seen: set = set()
    for d in library_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fname in sorted(files):
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                if fpath in seen:
                    continue
                seen.add(fpath)
                results.extend(extract_classes_from_file(fpath))
    return results


def get_property_keys_for_material(mat, library_dirs: list) -> set:
    """
    Return the set of property keys provided by a Material model object.
    Works for inline materials and library/file references (via AST).
    """
    from simudo.gui.model import Material  # local import to avoid circular
    if mat.source is None:
        return set(mat.properties.keys())
    # Library-ref or derived material: start with library class keys
    keys: set = set()
    if mat.source.startswith("library://"):
        cls_name = mat.source[len("library://"):]
        for info in scan_library_dirs(library_dirs):
            if info.class_name == cls_name:
                keys = {p.key for p in info.properties}
                break
    elif mat.source.startswith("file://"):
        rest = mat.source[len("file://"):]
        path, _, cls_name = rest.rpartition("::")
        for info in extract_classes_from_file(path):
            if info.class_name == cls_name:
                keys = {p.key for p in info.properties}
                break
    # Derived material: also include any additional override keys
    keys.update(mat.properties.keys())
    return keys
