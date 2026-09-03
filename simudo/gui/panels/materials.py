"""
Simudo GUI — Materials panel.

Materials supply property defaults at PRIORITY_MATERIAL (lowest).  Two kinds:
  • Inline      — key/value/unit pairs stored directly in the YAML.
  • Library ref — ``source: library://ClassName``, loaded by the runner.

The panel shows a browseable catalogue of classes found in library directories
and a list of project-specific materials.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import panel as pn

from simudo.gui.model import Material
from simudo.gui.panels.mat_ast import ClassInfo, PropertyInfo, scan_library_dirs
from simudo.gui.panels.shared import (
    INPUT_SS, SELECT_SS, BTN_LIGHT_SS, RADIO_SS, ADD_BTN_SS,
    sec_header, label,
)

if TYPE_CHECKING:
    from simudo.gui.app import SimudoApp

# ── Styles ─────────────────────────────────────────────────────────────────────

_DOT_SS = ["""
:host { display:inline-flex; align-items:center; }
.bk-btn { width:14px !important; height:14px !important; border-radius:50% !important;
    border:2px solid transparent !important; padding:0 !important;
    cursor:pointer !important; min-width:0 !important; flex-shrink:0 !important; }
.bk-btn:hover { filter:brightness(1.3) !important; }
"""]

_ROW_BTN_SS = ["""
:host { flex:1; min-width:0; display:block; }
.bk-btn { background:transparent !important; border:none !important;
    color:#c8d6e5 !important; font-size:12px !important;
    text-align:left !important; padding:0 !important; cursor:pointer !important;
    width:100% !important; white-space:nowrap !important;
    overflow:hidden !important; text-overflow:ellipsis !important; }
.bk-btn:hover { color:#4a9eff !important; }
"""]

_SMALL_BTN_SS = ["""
.bk-btn { background:#1a2435 !important; color:#8a9bb5 !important;
    border:1px solid #2d3748 !important; border-radius:3px !important;
    font-size:11px !important; padding:0 6px !important; cursor:pointer !important;
    height:22px !important; white-space:nowrap !important; }
.bk-btn:hover { color:#c8d6e5 !important; border-color:#4a9eff !important; }
"""]

_DEL_SS = ["""
:host { display:inline-flex; }
.bk-btn { background:transparent !important; border:none !important;
    color:#556070 !important; font-size:14px !important;
    padding:0 4px !important; cursor:pointer !important;
    line-height:1 !important; width:22px !important; }
.bk-btn:hover { color:#e74c3c !important; }
"""]

_BROWSE_SS = ["""
.bk-btn { background:#1a2435 !important; color:#8a9bb5 !important;
    border:1px solid #2d3748 !important; border-radius:3px !important;
    font-size:12px !important; padding:0 8px !important; cursor:pointer !important; }
.bk-btn:hover { color:#c8d6e5 !important; border-color:#4a9eff !important; }
"""]

_SUBSEC = (
    "font-size:11px;font-weight:700;letter-spacing:0.08em;color:#8a9bb5;"
    "border-left:2px solid #2d4a6e;padding-left:6px;margin-top:2px;"
)
_MAT_COLORS = [
    '#4a9eff', '#e07b39', '#27ae60', '#9b59b6', '#e74c3c',
    '#1abc9c', '#f39c12', '#2980b9', '#8e44ad', '#16a085',
]


def _mat_color(i: int) -> str:
    return _MAT_COLORS[i % len(_MAT_COLORS)]


def _sub(title: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f'<div style="{_SUBSEC}">{title}</div>',
        sizing_mode="stretch_width", height=20, margin=(6, 0, 2, 0),
    )


def _note(text: str, color: str = "#556070") -> pn.pane.HTML:
    return pn.pane.HTML(
        f'<div style="font-size:11px;color:{color};padding:2px 0 4px 0;">{text}</div>',
        sizing_mode="stretch_width", margin=0,
    )


def _sanitize_name(s: str) -> str:
    import re
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s.strip())
    if s and s[0].isdigit():
        s = '_' + s
    return s or "material"


def _unique_name(base: str, taken: Set[str]) -> str:
    name, n = base, 2
    while name in taken:
        name = f"{base}_{n}"; n += 1
    return name


def _tk_pick_dir() -> Optional[str]:
    import threading
    result = [None]
    def _run():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", 1)
            path = filedialog.askdirectory(title="Select material library directory")
            root.destroy()
            result[0] = path or None
        except Exception:
            pass
    t = threading.Thread(target=_run); t.start(); t.join(timeout=60)
    return result[0]


# ── Panel ──────────────────────────────────────────────────────────────────────

class MaterialsPanel:
    def __init__(self, app: "SimudoApp"):
        self.app = app
        self._sel_lib: Optional[int] = None   # index into _lib_classes (single-select)
        # Set of expanded project material indices (multi-expand)
        self._expanded_mats: Set[int] = set()
        self._lib_classes: List[ClassInfo] = []
        # Persistent record of which property keys couldn't be copied per material name
        self._skipped: Dict[str, List[str]] = {}

        self._view = pn.Column(
            sizing_mode="stretch_width",
            styles={"padding": "12px", "gap": "4px", "overflow-y": "auto"},
        )

    def view(self) -> pn.Column:
        self._rebuild_view()
        return self._view

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _lib_class_names(self) -> Set[str]:
        return {c.class_name for c in self._lib_classes}

    def _project_mat_names(self) -> Set[str]:
        return {m.name for m in self.app.project.materials}

    # ── Top-level rebuild ──────────────────────────────────────────────────────

    def _rebuild_view(self):
        self._lib_classes = scan_library_dirs(self.app.get_library_dirs())
        # Clamp expanded set to valid indices
        n = len(self.app.project.materials)
        self._expanded_mats = {i for i in self._expanded_mats if i < n}

        n_classes = len(self._lib_classes)
        lib_status = (
            f'{n_classes} class{"es" if n_classes != 1 else ""} found'
        )

        # Library-related cards grouped with left accent
        lib_group = pn.Column(
            pn.pane.HTML(
                f'<div style="font-size:10px;font-weight:700;letter-spacing:0.1em;'
                f'color:#4a9eff;margin-bottom:3px;">LIBRARY — {lib_status}</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            self._build_lib_dirs_card(),
            self._build_library_section(),
            sizing_mode="stretch_width",
            styles={
                "border-left": "3px solid #2d4a6e",
                "padding-left": "10px",
                "gap": "4px",
            },
        )

        objects = [
            sec_header("MATERIALS"),
            pn.pane.HTML(
                '<div style="font-size:12px;color:#8a9bb5;padding:2px 0 6px 0;">'
                'Materials are optional. They let you give the same parameter values '
                'to multiple layers at once.<br>'
                'Properties set here have <em>low priority</em> — anything set in '
                'the Layers panel overrides them.</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            lib_group,
            self._build_project_section(),
            self._build_validation_warning(),
        ]
        self._view.objects = objects

    # ── Library-directories config card ───────────────────────────────────────

    def _build_lib_dirs_card(self) -> pn.Column:
        bundled = self.app.bundled_lib_dir
        bundled_ok = os.path.isdir(bundled)
        bundled_html = (
            f'<span style="font-family:monospace;font-size:11px;color:#8a9bb5;">'
            f'{bundled}</span>'
            + (' <span style="color:#27ae60;font-size:10px;">(found)</span>'
               if bundled_ok else
               ' <span style="color:#e74c3c;font-size:10px;">(not found)</span>')
        )
        user_in = pn.widgets.TextInput(
            value=self.app.user_lib_dir,
            placeholder="Path to a folder of .py material files (optional)",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        browse_btn = pn.widgets.Button(
            name="Browse…", height=26, margin=0, stylesheets=_BROWSE_SS,
        )

        def _on_user(e):
            self.app.set_user_lib_dir(e.new.strip())

        def _on_browse(e):
            path = _tk_pick_dir()
            if path:
                user_in.value = path
                self.app.set_user_lib_dir(path)
                self._rebuild_view()

        user_in.param.watch(_on_user, "value")
        browse_btn.on_click(_on_browse)

        lib_dirs = self.app.get_library_dirs()
        n = len(self._lib_classes)
        status = (f'{n} class{"es" if n != 1 else ""} found in '
                  f'{len(lib_dirs)} director{"ies" if len(lib_dirs) != 1 else "y"}. '
                  f'User directory is searched first (takes precedence over bundled).')

        card = pn.Card(
            pn.Row(
                pn.pane.HTML('<span style="font-size:11px;color:#8a9bb5;">Bundled:</span>',
                             width=58, height=20, margin=0),
                pn.pane.HTML(bundled_html, sizing_mode="stretch_width", height=20, margin=0),
                margin=0, styles={"align-items": "center", "gap": "4px"},
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.pane.HTML('<span style="font-size:11px;color:#8a9bb5;">User dir:</span>',
                             width=58, height=26, margin=0),
                user_in, browse_btn,
                margin=0, styles={"gap": "4px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            _note(status),
            title="Materials file location",
            collapsed=True,
            sizing_mode="stretch_width",
            styles={"background": "#1a2435", "border": "1px solid #2d3748",
                    "border-radius": "4px"},
            margin=0,
        )
        hint = pn.pane.HTML(
            '<div style="font-size:11px;color:#556070;padding:1px 0 3px 0;">'
            'Folders containing Python <code>.py</code> material class files. '
            'Expand to add a user directory.</div>',
            sizing_mode="stretch_width", margin=0,
        )
        return pn.Column(hint, card, sizing_mode="stretch_width",
                         styles={"gap": "3px"}, margin=(0, 0, 4, 0))

    # ── Library classes section ────────────────────────────────────────────────

    def _build_library_section(self) -> pn.Card:
        # Detect duplicates within the library scan
        seen_names: Dict[str, int] = {}
        for cls in self._lib_classes:
            seen_names[cls.class_name] = seen_names.get(cls.class_name, 0) + 1
        dup_names = {n for n, cnt in seen_names.items() if cnt > 1}

        rows = []
        if not self._lib_classes:
            rows.append(_note("No classes found. Check library directories above."))
        for i, cls in enumerate(self._lib_classes):
            is_sel = (self._sel_lib == i)

            # Name line: class name + optional material_name
            name_line = f'<strong style="color:{"#4a9eff" if is_sel else "#c8d6e5"};">{cls.class_name}</strong>'
            if cls.material_name and cls.material_name != cls.class_name:
                name_line += (f' <span style="font-size:10px;color:#556070;">'
                              f'name: "{cls.material_name}"</span>')

            # Duplicate badge
            dup_badge = ""
            if cls.class_name in dup_names:
                dup_badge = (' <span style="font-size:9px;background:#5a1a1a;color:#e74c3c;'
                             'border-radius:2px;padding:1px 5px;border:1px solid #e74c3c;">'
                             'duplicate class name</span>')

            src_line = (f'<span style="font-size:10px;color:#556070;">'
                        f'{os.path.basename(cls.file_path)}</span>')

            view_btn = pn.widgets.Button(
                name="View", height=22, margin=0, stylesheets=_SMALL_BTN_SS,
            )
            derive_btn = pn.widgets.Button(
                name="Derive from this", height=22, margin=0, stylesheets=_SMALL_BTN_SS,
            )
            ref_btn = pn.widgets.Button(
                name="Add to project", height=22, margin=0, stylesheets=_SMALL_BTN_SS,
            )
            view_btn.on_click(lambda e, idx=i: self._select_lib(idx))
            derive_btn.on_click(lambda e, idx=i: self._derive_from_lib_class(idx))
            ref_btn.on_click(lambda e, idx=i: self._add_lib_ref(idx))

            bg = "rgba(74,158,255,0.06)" if is_sel else "transparent"
            rows.append(pn.Row(
                pn.pane.HTML(
                    f'{name_line}{dup_badge}<br>{src_line}',
                    sizing_mode="stretch_width", margin=0,
                ),
                pn.Column(view_btn, ref_btn, derive_btn,
                          margin=0, styles={"gap": "2px", "flex-shrink": "0"}),
                margin=(0, 0, 4, 0),
                styles={"align-items": "flex-start", "gap": "8px",
                        "background": bg, "border-radius": "4px", "padding": "4px 6px"},
                sizing_mode="stretch_width",
            ))

        # If a library class is selected, show its detail inline
        if self._sel_lib is not None and self._sel_lib < len(self._lib_classes):
            rows.append(pn.pane.HTML(
                '<hr style="border:none;border-top:1px solid #1e2d3d;margin:6px 0;">',
                sizing_mode="stretch_width", margin=0,
            ))
            rows += list(self._build_lib_detail(self._lib_classes[self._sel_lib]))

        # Legend
        rows.append(pn.pane.HTML(
            '<div style="font-size:10px;color:#445060;margin-top:6px;">'
            '<strong>Add to project</strong>: pure library reference — all properties '
            'including computed ones are available at runtime, no overrides. '
            '<strong>Derive from this</strong>: creates a derived material that uses the '
            'library class as a base; you can then add or override specific numeric values '
            'while keeping all computed properties.</div>',
            sizing_mode="stretch_width", margin=0,
        ))

        return pn.Card(
            *rows,
            title="Library materials",
            collapsed=False,
            sizing_mode="stretch_width",
            styles={"background": "#161d2e", "border": "1px solid #2d3748",
                    "border-radius": "4px"},
            margin=(0, 0, 8, 0),
        )

    def _build_lib_detail(self, cls: ClassInfo):
        n_copy = len(cls.copyable_properties)
        n_comp = len(cls.computed_properties)
        yield _note(
            f'<strong style="color:#c8d6e5;">{cls.class_name}</strong> — '
            f'{len(cls.properties)} properties: '
            f'{n_copy} with extractable values'
            + (f', {n_comp} computed at runtime' if n_comp else '')
            + '. Read-only. Use <em>Add to project</em> or <em>Add editable copy</em> to use in your project.'
        )
        if cls.properties:
            yield self._prop_table_readonly(cls.properties)
        else:
            yield _note("No properties found (no get_dict() or empty).", color="#7a8caa")

    # ── Project materials section ──────────────────────────────────────────────

    def _build_project_section(self) -> pn.Column:
        lib_names = self._lib_class_names()
        mat_names = [m.name for m in self.app.project.materials]
        # Detect duplicates within project materials
        seen: Dict[str, int] = {}
        for n in mat_names:
            seen[n] = seen.get(n, 0) + 1
        dup_mat = {n for n, cnt in seen.items() if cnt > 1}

        rows = []
        _title = pn.pane.HTML(
            f'<div style="{_SUBSEC}">Project materials</div>',
            sizing_mode="stretch_width", height=20, margin=(6, 0, 2, 0),
        )
        rows.append(_title)
        rows.append(_note(
            "Defined here, saved in the project file, and referenced by layers."
        ))

        materials = self.app.project.materials
        if not materials:
            rows.append(_note("None yet — use Create below.", color="#7a8caa"))

        for i, mat in enumerate(materials):
            is_expanded = i in self._expanded_mats
            color = _mat_color(i)

            # Name badges
            badges = ""
            if mat.name in dup_mat:
                badges += (' <span style="font-size:9px;background:#5a1a1a;color:#e74c3c;'
                           'border-radius:2px;padding:1px 5px;border:1px solid #e74c3c;">'
                           'duplicate name</span>')
            if mat.name in lib_names:
                badges += (' <span style="font-size:9px;background:#2d1a00;color:#e07b39;'
                           'border-radius:2px;padding:1px 5px;border:1px solid #e07b39;">'
                           'same name as library class</span>')

            dot = pn.widgets.Button(
                name="", height=14, width=14, margin=0,
                stylesheets=_DOT_SS,
                styles={"background": color,
                        "border-color": "#ffffff" if is_expanded else color},
            )
            name_btn = pn.widgets.Button(
                name=mat.name, margin=0, stylesheets=_ROW_BTN_SS,
                styles={"font-weight": "600" if is_expanded else "normal"},
            )
            src_label = mat.source or "inline"
            src_pane = pn.pane.HTML(
                f'<span style="font-size:10px;color:#556070;">{src_label}</span>'
                + badges,
                sizing_mode="stretch_width", margin=0,
            )
            del_btn = pn.widgets.Button(name="✕", margin=0, stylesheets=_DEL_SS)

            dot.on_click(lambda e, idx=i: self._toggle_mat(idx))
            name_btn.on_click(lambda e, idx=i: self._toggle_mat(idx))
            del_btn.on_click(lambda e, idx=i: self._delete_mat(idx))

            bg = "rgba(74,158,255,0.07)" if is_expanded else "transparent"
            rows.append(pn.Row(
                dot,
                pn.Column(
                    name_btn, src_pane,
                    margin=0, styles={"gap": "0px", "flex": "1", "min-width": "0"},
                ),
                del_btn,
                margin=(0, 0, 2, 0),
                styles={"align-items": "center", "gap": "6px",
                        "background": bg, "border-radius": "4px", "padding": "2px 4px"},
                sizing_mode="stretch_width",
            ))

            # Inline detail when expanded (independent of other materials)
            if is_expanded:
                rows.append(pn.Column(
                    *self._build_mat_detail(mat),
                    sizing_mode="stretch_width",
                    styles={"background": "#1a2435", "border": "1px solid #2d3748",
                            "border-radius": "4px", "padding": "8px 10px",
                            "margin": "2px 0 6px 22px"},
                ))

        rows.append(self._build_create_form())
        return pn.Column(*rows, sizing_mode="stretch_width")

    # ── Create new material form ───────────────────────────────────────────────

    def _build_create_form(self) -> pn.Column:
        lib_names = self._lib_class_names()
        mat_names = self._project_mat_names()

        name_in = pn.widgets.TextInput(
            placeholder="material name",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        name_err = pn.pane.HTML("", visible=False, margin=0, sizing_mode="stretch_width")

        inherit_opts = ["(blank — start empty)"]
        inherit_opts += [f"library: {c.class_name}" for c in self._lib_classes]
        inherit_opts += [f"copy: {m.name}" for m in self.app.project.materials]
        inherit_sel = pn.widgets.Select(
            options=inherit_opts, value=inherit_opts[0],
            height=26, margin=0, stylesheets=SELECT_SS, sizing_mode="stretch_width",
        )
        create_btn = pn.widgets.Button(
            name="+ Create material", button_type="light",
            height=26, margin=0, stylesheets=ADD_BTN_SS, sizing_mode="stretch_width",
        )

        def _check_name(e):
            raw = e.new.strip()
            if not raw:
                name_err.object = ""; name_err.visible = False; return
            san = _sanitize_name(raw)
            if san in lib_names:
                name_err.object = (
                    '<span style="font-size:11px;color:#e74c3c;">'
                    f'⚠ "{san}" is also a library class name. Choose a different name '
                    'to avoid confusion.</span>'
                )
                name_err.visible = True
            elif san in mat_names:
                name_err.object = (
                    '<span style="font-size:11px;color:#e74c3c;">'
                    f'⚠ A project material named "{san}" already exists.</span>'
                )
                name_err.visible = True
            else:
                name_err.object = ""; name_err.visible = False

        name_in.param.watch(_check_name, "value")

        def _create(e):
            raw = name_in.value.strip()
            name = _sanitize_name(raw) if raw else ""
            if not name:
                return
            choice = inherit_sel.value
            skipped: List[str] = []

            if choice == "(blank — start empty)":
                mat = Material(name=name)
            elif choice.startswith("library: "):
                cls_name = choice[len("library: "):]
                # Create a derived material (library source + empty overrides).
                # Computed properties remain available at runtime via the library class.
                mat = Material(name=name, source=f"library://{cls_name}", properties={})
            elif choice.startswith("copy: "):
                src_name = choice[len("copy: "):]
                src = next((m for m in self.app.project.materials if m.name == src_name), None)
                import copy as _copy
                mat = Material(name=name,
                               properties=_copy.deepcopy(src.properties) if src else {})
            else:
                mat = Material(name=name)

            if skipped:
                self._skipped[name] = skipped

            self.app.project.materials.append(mat)
            new_idx = len(self.app.project.materials) - 1
            self._expanded_mats.add(new_idx)
            self._sel_lib = None
            name_in.value = ""
            inherit_sel.value = inherit_opts[0]
            name_err.object = ""; name_err.visible = False
            self.app.autosave()
            self.app.refresh_layers_panel()
            self._rebuild_view()

        create_btn.on_click(_create)

        return pn.Column(
            pn.pane.HTML(
                f'<div style="{_SUBSEC}">Create new material</div>',
                sizing_mode="stretch_width", height=20, margin=(10, 0, 2, 0),
            ),
            name_in,
            name_err,
            pn.Row(
                pn.pane.HTML('<span style="font-size:11px;color:#8a9bb5;">From:</span>',
                             width=34, height=26, margin=0),
                inherit_sel,
                margin=(2, 0, 0, 0),
                styles={"gap": "4px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            create_btn,
            sizing_mode="stretch_width",
        )

    # ── Material detail (editable) ─────────────────────────────────────────────

    def _build_mat_detail(self, mat: Material):
        lib_names = self._lib_class_names()

        # ── Skipped-properties warning ─────────────────────────────────────────
        skipped = self._skipped.get(mat.name, [])
        if skipped:
            skip_items = "".join(
                f'<li style="font-family:monospace;font-size:11px;">{k}</li>'
                for k in skipped
            )
            # Always-visible description line
            yield pn.pane.HTML(
                f'<div style="font-size:11px;color:#e07b39;'
                f'background:rgba(224,123,57,0.08);'
                f'border:1px solid rgba(224,123,57,0.4);border-radius:4px;'
                f'padding:5px 8px;margin-bottom:3px;">'
                f'⚠ {len(skipped)} propert{"ies" if len(skipped)!=1 else "y"} '
                f'computed at runtime by the library class could not be copied as '
                f'plain numbers. Expand below to see the list.</div>',
                sizing_mode="stretch_width", margin=0,
            )
            # Collapsible list of keys
            yield pn.Card(
                pn.pane.HTML(
                    f'<ul style="margin:4px 0 0 0;padding-left:16px;color:#e07b39;">'
                    f'{skip_items}</ul>',
                    sizing_mode="stretch_width", margin=0,
                ),
                title=f"Show {len(skipped)} skipped properties",
                collapsed=True,
                sizing_mode="stretch_width",
                styles={"background": "rgba(224,123,57,0.06)",
                        "border": "1px solid rgba(224,123,57,0.3)",
                        "border-radius": "4px"},
                margin=(0, 0, 6, 0),
            )

        # ── Name warning ───────────────────────────────────────────────────────
        if mat.name in lib_names:
            yield pn.pane.HTML(
                '<div style="font-size:11px;color:#e07b39;padding:2px 0 4px 0;">'
                f'⚠ This material has the same name as a library class '
                f'(<code>{mat.name}</code>). Consider renaming to avoid confusion.</div>',
                sizing_mode="stretch_width", margin=0,
            )

        # ── Name ──────────────────────────────────────────────────────────────
        name_in = pn.widgets.TextInput(
            value=mat.name, height=26, margin=0,
            stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        name_err = pn.pane.HTML("", visible=False, margin=0, sizing_mode="stretch_width")
        mat_names_other = {m.name for m in self.app.project.materials if m is not mat}

        def _on_name(e):
            new = _sanitize_name(e.new) if e.new.strip() else mat.name
            if new in mat_names_other:
                name_err.object = (
                    f'<span style="font-size:11px;color:#e74c3c;">'
                    f'⚠ Name "{new}" already in use.</span>'
                )
                name_err.visible = True
                return
            # Transfer skipped record if name changes
            if mat.name in self._skipped:
                self._skipped[new] = self._skipped.pop(mat.name)
            mat.name = new
            name_err.object = ""; name_err.visible = False
            self.app.autosave()
            self._rebuild_view()

        name_in.param.watch(_on_name, "value")
        yield pn.Row(label("Name:", 46), name_in, margin=0,
                     styles={"gap": "4px", "align-items": "center"},
                     sizing_mode="stretch_width")
        yield name_err

        if mat.source:
            yield from self._build_derived_detail(mat)
        else:
            yield _note("Inline properties — applied at lowest priority on all layers "
                        "using this material.")
            yield self._build_prop_editor(mat)

    def _build_derived_detail(self, mat: Material):
        """Detail panel for library-reference and derived materials.

        Shows all library properties (numeric + computed) as an editable table.
        Numeric rows at library-default are dimmed; overridden rows are amber with a
        "Restore library value" button.  Computed rows show a "calculated" badge when
        not overridden; once edited they become amber with a ✕ delete button.
        Custom keys (in mat.properties but not from the library) appear after the
        library section.  A form at the bottom adds new keys.
        """
        from simudo.gui.panels.mat_ast import extract_classes_from_file

        # ── Resolve library class info ────────────────────────────────────────
        info = None
        if mat.source.startswith("library://"):
            cls_name = mat.source[len("library://"):]
            info = next((c for c in self._lib_classes if c.class_name == cls_name), None)
        elif mat.source.startswith("file://"):
            rest = mat.source[len("file://"):]
            path, _, cls_name = rest.rpartition("::")
            infos = extract_classes_from_file(path)
            info = next((c for c in infos if c.class_name == cls_name), None)

        lib_copyable: dict = {}       # key → PropertyInfo  (numeric library props)
        lib_computed_keys: list = []  # ordered list of computed key names
        lib_computed_set: set  = set()
        if info:
            for p in info.copyable_properties:
                lib_copyable[p.key] = p
            lib_computed_keys = [p.key for p in info.computed_properties]
            lib_computed_set  = set(lib_computed_keys)

        # ── Source info header ────────────────────────────────────────────────
        yield pn.pane.HTML(
            f'<div style="font-size:11px;color:#8a9bb5;padding:4px 0 4px 0;">'
            f'Source: <code style="color:#4a9eff;">{mat.source}</code> — '
            f'library properties applied at base priority; overrides (amber) win.</div>',
            sizing_mode="stretch_width", margin=0,
        )

        KEY_WIDTH = 400  # wide enough for ~60 monospace characters at 11 px

        def _make_row(key, val_str, unit_str, is_override, is_custom, is_computed=False):
            """Build one property row widget.

            is_computed — runtime-computed library property (no fixed library value).
            is_custom   — not in the library at all (user-added key).
            is_override — currently stored in mat.properties.

            A single ✕ button appears only when is_override; clicking it removes the
            key from mat.properties (restores library default for library props, or
            removes the row entirely for custom keys).
            """
            if is_custom:
                _dflt_color = "#c8d6e5"
            elif is_computed:
                _dflt_color = "#5a7090"
            else:
                _dflt_color = "#7a8caa"

            def _key_html(overridden: bool) -> str:
                color = "#c8a060" if overridden else _dflt_color
                badge = ""
                if is_computed and not overridden:
                    badge = ('<span style="font-size:9px;background:#1e2d3d;color:#5a8aaa;'
                             'border-radius:2px;padding:1px 4px;border:1px solid #2d4a6e;'
                             'margin-left:4px;white-space:nowrap;">calculated</span>')
                return (
                    f'<span style="font-size:11px;color:{color};font-family:monospace;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
                    f'display:inline-block;max-width:{KEY_WIDTH - 10}px;" title="{key}">'
                    f'{key}</span>{badge}'
                )

            key_pane = pn.pane.HTML(
                _key_html(is_override), width=KEY_WIDTH, height=24, margin=0,
                sizing_mode="fixed",
            )
            val_in = pn.widgets.TextInput(
                value=val_str,
                placeholder="type to override" if is_computed and not is_override else "",
                height=24, margin=0, stylesheets=INPUT_SS, width=100,
            )
            unit_in = pn.widgets.TextInput(
                value=unit_str, height=24, margin=0, stylesheets=INPUT_SS, width=80,
            )
            # ✕ visible only when the property is overridden; removes override on click
            clear_btn = pn.widgets.Button(
                name="✕", margin=0, stylesheets=_DEL_SS, width=22,
                visible=is_override,
            )

            def _set_overridden():
                key_pane.object = _key_html(True)
                row.styles = {
                    "align-items": "center", "gap": "3px",
                    "background": "rgba(200,160,60,0.10)",
                    "border-radius": "3px", "padding": "1px 4px",
                }
                clear_btn.visible = True

            def _on_val(e, k=key, u_in=unit_in):
                if not e.new.strip():
                    return
                try: v = float(e.new)
                except ValueError: v = e.new
                mat.properties[k] = {"value": v, "unit": u_in.value.strip()}
                _set_overridden()
                self.app.autosave()
                self.app.refresh_layers_panel()

            def _on_unit(e, k=key, v_in=val_in):
                raw = v_in.value.strip()
                if not raw:
                    return
                try: v = float(raw)
                except ValueError: v = raw
                mat.properties[k] = {"value": v, "unit": e.new.strip()}
                _set_overridden()
                self.app.autosave()

            def _on_clear(e, k=key):
                mat.properties.pop(k, None)
                self.app.autosave()
                self.app.refresh_layers_panel()
                self._rebuild_view()

            val_in.param.watch(_on_val, "value")
            unit_in.param.watch(_on_unit, "value")
            clear_btn.on_click(_on_clear)

            bg = "rgba(200,160,60,0.10)" if is_override else "transparent"
            row = pn.Row(
                key_pane,
                val_in, unit_in, clear_btn,
                pn.Spacer(sizing_mode="stretch_width"),
                margin=(0, 0, 1, 0),
                sizing_mode="stretch_width",
                styles={"align-items": "center", "gap": "3px",
                        "background": bg, "border-radius": "3px",
                        "padding": "1px 4px"},
            )
            return row

        # ── Build property rows ───────────────────────────────────────────────
        rows = []
        lib_all = list(lib_copyable) + lib_computed_keys

        if lib_all:
            rows.append(pn.pane.HTML(
                '<div style="font-size:10px;color:#556070;font-weight:600;'
                'letter-spacing:0.08em;margin:4px 0 2px 0;">LIBRARY PROPERTIES</div>',
                sizing_mode="stretch_width", margin=0,
            ))
            # Numeric (copyable) properties first
            for key, p in lib_copyable.items():
                is_override = key in mat.properties
                if is_override:
                    entry = mat.properties[key]
                    val_str  = str(entry.get("value", "")) if isinstance(entry, dict) else str(entry)
                    unit_str = entry.get("unit", "") if isinstance(entry, dict) else ""
                else:
                    val_str  = str(p.value) if p.value is not None else ""
                    unit_str = p.unit or ""
                rows.append(_make_row(key, val_str, unit_str,
                                      is_override=is_override, is_custom=False,
                                      is_computed=False))
            # Computed properties after
            for key in lib_computed_keys:
                is_override = key in mat.properties
                if is_override:
                    entry = mat.properties[key]
                    val_str  = str(entry.get("value", "")) if isinstance(entry, dict) else str(entry)
                    unit_str = entry.get("unit", "") if isinstance(entry, dict) else ""
                else:
                    val_str = ""; unit_str = ""
                rows.append(_make_row(key, val_str, unit_str,
                                      is_override=is_override, is_custom=False,
                                      is_computed=True))

        # ── Custom keys (in mat.properties but not in library at all) ─────────
        custom_keys = [k for k in mat.properties
                       if k not in lib_copyable and k not in lib_computed_set]
        if custom_keys:
            rows.append(pn.pane.HTML(
                '<div style="font-size:10px;color:#556070;font-weight:600;'
                'letter-spacing:0.08em;margin:8px 0 2px 0;">CUSTOM OVERRIDES</div>',
                sizing_mode="stretch_width", margin=0,
            ))
            for key in custom_keys:
                entry = mat.properties[key]
                val_str  = str(entry.get("value", "")) if isinstance(entry, dict) else str(entry)
                unit_str = entry.get("unit", "") if isinstance(entry, dict) else ""
                rows.append(_make_row(key, val_str, unit_str,
                                      is_override=True, is_custom=True, is_computed=False))

        if not info:
            rows.append(_note("Could not read class (file not found or parse error).",
                              color="#e07b39"))
        elif not lib_all:
            rows.append(_note("No properties found in this class.", color="#7a8caa"))

        yield pn.Column(*rows, sizing_mode="stretch_width")

        # ── Add new custom key ────────────────────────────────────────────────
        add_key_in  = pn.widgets.TextInput(
            placeholder="new key  (e.g. CB/mobility)",
            height=24, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        add_val_in  = pn.widgets.TextInput(
            placeholder="value", height=24, margin=0, stylesheets=INPUT_SS, width=100,
        )
        add_unit_in = pn.widgets.TextInput(
            placeholder="unit", height=24, margin=0, stylesheets=INPUT_SS, width=80,
        )
        add_btn = pn.widgets.Button(
            name="+ Add key", height=24, margin=0,
            stylesheets=ADD_BTN_SS, sizing_mode="stretch_width",
        )

        def _add_key(e):
            k = add_key_in.value.strip()
            if not k:
                return
            try: v = float(add_val_in.value.strip())
            except ValueError: v = add_val_in.value.strip()
            u = add_unit_in.value.strip()
            mat.properties[k] = {"value": v, "unit": u} if u else v
            add_key_in.value = add_val_in.value = add_unit_in.value = ""
            self.app.autosave()
            self.app.refresh_layers_panel()
            self._rebuild_view()

        add_btn.on_click(_add_key)
        yield pn.Column(
            pn.pane.HTML(
                '<div style="font-size:10px;color:#556070;font-weight:600;'
                'letter-spacing:0.08em;margin:8px 0 2px 0;">ADD KEY</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            pn.Row(add_key_in, add_val_in, add_unit_in,
                   margin=0, styles={"gap": "3px", "align-items": "center"}),
            add_btn,
            sizing_mode="stretch_width",
        )

    # ── Read-only property table ───────────────────────────────────────────────

    def _prop_table_readonly(self, props: List[PropertyInfo]) -> pn.pane.HTML:
        rows = []
        for p in props:
            color = "#c8d6e5" if p.copyable else "#7a8caa"
            badge = ""
            if not p.copyable:
                badge = (' <span style="font-size:9px;background:#2d3748;color:#7a8caa;'
                         'border-radius:2px;padding:1px 4px;">runtime</span>')
            rows.append(
                f'<tr>'
                f'<td style="padding:2px 8px 2px 0;font-family:monospace;font-size:11px;'
                f'color:#8a9bb5;">{p.key}</td>'
                f'<td style="padding:2px 0;font-size:11px;color:{color};">'
                f'{p.display}{badge}</td>'
                f'</tr>'
            )
        return pn.pane.HTML(
            '<table style="border-collapse:collapse;width:100%;margin-top:4px;">'
            '<thead><tr>'
            '<th style="text-align:left;font-size:10px;color:#556070;'
            'font-weight:600;padding-bottom:3px;min-width:200px;">Key</th>'
            '<th style="text-align:left;font-size:10px;color:#556070;'
            'font-weight:600;padding-bottom:3px;">Value</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>',
            sizing_mode="stretch_width", margin=0,
        )

    # ── Inline property editor ─────────────────────────────────────────────────

    def _build_prop_editor(self, mat: Material) -> pn.Column:
        rows = []
        for key, entry in list(mat.properties.items()):
            if isinstance(entry, dict):
                val_str  = str(entry.get("value", ""))
                unit_str = str(entry.get("unit", ""))
            else:
                val_str = str(entry); unit_str = ""

            key_pane = pn.pane.HTML(
                f'<span style="font-size:11px;color:#8a9bb5;font-family:monospace;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
                f'display:inline-block;max-width:220px;">{key}</span>',
                width=220, height=24, margin=0,
            )
            val_in  = pn.widgets.TextInput(
                value=val_str, height=24, margin=0, stylesheets=INPUT_SS, width=100,
            )
            unit_in = pn.widgets.TextInput(
                value=unit_str, height=24, margin=0, stylesheets=INPUT_SS, width=90,
            )
            del_btn = pn.widgets.Button(name="✕", margin=0, stylesheets=_DEL_SS, width=22)

            def _on_val(e, k=key):
                try: v = float(e.new)
                except ValueError: v = e.new
                e2 = mat.properties.get(k, {})
                if isinstance(e2, dict):
                    e2["value"] = v; mat.properties[k] = e2
                else:
                    mat.properties[k] = v
                self.app.autosave()
                self.app.refresh_layers_panel()

            def _on_unit(e, k=key):
                e2 = mat.properties.get(k, {})
                if isinstance(e2, dict):
                    e2["unit"] = e.new.strip(); mat.properties[k] = e2
                self.app.autosave()

            def _on_del(e, k=key):
                mat.properties.pop(k, None)
                self.app.autosave()
                self.app.refresh_layers_panel()
                self._rebuild_view()

            val_in.param.watch(_on_val, "value")
            unit_in.param.watch(_on_unit, "value")
            del_btn.on_click(_on_del)
            rows.append(pn.Row(
                key_pane, pn.Spacer(sizing_mode="stretch_width"),
                val_in, unit_in, del_btn,
                margin=(0, 0, 2, 0),
                styles={"align-items": "center", "gap": "4px"},
            ))

        key_in   = pn.widgets.TextInput(
            placeholder="key  (e.g. CB/energy_level)",
            height=24, margin=0, stylesheets=INPUT_SS, width=220,
        )
        val_in2  = pn.widgets.TextInput(
            placeholder="value", height=24, margin=0, stylesheets=INPUT_SS, width=100,
        )
        unit_in2 = pn.widgets.TextInput(
            placeholder="unit", height=24, margin=0, stylesheets=INPUT_SS, width=90,
        )
        add_btn  = pn.widgets.Button(
            name="+", height=24, width=26, margin=0, stylesheets=ADD_BTN_SS,
        )

        def _add_prop(e):
            k = key_in.value.strip()
            if not k:
                return
            try: v = float(val_in2.value.strip())
            except ValueError: v = val_in2.value.strip()
            u = unit_in2.value.strip()
            mat.properties[k] = {"value": v, "unit": u} if u else v
            key_in.value = val_in2.value = unit_in2.value = ""
            self.app.autosave()
            self.app.refresh_layers_panel()
            self._rebuild_view()

        add_btn.on_click(_add_prop)
        return pn.Column(
            *rows,
            pn.Row(key_in, val_in2, unit_in2, add_btn,
                   margin=(4, 0, 0, 0), styles={"gap": "4px", "align-items": "center"}),
            sizing_mode="stretch_width",
        )

    # ── Validation warning ─────────────────────────────────────────────────────

    def _build_validation_warning(self) -> pn.pane.HTML:
        defined = {m.name for m in self.app.project.materials}
        refs = [(l.name, l.material)
                for l in self.app.project.layers
                if l.material and l.material not in defined]
        if not refs:
            return pn.pane.HTML("", visible=False)
        items = "".join(
            f"<li><code style='color:#4a9eff'>{ln}</code> → "
            f"<code style='color:#e07b39'>{mn}</code></li>"
            for ln, mn in refs
        )
        return pn.pane.HTML(
            '<div style="font-size:11px;color:#e07b39;background:rgba(224,123,57,0.08);'
            'border:1px solid rgba(224,123,57,0.3);border-radius:4px;'
            'padding:6px 10px;margin-top:6px;">'
            '<strong>Layers reference undefined material names:</strong>'
            f'<ul style="margin:4px 0 0 0;padding-left:18px;">{items}</ul>'
            'Add a matching entry above, or clear the material field on those layers.</div>',
            sizing_mode="stretch_width", margin=0,
        )

    # ── Actions ────────────────────────────────────────────────────────────────

    def _select_lib(self, idx: int):
        self._sel_lib = idx if self._sel_lib != idx else None
        self._rebuild_view()

    def _toggle_mat(self, idx: int):
        if idx in self._expanded_mats:
            self._expanded_mats.discard(idx)
        else:
            self._expanded_mats.add(idx)
        self._rebuild_view()

    def _delete_mat(self, idx: int):
        mat = self.app.project.materials[idx]
        self._skipped.pop(mat.name, None)
        self.app.project.materials.pop(idx)
        # Remove deleted index; shift all higher indices down by 1
        self._expanded_mats.discard(idx)
        self._expanded_mats = {i - 1 if i > idx else i for i in self._expanded_mats}
        self.app.autosave()
        self.app.refresh_layers_panel()
        self._rebuild_view()

    def _derive_from_lib_class(self, idx: int):
        """Create a derived material from a library class (source + empty overrides)."""
        cls = self._lib_classes[idx]
        base_name = _sanitize_name(cls.material_name or cls.class_name)
        taken = self._lib_class_names() | self._project_mat_names()
        name = _unique_name(base_name, taken)

        self.app.project.materials.append(
            Material(name=name, source=f"library://{cls.class_name}", properties={})
        )
        new_idx = len(self.app.project.materials) - 1
        self._expanded_mats.add(new_idx)
        self._sel_lib = None
        self.app.autosave()
        self.app.refresh_layers_panel()
        self._rebuild_view()

    def _copy_lib_class(self, idx: int):
        """Create an inline copy of a library class (numeric values only).

        Kept for backward compatibility; the library browser now uses
        _derive_from_lib_class instead.
        """
        cls = self._lib_classes[idx]
        base_name = _sanitize_name(cls.material_name or cls.class_name)
        taken = self._lib_class_names() | self._project_mat_names()
        name = _unique_name(base_name, taken)

        props = {p.key: {"value": p.value, "unit": p.unit}
                 for p in cls.copyable_properties}
        skipped = [p.key for p in cls.computed_properties]
        if skipped:
            self._skipped[name] = skipped

        self.app.project.materials.append(Material(name=name, properties=props))
        new_idx = len(self.app.project.materials) - 1
        self._expanded_mats.add(new_idx)
        self._sel_lib = None
        self.app.autosave()
        self.app.refresh_layers_panel()
        self._rebuild_view()

    def _add_lib_ref(self, idx: int):
        """Create a library-reference material (covers all properties including computed)."""
        cls = self._lib_classes[idx]
        base_name = _sanitize_name(cls.material_name or cls.class_name)
        taken = self._lib_class_names() | self._project_mat_names()
        name = _unique_name(base_name, taken)

        self.app.project.materials.append(
            Material(name=name, source=f"library://{cls.class_name}")
        )
        new_idx = len(self.app.project.materials) - 1
        self._expanded_mats.add(new_idx)
        self._sel_lib = None
        self.app.autosave()
        self.app.refresh_layers_panel()
        self._rebuild_view()
