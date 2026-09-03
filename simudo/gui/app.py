"""
Simudo GUI — main entry point.

Run with:
    panel serve app.py --show --port 5006
from this directory (containing app.py), in an environment where panel is installed.

The working directory matters: the panels below are imported by bare name
(``from model import ...``), so serving app.py from anywhere else fails on
import.
"""

from __future__ import annotations
import ast, os, sys, json, threading, signal, subprocess, platform

import panel as pn
import param
import yaml

from simudo.gui.model import ExecutionProfile, Project, default_bands
from simudo.gui.yaml_io import load_project, save_project
from simudo.gui.panels.shared import scan_checkpoint_dir
from simudo.gui.panels.layers import LayersPanel
from simudo.gui.panels.materials import MaterialsPanel
from simudo.gui.panels.bands import BandsPanel
from simudo.gui.panels.processes import ProcessesPanel
from simudo.gui.panels.opt_fields import OptFieldsPanel
from simudo.gui.panels.bcs import BCsPanel
from simudo.gui.panels.simulation import SimulationPanel
from simudo.gui.panels.output import OutputPanel

pn.extension(sizing_mode="stretch_width")

# ── Global CSS ─────────────────────────────────────────────────────────────────

pn.config.raw_css.append("""
html, body { background: #0f1623 !important; color: #c8d6e5;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 0; }
.bk-root { background: #0f1623 !important; }
""")

# ── Nav button stylesheets ─────────────────────────────────────────────────────

_NAV_BTN_BASE = (
    "width:100% !important; text-align:left !important;"
    "border:none !important; border-radius:0 !important;"
    "padding:7px 16px !important; font-size:13px !important;"
    "cursor:pointer !important; box-shadow:none !important;"
    "display:flex !important; justify-content:space-between !important;"
    "align-items:center !important;"
)
_NAV_INACTIVE = (
    ":host { width:100%; display:block; overflow:hidden; }"
    f".bk-btn {{ {_NAV_BTN_BASE}"
    "background:transparent !important; border-left:3px solid transparent !important;"
    "color:#7a8caa !important; font-weight:normal !important;"
    "transition:color 0.12s,background 0.12s !important; }"
    ".bk-btn:hover { color:#c8d6e5 !important;"
    "background:rgba(255,255,255,0.04) !important; }"
)
_NAV_ACTIVE = (
    ":host { width:100%; display:block; overflow:hidden; }"
    f".bk-btn {{ {_NAV_BTN_BASE}"
    "background:rgba(74,158,255,0.12) !important;"
    "border-left:3px solid #4a9eff !important;"
    "color:#4a9eff !important; font-weight:600 !important; }"
)
# Variants with yellow ● badge rendered via ::after (flex keeps it in-bounds)
_NAV_INACTIVE_BADGE = (
    _NAV_INACTIVE
    + ".bk-btn::after { content:'●'; color:#f0c040; font-size:9px; flex-shrink:0;"
    "margin-left:4px; }"
)
_NAV_ACTIVE_BADGE = (
    _NAV_ACTIVE
    + ".bk-btn::after { content:'●'; color:#f0c040; font-size:9px; flex-shrink:0;"
    "margin-left:4px; }"
)
_FILEPATH_SS = ["""
:host { flex:1; min-width:0; display:block; }
.bk-input-group { height:100%; }
.bk-input { background:transparent !important; border:none !important;
    border-bottom:1px solid transparent !important; border-radius:0 !important;
    color:#7a8caa !important; font-size:12px !important;
    padding:0 4px !important; height:28px !important; cursor:text !important;
    transition:border-color 0.15s, color 0.15s !important; }
.bk-input:hover { border-bottom-color:#556070 !important; }
.bk-input:focus { border-bottom-color:#4a9eff !important;
    color:#c8d6e5 !important; outline:none !important; }
"""]

# ── Material library helpers ───────────────────────────────────────────────────

# Bundled library: simudo/materials/ (one level up from simudo/gui/)
_BUNDLED_LIB = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "materials")
)

_GUI_CONFIG_FILE = os.path.expanduser("~/.simudo_gui.yaml")


def _load_gui_config() -> dict:
    try:
        with open(_GUI_CONFIG_FILE, encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_gui_config(cfg: dict) -> None:
    try:
        with open(_GUI_CONFIG_FILE, "w", encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    except Exception:
        pass


def _load_profiles_by_type(cfg: dict) -> dict:
    """Load per-type last-used profiles from ~/.simudo_gui.yaml."""
    from simudo.gui.yaml_io import _load_execution_profile
    result = {}
    for type_str, d in cfg.get("execution_profiles", {}).items():
        try:
            result[type_str] = _load_execution_profile(dict(d, type=type_str))
        except Exception:
            pass
    return result


def _load_execution_profile_from_config(cfg: dict) -> ExecutionProfile:
    """Build an ExecutionProfile from ~/.simudo_gui.yaml, with sensible defaults."""
    from simudo.gui.yaml_io import _load_execution_profile
    if "execution" in cfg:
        return _load_execution_profile(cfg["execution"])
    return ExecutionProfile(
        type="docker",
        docker_container="",
        docker_host_root="",
        docker_container_root="/home/user/simudo",
        python_cmd="python3",
    )


def discover_material_classes(library_dirs: list) -> list[str]:
    """AST-scan library dirs for top-level class names in .py files."""
    classes = []
    for d in library_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                try:
                    tree = ast.parse(open(os.path.join(root, fname), encoding='utf-8').read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            classes.append(node.name)
                except Exception:
                    pass
    return sorted(set(classes))


# ── Nav items ──────────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("Layers",         "layers"),
    ("Materials",      "materials"),
    ("Bands",          "bands"),
    ("Processes",      "processes"),
    ("Opt. Fields",    "opt_fields"),
    ("Boundary Cond.", "bcs"),
    None,
    ("Simulation",     "simulation"),
    ("Output",         "output"),
]

# ── Recent-files helpers ───────────────────────────────────────────────────────

_RECENTS_FILE = os.path.expanduser("~/.simudo_gui_recents.json")

def _load_recents() -> list:
    try:
        with open(_RECENTS_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def _save_recents(paths: list) -> None:
    try:
        with open(_RECENTS_FILE, "w", encoding='utf-8') as f:
            json.dump(paths[:10], f)
    except Exception:
        pass

def _add_recent(path: str) -> list:
    recents = [p for p in _load_recents() if p != path]
    recents.insert(0, path)
    _save_recents(recents)
    return recents[:10]

# ── Native file dialogs (local use only) ──────────────────────────────────────

def _osascript_dialog(mode: str, initial_dir: str = "~",
                      initial_filename: str = "") -> str | None:
    """macOS file dialog via AppleScript.

    Works with any Python build (including conda), because it drives the OS
    dialog via subprocess rather than the tkinter/Tk framework.
    Returns the chosen POSIX path, or None if the user cancelled.
    """
    initial_dir = os.path.abspath(os.path.expanduser(initial_dir or "~"))
    if not os.path.isdir(initial_dir):
        initial_dir = os.path.expanduser("~")
    # Escape characters that would break an AppleScript string literal.
    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')
    try:
        if mode == "open":
            script = (
                'try\n'
                f'    set f to choose file with prompt "Open Simudo project"'
                f' default location POSIX file "{_esc(initial_dir)}"\n'
                '    return POSIX path of f\n'
                'on error\n'
                '    return ""\n'
                'end try'
            )
        else:
            name = os.path.basename(initial_filename) if initial_filename else "project.yaml"
            script = (
                'try\n'
                f'    set f to choose file name with prompt "Save Simudo project as"'
                f' default name "{_esc(name)}"'
                f' default location POSIX file "{_esc(initial_dir)}"\n'
                '    return POSIX path of f\n'
                'on error\n'
                '    return ""\n'
                'end try'
            )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=120,
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception as e:
        print(f"osascript dialog error: {e}")
        return None


def _tk_dialog(mode: str, initial_dir: str = ".") -> str | None:
    """Windows / Linux file dialog via tkinter.filedialog."""
    result = [None]
    def _run():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", 1)
            if mode == "open":
                path = filedialog.askopenfilename(
                    initialdir=initial_dir,
                    filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
                    title="Open Simudo project",
                )
            else:
                path = filedialog.asksaveasfilename(
                    initialdir=initial_dir, defaultextension=".yaml",
                    filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
                    title="Save Simudo project as",
                )
            root.destroy()
            result[0] = path or None
        except Exception as e:
            print(f"tkinter dialog error: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=120)
    return result[0]


def _native_dir_dialog(initial_dir: str = ".") -> str | None:
    """Show a native OS directory-picker dialog.  Blocks the calling thread.

    • macOS   — AppleScript ``choose folder`` via ``osascript``.
    • Windows / Linux — ``tkinter.filedialog.askdirectory``.

    Returns the chosen directory path, or ``None`` if the user cancelled.
    """
    initial_dir = os.path.abspath(os.path.expanduser(initial_dir or "."))
    if not os.path.isdir(initial_dir):
        initial_dir = os.path.expanduser("~")

    if platform.system() == "Darwin":
        def _esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            'try\n'
            f'    set f to choose folder with prompt "Select output folder"'
            f' default location POSIX file "{_esc(initial_dir)}"\n'
            '    return POSIX path of f\n'
            'on error\n'
            '    return ""\n'
            'end try'
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            path = result.stdout.strip().rstrip("/")
            return path if path else None
        except Exception as e:
            print(f"osascript dir dialog error: {e}")
            return None

    # Windows / Linux — tkinter in a thread (avoids main-thread restriction)
    result = [None]
    def _run():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", 1)
            path = filedialog.askdirectory(
                initialdir=initial_dir,
                title="Select output folder",
            )
            root.destroy()
            result[0] = path or None
        except Exception as e:
            print(f"tkinter dir dialog error: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=120)
    return result[0]


def _native_file_dialog(mode: str, initial_dir: str = ".",
                        initial_filename: str = "") -> str | None:
    """Show a native OS file-picker dialog.  Blocks the calling thread.

    • macOS   — AppleScript via ``osascript``; works with conda Python.
    • Windows / Linux — ``tkinter.filedialog`` (ships with CPython).

    Returns the chosen path, or ``None`` if the user cancelled.
    """
    if platform.system() == "Darwin":
        return _osascript_dialog(mode, initial_dir, initial_filename)
    return _tk_dialog(mode, initial_dir)


# Stylesheet for the 📁 Browse button used in Open / Save-As dropdowns.
_BROWSE_BTN_SS = ["""
    .bk-btn { background:#1e2a3a !important; color:#7a9cc5 !important;
        border:1px solid #2d3d50 !important; border-radius:3px !important;
        font-size:15px !important; line-height:1 !important;
        cursor:pointer !important; padding:0 4px !important; }
    .bk-btn:hover { background:#2d4a6e !important; color:#c8d6e5 !important; }
"""]


def _make_browse_btn(mode: str, path_input: pn.widgets.TextInput,
                     height: int = 28) -> pn.widgets.Button:
    """Return a small 📁 button that opens a native file-picker and
    populates *path_input* with the result.

    The dialog runs synchronously: the Tornado IOLoop (and therefore the
    browser tab) is frozen while the OS dialog is open, which is fine for
    a single-user local app — the user is interacting with the OS dialog
    and not the browser anyway.
    """
    btn = pn.widgets.Button(name="📁  Browse…", width=110, height=height, margin=0,
                             stylesheets=_BROWSE_BTN_SS)

    def _on_browse(event):
        current = path_input.value.strip()
        initial_dir = (
            os.path.dirname(os.path.abspath(current))
            if current else os.getcwd()
        )
        initial_filename = current if mode == "save" else ""
        path = _native_file_dialog(mode, initial_dir=initial_dir,
                                   initial_filename=initial_filename)
        if path:
            path_input.value = path

    btn.on_click(_on_browse)
    return btn

# ── App controller ─────────────────────────────────────────────────────────────

class SimudoApp:
    def __init__(self):
        self.project = Project()
        self._current_section = "layers"
        self._recents: list = _load_recents()
        cfg = _load_gui_config()
        self.user_lib_dir: str = cfg.get("user_library_dir", "")
        self._default_execution: ExecutionProfile = _load_execution_profile_from_config(cfg)
        self._execution_profiles_by_type: dict = _load_profiles_by_type(cfg)
        self._last_run_output_dir: str | None = None

        self._panels: dict = {}
        self._nav_btns: dict = {}   # populated by _build_sidebar; must exist before _rebuild_panels
        self._rebuild_panels()

        # ── Persistent layout widgets ──────────────────────────────────────────
        self._sidebar_col = pn.Column(width=155, styles={
            "background": "#111827", "border-right": "1px solid #1e2d3d",
            "padding-top": "8px", "flex-shrink": "0",
            "overflow-x": "hidden", "overflow-y": "auto",
        })

        # Filepath: editable TextInput styled as plain text
        self._filepath_w = pn.widgets.TextInput(
            value="", placeholder="(no save location set — click to set or use Save)",
            stylesheets=_FILEPATH_SS, sizing_mode="stretch_width",
        )
        def _on_filepath_change(event):
            p = event.new.strip()
            if p:
                self.project.filepath = p
        self._filepath_w.param.watch(_on_filepath_change, "value")

        # "Unsaved" badge shown when no filepath is set
        self._unsaved_badge = pn.pane.HTML(
            '<span style="font-size:11px;background:#5a3e00;color:#f0c040;'
            'border-radius:3px;padding:1px 6px;margin-left:4px;white-space:nowrap;">'
            'unsaved</span>',
            visible=True, width=70,
        )

        # Open dropdown (inline, below header)
        self._open_dropdown = pn.Column(visible=False, styles={
            "background": "#1a2435",
            "border-bottom": "1px solid #2d3748",
            "box-shadow": "0 6px 20px rgba(0,0,0,0.5)",
            "z-index": "50",
        })

        # Save-as dropdown (inline, below header)
        self._save_dropdown = pn.Column(visible=False, styles={
            "background": "#1a2435",
            "border-bottom": "1px solid #2d3748",
            "box-shadow": "0 6px 20px rgba(0,0,0,0.5)",
            "z-index": "50",
        })

        # Checkpoint dropdown (inline, below header) — two-step: pick a
        # project, then pick one of its checkpoints. See gui/TODO.md
        # "Checkpoint / Resume UX".
        self._checkpoint_dropdown = pn.Column(visible=False, styles={
            "background": "#1a2435",
            "border-bottom": "1px solid #2d3748",
            "box-shadow": "0 6px 20px rgba(0,0,0,0.5)",
            "z-index": "50",
        })

        # Error banner — shown when autosave or other operations fail; dismissed by ✕
        self._error_banner = pn.Row(visible=False, sizing_mode="stretch_width", styles={
            "background": "rgba(200,50,50,0.15)",
            "border-bottom": "1px solid rgba(200,50,50,0.40)",
            "padding": "6px 16px",
            "align-items": "center",
            "gap": "10px",
            "flex-shrink": "0",
        })

        # Save-location banner — shown prominently until a filepath is set
        _set_loc_btn = pn.widgets.Button(
            name="Set save location", button_type="warning",
            width=150, height=28, styles={"font-size": "12px", "flex-shrink": "0"},
        )
        _set_loc_btn.on_click(lambda e: self._toggle_save_dropdown())
        self._save_banner = pn.Row(
            pn.pane.HTML(
                '<span style="font-size:18px;line-height:1;">⚠</span>'
                '<span style="font-size:13px;color:#c8a060;margin-left:8px;">'
                '<strong style="color:#f0c040;">No save location set.</strong>'
                '&nbsp; Work will be lost if the server stops. '
                'Use <em>Save</em> in the header or click here to set a location now.'
                '</span>',
                sizing_mode="stretch_width",
            ),
            _set_loc_btn,
            visible=True,
            sizing_mode="stretch_width",
            styles={
                "background": "rgba(240,160,40,0.10)",
                "border-bottom": "1px solid rgba(240,160,40,0.30)",
                "padding": "8px 16px",
                "align-items": "center",
                "gap": "12px",
                "flex-shrink": "0",
            },
        )

        self._content_col = pn.Column(sizing_mode="stretch_both", styles={
            "background": "#161d2e", "overflow-y": "auto",
        })

        self._build_sidebar()
        self._show_section("layers")

    # ── Project management ─────────────────────────────────────────────────────

    @property
    def bundled_lib_dir(self) -> str:
        return _BUNDLED_LIB

    def get_library_dirs(self) -> list[str]:
        """Return valid library directories: user dir first (higher precedence)."""
        dirs = []
        if self.user_lib_dir and os.path.isdir(self.user_lib_dir):
            dirs.append(self.user_lib_dir)
        if os.path.isdir(_BUNDLED_LIB):
            dirs.append(_BUNDLED_LIB)
        return dirs

    def set_user_lib_dir(self, path: str):
        self.user_lib_dir = path
        cfg = _load_gui_config()
        cfg["user_library_dir"] = path
        _save_gui_config(cfg)

    def get_effective_execution_profile(self) -> ExecutionProfile:
        """Return the active execution profile: per-project override if set, else user default."""
        if self.project.execution is not None:
            return self.project.execution
        return self._default_execution

    def set_last_run_output_dir(self, host_path: str):
        """Called by the simulation panel once the runner's output directory is known."""
        self._last_run_output_dir = host_path

    def notify_run_output_dir(self, host_path: str):
        """Store the output dir and update both panels that display it."""
        self._last_run_output_dir = host_path
        # Keep the in-memory project's output_folder in sync with what the
        # runner itself just wrote into the live project YAML (see
        # simudo_1d_runner.py's ensure_new_dir write-back) -- otherwise the
        # next autosave would clobber that write-back with our stale value.
        if self.project.filepath:
            try:
                rel = os.path.relpath(
                    host_path, os.path.dirname(os.path.abspath(self.project.filepath)))
            except ValueError:
                rel = host_path
            if self.project.output_folder != rel:
                self.project.output_folder = rel
        op = self._panels.get("output")
        if op is not None:
            op.set_folder(host_path)
        sp = self._panels.get("simulation")
        if sp is not None:
            sp.set_actual_output_dir(host_path)

    def refresh_output_panel(self):
        """Refresh the Output panel plot (called after a run completes)."""
        op = self._panels.get("output")
        if op is not None:
            op.refresh()
        # Ensure the sim panel's "Last run" label is also up to date.
        if self._last_run_output_dir:
            sp = self._panels.get("simulation")
            if sp is not None:
                sp.set_actual_output_dir(self._last_run_output_dir)

    def set_default_execution_profile(self, profile: ExecutionProfile):
        """Persist profile as the user-level default in ~/.simudo_gui.yaml."""
        from simudo.gui.yaml_io import _execution_profile_to_dict
        self._default_execution = profile
        cfg = _load_gui_config()
        cfg["execution"] = _execution_profile_to_dict(profile)
        _save_gui_config(cfg)

    def get_execution_profile_for_type(self, type_str: str):
        """Return the last-used ExecutionProfile for this type, or None."""
        return self._execution_profiles_by_type.get(type_str)

    def save_execution_profile_for_type(self, type_str: str, profile: ExecutionProfile):
        """Remember the profile for this type in memory and persist to config."""
        from simudo.gui.yaml_io import _execution_profile_to_dict
        self._execution_profiles_by_type[type_str] = profile
        cfg = _load_gui_config()
        profiles = cfg.setdefault("execution_profiles", {})
        profiles[type_str] = _execution_profile_to_dict(profile)
        _save_gui_config(cfg)

    def new_project(self):
        """Reset to a blank project (default bands, no layers, no filepath)."""
        self.project = Project()
        self._rebuild_panels()
        self._update_filepath_display()
        self._show_section(self._current_section)

    def _rebuild_panels(self):
        self._panels = {
            "layers":     LayersPanel(self),
            "materials":  MaterialsPanel(self),
            "bands":      BandsPanel(self),
            "processes":  ProcessesPanel(self),
            "opt_fields": OptFieldsPanel(self),
            "bcs":        BCsPanel(self),
            "simulation": SimulationPanel(self),
            "output":     OutputPanel(self),
        }
        # Refresh badge after panels (and their missing-param computations) exist
        # _nav_btns may not exist yet on first call (before _build_sidebar)
        if self._nav_btns:
            self.update_layers_badge()

    def _update_filepath_display(self):
        p = self.project.filepath or ""
        self._filepath_w.value = p
        has_path = bool(p)
        self._unsaved_badge.visible = not has_path
        self._save_banner.visible   = not has_path

    def load_yaml(self, path: str, add_to_recents: bool = True):
        path = os.path.abspath(path)
        try:
            self.project = load_project(path)
            if add_to_recents:
                self._recents = _add_recent(path)
            self._rebuild_panels()
            self._update_filepath_display()
            self._show_section(self._current_section)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise

    def load_from_checkpoint(self, project_path: str, checkpoint_relpath: str):
        """Start a new (already-open, already-saved) project resuming from
        *checkpoint_relpath* (e.g. "checkpoints/sim_V=0.4.yaml") found under
        *project_path*'s configured output folder.

        Loads the *frozen* device-definition YAML that the runner copies
        into the output folder for provenance -- not project_path itself,
        which may have kept changing since that run -- so the resumed
        project's layers/materials/bands/etc. are guaranteed to match what
        actually produced the checkpoint. Saves in place inside that run
        folder (see gui/TODO.md "Checkpoint / Resume UX" for why).
        """
        project_path = os.path.abspath(project_path)
        src_proj = load_project(project_path)
        run_folder = os.path.abspath(os.path.join(
            os.path.dirname(project_path), src_proj.output_folder or "out"))
        frozen_path = os.path.join(run_folder, os.path.basename(project_path))
        if not os.path.isfile(frozen_path):
            raise FileNotFoundError(
                f"No project YAML found at {frozen_path} — this doesn't "
                f"look like an output folder produced by "
                f"simudo_1d_runner.py.")
        self.load_yaml(frozen_path)
        self.project.simulation.resume_from = checkpoint_relpath
        self.autosave()
        self._show_section("simulation")

    def save_yaml(self, path: str = None):
        """Save project. If no path given and no filepath set, prompt with dialog."""
        if path is None:
            path = self.project.filepath
        if not path:
            path = _native_file_dialog("save", initial_dir=os.getcwd())
        if not path:
            return  # user cancelled
        path = os.path.abspath(path)
        # Let exceptions propagate so callers (e.g. the save dropdown) can
        # display them; just log to the console as well for debugging.
        try:
            save_project(self.project, path)
        except Exception as e:
            print(f"Error saving {path}: {e}")
            raise
        self._recents = _add_recent(path)
        self._update_filepath_display()

    def autosave(self):
        """Write to disk silently if a filepath is already set."""
        if self.project.filepath:
            try:
                save_project(self.project, self.project.filepath)
            except Exception as e:
                print(f"Autosave failed: {e}")
                self.show_error(f"Autosave failed: {e}")

    def show_error(self, message: str):
        """Display a dismissable red banner below the header with *message*."""
        dismiss_btn = pn.widgets.Button(
            name="✕", width=24, height=24, margin=0,
            stylesheets=["""
                .bk-btn { background: transparent !important; border: none !important;
                    color: #e05555 !important; font-size: 14px !important;
                    cursor: pointer !important; padding: 0 !important; }
                .bk-btn:hover { color: #ff8080 !important; }
            """],
        )
        msg_pane = pn.pane.HTML(
            f'<span style="font-size:13px;color:#e07070;">'
            f'<strong style="color:#ff8080;">⚠ Error:</strong>&nbsp;{message}</span>',
            sizing_mode="stretch_width",
        )
        dismiss_btn.on_click(lambda e: setattr(self._error_banner, "visible", False))
        self._error_banner.objects = [msg_pane, dismiss_btn]
        self._error_banner.visible = True

    def refresh_layers_missing_params(self):
        """Recompute and redisplay the missing-params section and badge."""
        layers_panel = self._panels.get("layers")
        if layers_panel is None:
            return
        layers_panel._missing_params_pane.object = (
            layers_panel._build_missing_params_html()
        )
        self.update_layers_badge()

    def refresh_layers_panel(self):
        """Full rebuild of the Layers panel (table, schematic, overlays, missing params).

        Call this after any change that affects what layers can reference —
        e.g. when project materials are added or removed.
        """
        layers_panel = self._panels.get("layers")
        if layers_panel is None:
            return
        layers_panel.rebuild()
        self.update_layers_badge()

    def update_layers_badge(self):
        """Update the yellow ● badge on the Layers nav button.

        The badge is shown when any required parameter (from bands or EOPs)
        is unset in every region.  Called after any property edit and on rebuild.
        """
        layers_panel = self._panels.get("layers")
        has_missing = bool(
            layers_panel and layers_panel._compute_missing_params()
        )
        btn = self._nav_btns.get("layers")
        if btn is None:
            return
        active = self._current_section == "layers"
        if has_missing:
            btn.stylesheets = [_NAV_ACTIVE_BADGE if active else _NAV_INACTIVE_BADGE]
        else:
            btn.stylesheets = [_NAV_ACTIVE if active else _NAV_INACTIVE]

    # ── Section navigation ─────────────────────────────────────────────────────

    def _show_section(self, key: str):
        self._current_section = key
        for k, btn in self._nav_btns.items():
            btn.stylesheets = [_NAV_ACTIVE if k == key else _NAV_INACTIVE]
        panel = self._panels.get(key)
        self._content_col.objects = [panel.view() if panel else pn.pane.HTML("")]
        self.update_layers_badge()

    # ── Sidebar ────────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        items = []
        for item in NAV_ITEMS:
            if item is None:
                items.append(pn.pane.HTML(
                    '<hr style="border:none;border-top:1px solid #1e2d3d;margin:4px 0;">',
                    height=9,
                ))
                continue
            label, key = item
            btn = pn.widgets.Button(name=label, stylesheets=[_NAV_INACTIVE], width=155)
            btn.on_click(lambda e, k=key: self._show_section(k))
            self._nav_btns[key] = btn
            items.append(btn)
        self._sidebar_col.objects = items

    # ── Open dropdown ──────────────────────────────────────────────────────────

    def _build_open_content(self) -> pn.Column:
        path_input = pn.widgets.TextInput(
            placeholder="Path to .yaml file…",
            sizing_mode="stretch_width",
            stylesheets=_FILEPATH_SS,
        )
        error_pane = pn.pane.HTML("", visible=False, sizing_mode="stretch_width")

        def _close():
            self._open_dropdown.visible = False

        open_btn   = pn.widgets.Button(name="Open",   button_type="primary", width=65,
                                        styles={"font-size": "13px"})
        cancel_btn = pn.widgets.Button(name="Cancel", button_type="light",   width=70,
                                        styles={"font-size": "13px"})
        browse_btn = _make_browse_btn("open", path_input)

        recents_col = pn.Column(styles={"gap": "2px"})

        _RECENT_BTN_SS = ["""
        :host { display:block; }
        .bk-btn { width:100% !important; text-align:left !important;
            background:transparent !important; border:none !important;
            border-radius:3px !important; color:#8a9bb5 !important;
            font-size:12px !important; padding:4px 8px !important;
            cursor:pointer !important; overflow:hidden;
            text-overflow:ellipsis !important; white-space:nowrap !important; }
        .bk-btn:hover { background:rgba(74,158,255,0.08) !important;
            color:#c8d6e5 !important; }
        """]

        def refresh_recents():
            rows = []
            for p in self._recents:
                short = os.path.basename(p)
                r_btn = pn.widgets.Button(name=short, stylesheets=_RECENT_BTN_SS,
                                          sizing_mode="stretch_width")
                r_btn.on_click(lambda e, fp=p: _do_open_path(fp))
                rows.append(r_btn)
            if not rows:
                rows.append(pn.pane.HTML(
                    '<div style="color:#556070;font-size:12px;padding:6px 8px;">No recent files.</div>'
                ))
            recents_col.objects = rows

        def _do_open_path(path: str):
            path = (path or "").strip()
            if not path:
                return
            error_pane.visible = False
            if not os.path.exists(path):
                error_pane.object = (
                    f'<div style="color:#e74c3c;font-size:12px;padding:4px 0;">'
                    f'File not found: {path}</div>'
                )
                error_pane.visible = True
                return
            try:
                _close()
                self.load_yaml(path)
            except Exception as ex:
                error_pane.object = (
                    f'<div style="color:#e74c3c;font-size:12px;padding:4px 0;">'
                    f'Error loading file: {ex}</div>'
                )
                error_pane.visible = True
                self._open_dropdown.visible = True   # re-open to show error

        open_btn.on_click(lambda e: _do_open_path(path_input.value))
        cancel_btn.on_click(lambda e: _close())
        # Enter key in the path field also triggers open
        path_input.param.watch(lambda e: _do_open_path(e.new), "value")

        refresh_recents()

        return pn.Column(
            pn.Row(
                pn.pane.HTML(
                    '<div style="font-size:11px;font-weight:600;color:#7a8caa;'
                    'letter-spacing:0.08em;">OPEN PROJECT</div>',
                    sizing_mode="stretch_width",
                ),
                cancel_btn,
                styles={"padding": "10px 12px 6px", "align-items": "center"},
            ),
            pn.Row(path_input, browse_btn, open_btn,
                   styles={"padding": "0 12px 4px", "gap": "6px", "align-items": "center"}),
            pn.Row(error_pane, styles={"padding": "0 12px 4px"}, sizing_mode="stretch_width"),
            pn.pane.HTML(
                '<div style="font-size:11px;font-weight:600;color:#7a8caa;'
                'letter-spacing:0.08em;padding:4px 12px 4px;">RECENT</div>'
            ),
            pn.Column(recents_col, styles={"padding": "0 8px 10px"}),
        )

    # ── Save As dropdown ───────────────────────────────────────────────────────

    def _toggle_save_dropdown(self):
        if not self._save_dropdown.visible:
            self._save_dropdown.objects = [self._build_save_content()]
            self._save_dropdown.visible = True
            self._open_dropdown.visible = False
            self._checkpoint_dropdown.visible = False
        else:
            self._save_dropdown.visible = False

    def _build_save_content(self) -> pn.Column:
        def _close():
            self._save_dropdown.visible = False

        cancel_btn = pn.widgets.Button(name="Cancel", button_type="light", width=70,
                                       styles={"font-size": "13px"})
        cancel_btn.on_click(lambda e: _close())

        initial = (os.path.dirname(self.project.filepath)
                   if self.project.filepath else os.getcwd())
        suggestion = os.path.join(initial, "project.yaml")

        path_input = pn.widgets.TextInput(
            value=self.project.filepath or suggestion,
            placeholder="Path to save .yaml file…",
            sizing_mode="stretch_width",
            stylesheets=_FILEPATH_SS,
        )
        error_pane = pn.pane.HTML("", visible=False, sizing_mode="stretch_width")

        save_btn   = pn.widgets.Button(name="Save", button_type="primary", width=65,
                                       styles={"font-size": "13px"})
        browse_btn = _make_browse_btn("save", path_input)

        def _do_save(e=None):
            p = path_input.value.strip()
            if not p:
                return
            error_pane.visible = False
            try:
                self.save_yaml(path=p)
                _close()
            except Exception as ex:
                error_pane.object = (
                    f'<div style="color:#e74c3c;font-size:12px;padding:4px 0;">'
                    f'Error: {ex}</div>'
                )
                error_pane.visible = True

        save_btn.on_click(_do_save)
        path_input.param.watch(lambda e: _do_save(), "value")

        return pn.Column(
            pn.Row(
                pn.pane.HTML(
                    '<div style="font-size:11px;font-weight:600;color:#7a8caa;'
                    'letter-spacing:0.08em;">SAVE AS</div>',
                    sizing_mode="stretch_width",
                ),
                cancel_btn,
                styles={"padding": "10px 12px 6px", "align-items": "center"},
            ),
            pn.Row(
                path_input, browse_btn, save_btn,
                styles={"padding": "0 12px 4px", "gap": "6px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            pn.Row(
                error_pane,
                styles={"padding": "0 12px 10px"},
                sizing_mode="stretch_width",
            ),
        )

    # ── Checkpoint dropdown ─────────────────────────────────────────────────────
    # Two-step "start a project from a checkpoint" flow. Step 1 picks a
    # *project* (recents + browse, same list as Open — no separate "recent
    # checkpoints" list); step 2 scans that project's output folder for
    # checkpoints and lists them. See gui/TODO.md "Checkpoint / Resume UX".

    def _toggle_checkpoint_dropdown(self):
        if not self._checkpoint_dropdown.visible:
            self._checkpoint_dropdown.objects = [self._build_checkpoint_content()]
            self._checkpoint_dropdown.visible = True
            self._open_dropdown.visible = False
            self._save_dropdown.visible = False
        else:
            self._checkpoint_dropdown.visible = False

    def _build_checkpoint_content(self) -> pn.Column:
        _ROW_BTN_SS = ["""
        :host { display:block; }
        .bk-btn { width:100% !important; text-align:left !important;
            background:transparent !important; border:none !important;
            border-radius:3px !important; color:#8a9bb5 !important;
            font-size:12px !important; padding:4px 8px !important;
            cursor:pointer !important; overflow:hidden;
            text-overflow:ellipsis !important; white-space:nowrap !important; }
        .bk-btn:hover { background:rgba(74,158,255,0.08) !important;
            color:#c8d6e5 !important; }
        """]

        body = pn.Column(styles={"padding": "0 0 4px"})

        def _close():
            self._checkpoint_dropdown.visible = False

        def _step1():
            cancel_btn = pn.widgets.Button(name="Cancel", button_type="light", width=70,
                                           styles={"font-size": "13px"})
            cancel_btn.on_click(lambda e: _close())

            path_input = pn.widgets.TextInput(
                placeholder="Path to .yaml file…",
                sizing_mode="stretch_width",
                stylesheets=_FILEPATH_SS,
            )
            error_pane = pn.pane.HTML("", visible=False, sizing_mode="stretch_width")
            browse_btn = _make_browse_btn("open", path_input)
            next_btn = pn.widgets.Button(name="Next →", button_type="primary", width=70,
                                         styles={"font-size": "13px"})

            def _go(path):
                path = (path or "").strip()
                if not path:
                    return
                error_pane.visible = False
                if not os.path.exists(path):
                    error_pane.object = (
                        f'<div style="color:#e74c3c;font-size:12px;padding:4px 0;">'
                        f'File not found: {path}</div>')
                    error_pane.visible = True
                    return
                body.objects = [_step2(os.path.abspath(path))]

            next_btn.on_click(lambda e: _go(path_input.value))
            path_input.param.watch(lambda e: _go(e.new), "value")

            recents_col = pn.Column(styles={"gap": "2px"})
            rows = []
            for p in self._recents:
                r_btn = pn.widgets.Button(name=os.path.basename(p), stylesheets=_ROW_BTN_SS,
                                          sizing_mode="stretch_width")
                r_btn.on_click(lambda e, fp=p: _go(fp))
                rows.append(r_btn)
            if not rows:
                rows.append(pn.pane.HTML(
                    '<div style="color:#556070;font-size:12px;padding:6px 8px;">'
                    'No recent files.</div>'))
            recents_col.objects = rows

            return pn.Column(
                pn.Row(
                    pn.pane.HTML(
                        '<div style="font-size:11px;font-weight:600;color:#7a8caa;'
                        'letter-spacing:0.08em;">RESUME FROM CHECKPOINT</div>',
                        sizing_mode="stretch_width",
                    ),
                    cancel_btn,
                    styles={"padding": "10px 12px 6px", "align-items": "center"},
                ),
                pn.pane.HTML(
                    '<div style="font-size:11px;color:#8a9bb5;padding:0 12px 6px;">'
                    'Pick the project whose run you want to resume.</div>',
                ),
                pn.Row(path_input, browse_btn, next_btn,
                       styles={"padding": "0 12px 4px", "gap": "6px", "align-items": "center"}),
                pn.Row(error_pane, styles={"padding": "0 12px 4px"}, sizing_mode="stretch_width"),
                pn.pane.HTML(
                    '<div style="font-size:11px;font-weight:600;color:#7a8caa;'
                    'letter-spacing:0.08em;padding:4px 12px 4px;">RECENT</div>'
                ),
                pn.Column(recents_col, styles={"padding": "0 8px 10px"}),
            )

        def _step2(project_path: str):
            back_btn = pn.widgets.Button(name="← Back", button_type="light", width=65,
                                         styles={"font-size": "13px"})
            def _go_back(e):
                body.objects = [_step1()]
            back_btn.on_click(_go_back)
            cancel_btn = pn.widgets.Button(name="Cancel", button_type="light", width=70,
                                           styles={"font-size": "13px"})
            cancel_btn.on_click(lambda e: _close())

            error_pane = pn.pane.HTML("", visible=False, sizing_mode="stretch_width")
            ckpt_col  = pn.Column(styles={"gap": "2px"})
            scan_note = pn.pane.HTML("", sizing_mode="stretch_width")

            def _pick(folder, fname):
                error_pane.visible = False
                try:
                    _close()
                    self.load_from_checkpoint(project_path, f"checkpoints/{fname}")
                except Exception as ex:
                    error_pane.object = (
                        f'<div style="color:#e74c3c;font-size:12px;padding:4px 0;">'
                        f'Error: {ex}</div>')
                    error_pane.visible = True
                    self._checkpoint_dropdown.visible = True

            def _do_scan(folder):
                found = scan_checkpoint_dir(os.path.join(folder, "checkpoints"))
                rows = []
                for label_str, fname in found:
                    b = pn.widgets.Button(name=label_str, stylesheets=_ROW_BTN_SS,
                                          sizing_mode="stretch_width")
                    b.on_click(lambda e, f=fname: _pick(folder, f))
                    rows.append(b)
                if not rows:
                    rows.append(pn.pane.HTML(
                        '<div style="color:#556070;font-size:12px;padding:6px 8px;">'
                        'No checkpoints found here.</div>'))
                ckpt_col.objects = rows
                scan_note.object = (
                    f'<div style="font-size:10px;color:#556070;padding:2px 12px;">'
                    f'Scanned: <code style="color:#7a8caa;">'
                    f'{os.path.join(folder, "checkpoints")}</code></div>')

            try:
                src = load_project(project_path)
                initial_folder = os.path.abspath(os.path.join(
                    os.path.dirname(project_path), src.output_folder or "out"))
            except Exception as ex:
                initial_folder = os.path.dirname(project_path)
                error_pane.object = (
                    f'<div style="color:#e74c3c;font-size:12px;padding:4px 0;">'
                    f'Could not read project: {ex}</div>')
                error_pane.visible = True

            browse_folder_btn = pn.widgets.Button(
                name="📁 Browse for a different output folder…",
                button_type="light", sizing_mode="stretch_width",
                styles={"font-size": "12px"},
            )
            def _browse_folder(e):
                folder = _native_dir_dialog(initial_dir=initial_folder)
                if folder:
                    _do_scan(folder)
            browse_folder_btn.on_click(_browse_folder)

            _do_scan(initial_folder)

            return pn.Column(
                pn.Row(
                    pn.pane.HTML(
                        '<div style="font-size:11px;font-weight:600;color:#7a8caa;'
                        'letter-spacing:0.08em;">PICK A CHECKPOINT</div>',
                        sizing_mode="stretch_width",
                    ),
                    back_btn, cancel_btn,
                    styles={"padding": "10px 12px 6px", "align-items": "center", "gap": "4px"},
                ),
                pn.pane.HTML(
                    f'<div style="font-size:11px;color:#8a9bb5;padding:0 12px 6px;">'
                    f'From <code style="color:#7a8caa;">'
                    f'{os.path.basename(project_path)}</code>:</div>',
                ),
                pn.Column(ckpt_col, styles={"padding": "0 8px 4px"}),
                scan_note,
                pn.Row(error_pane, styles={"padding": "4px 12px 4px"}, sizing_mode="stretch_width"),
                pn.Row(browse_folder_btn, styles={"padding": "6px 12px 10px"}),
            )

        body.objects = [_step1()]
        return body

    # ── Full layout ────────────────────────────────────────────────────────────

    def view(self) -> pn.Column:
        open_btn = pn.widgets.Button(name="Open ▾", button_type="light",
                                     width=75, height=30, styles={"font-size": "13px"})
        def _toggle_open(e):
            if not self._open_dropdown.visible:
                self._open_dropdown.objects = [self._build_open_content()]
                self._open_dropdown.visible = True
                self._checkpoint_dropdown.visible = False
            else:
                self._open_dropdown.visible = False
        open_btn.on_click(_toggle_open)

        checkpoint_btn = pn.widgets.Button(name="Checkpoint ▾", button_type="light",
                                           width=105, height=30, styles={"font-size": "13px"})
        checkpoint_btn.on_click(lambda e: self._toggle_checkpoint_dropdown())

        new_btn = pn.widgets.Button(name="New", button_type="light",
                                    width=55, height=30, styles={"font-size": "13px"})
        new_btn.on_click(lambda e: self.new_project())

        save_btn = pn.widgets.Button(name="Save As", button_type="primary",
                                     width=80, height=30, styles={"font-size": "13px"})
        save_btn.on_click(lambda e: self._toggle_save_dropdown())

        quit_btn = pn.widgets.Button(
            name="Quit", width=55, height=30,
            stylesheets=["""
                .bk-btn { background: #1e1e1e !important; color: #8a9bb5 !important;
                    border: 1px solid #2d3748 !important; border-radius: 4px !important;
                    font-size: 13px !important; cursor: pointer !important; }
                .bk-btn:hover { background: #3d1515 !important; color: #e05555 !important;
                    border-color: #7a2020 !important; }
            """],
        )
        def _quit(e):
            import tornado.ioloop
            tornado.ioloop.IOLoop.current().add_callback(
                lambda: tornado.ioloop.IOLoop.current().stop()
            )
        quit_btn.on_click(_quit)

        header = pn.Row(
            pn.pane.HTML(
                '<span style="font-size:17px;font-weight:700;color:#4a9eff;'
                'letter-spacing:-0.02em;white-space:nowrap;">Simudo</span>',
                width=72, height=46,
                styles={"display": "flex", "align-items": "center"},
            ),
            self._filepath_w,
            self._unsaved_badge,
            pn.Spacer(sizing_mode="stretch_width"),
            new_btn, open_btn, checkpoint_btn, save_btn, quit_btn,
            height=46,
            styles={
                "background": "#111827",
                "border-bottom": "1px solid #1e2d3d",
                "padding": "0 12px",
                "align-items": "center",
                "gap": "8px",
                "flex-shrink": "0",
            },
            sizing_mode="stretch_width",
        )

        body = pn.Row(
            self._sidebar_col,
            self._content_col,
            sizing_mode="stretch_both",
            styles={"overflow": "hidden", "flex": "1"},
        )

        return pn.Column(
            header,
            self._open_dropdown,       # inline, collapses to 0 height when hidden
            self._save_dropdown,       # inline, collapses to 0 height when hidden
            self._checkpoint_dropdown, # inline, collapses to 0 height when hidden
            self._error_banner,    # red dismissable error strip (hidden until needed)
            self._save_banner,     # amber warning until a filepath is set
            body,
            sizing_mode="stretch_both",
            styles={"background": "#0f1623", "overflow": "hidden"},
        )


# ── Entry point ────────────────────────────────────────────────────────────────

_app = SimudoApp()

# Priority: env var > CLI args > most-recent file > bundled example
# SIMUDO_GUI_LOAD_FILE is intended for scripted/automated invocations (e.g. screenshot tools)
# where the Panel serve CLI arg mechanism does not reliably propagate sys.argv.
_env_yaml = os.environ.get("SIMUDO_GUI_LOAD_FILE", "").strip() or None
_cli_yaml = _env_yaml or next(
    (a for a in sys.argv[1:] if a.endswith(".yaml") and os.path.exists(a)), None
)
if _cli_yaml:
    try:
        _app.load_yaml(_cli_yaml)
    except Exception as e:
        print(f"Could not load {_cli_yaml}: {e}")
else:
    _loaded = False
    for _recent in _load_recents():
        if os.path.exists(_recent):
            try:
                _app.load_yaml(_recent)
                _loaded = True
                break
            except Exception as e:
                print(f"Could not load recent {_recent}: {e}")
    if not _loaded:
        _example = os.path.join(os.path.dirname(__file__), "fourlayer_example.yaml")
        if os.path.exists(_example):
            try:
                _app.load_yaml(_example, add_to_recents=False)
            except Exception as e:
                print(f"Could not auto-load example: {e}")

_app.view().servable(title="Simudo")
