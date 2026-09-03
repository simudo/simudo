"""
Runner backend abstraction.

Each backend knows how to launch simudo_1d_runner.py in a particular
execution environment (local subprocess, Docker, SSH, SSH+Docker) and
stream its stdout back as a line iterator.

Adding a new backend only requires subclassing RunnerBackend and implementing
launch().  The Simulation panel is backend-agnostic.
"""

from __future__ import annotations
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import unicodedata
from datetime import date
from typing import Iterator, List, Optional


def _sanitize_path(s: str) -> str:
    """Strip invisible Unicode format characters (U+2060 word joiner,
    U+200B zero-width space, U+FEFF BOM, etc.) that can sneak into
    path strings via copy-paste from terminals or web pages.
    All characters in Unicode category Cf (format/invisible) are removed.
    """
    return "".join(ch for ch in s if unicodedata.category(ch) != 'Cf').strip()


# ── Run handles ───────────────────────────────────────────────────────────────

class RunHandle:
    """Wraps an active subprocess.  Subclasses override path-translation and
    post-run hooks for environment-specific behaviour (Docker, SSH, etc.)."""

    def __init__(self, proc: subprocess.Popen, cmd_str: str):
        self.proc = proc
        self.cmd_str = cmd_str

    def stdout_lines(self) -> Iterator[str]:
        """Yield stdout+stderr lines until the process exits."""
        for line in self.proc.stdout:
            yield line
        self.proc.wait()

    @property
    def returncode(self) -> Optional[int]:
        return self.proc.returncode

    def stop(self):
        try:
            self.proc.terminate()
        except Exception:
            pass

    #: Seconds between automatic intermediate syncs while the run is active.
    #: 0 means no periodic sync (default for local and Docker runs).
    periodic_sync_interval: int = 0

    def do_intermediate_sync(self) -> "Iterator[str]":
        """Perform one intermediate sync and yield log-line strings.
        Default: no-op (local and Docker runs need no file transfer).
        """
        return iter([])

    def translate_output_path(self, path: str) -> str:
        """Translate an execution-environment path to a host-local path.
        Default: identity (local backend needs no translation).
        """
        return path

    def initial_msgs(self) -> Iterator[str]:
        """Messages to emit before stdout streaming starts (e.g. announce paths).
        Default: nothing.
        """
        return iter([])

    def finalize_msgs(self) -> Iterator[str]:
        """Post-run actions.  Yields log-lines while executing (e.g. rsync).
        Default: nothing to do.
        """
        return iter([])


class DockerRunHandle(RunHandle):
    """RunHandle that translates Docker container paths back to host paths."""

    def __init__(self, proc, cmd_str, host_root: str, container_root: str):
        super().__init__(proc, cmd_str)
        self._host_root = host_root.rstrip("/")
        self._container_root = container_root.rstrip("/")

    def translate_output_path(self, path: str) -> str:
        if path.startswith(self._container_root + "/") or path == self._container_root:
            rel = path[len(self._container_root):].lstrip("/")
            return os.path.join(self._host_root, rel)
        return path


class SSHRunHandle(RunHandle):
    """RunHandle for SSH execution.

    Tracks the negotiated remote and local output directories separately.
    Path translation maps remote_out_dir → local_out_dir.
    Post-run: rsyncs remote_out_dir back to local_out_dir, removes the
    hidden staging directory, and optionally deletes the remote output copy.

    Periodic sync: ``do_intermediate_sync()`` rsyncs the remote output
    directory to the local one without any cleanup, so partially-written
    output files (sim_V.csv, info.log, …) appear locally while the run
    is still active.  ``periodic_sync_interval`` is set to 120 s.
    """

    periodic_sync_interval: int = 120

    def __init__(
        self,
        proc: subprocess.Popen,
        cmd_str: str,
        host_spec: str,
        rsync_ssh_opts: List[str],
        remote_out_dir: str,
        local_out_dir: str,
        staging_dir: str,
        delete_remote: bool = False,
    ):
        super().__init__(proc, cmd_str)
        self._host_spec = host_spec
        self._rsync_ssh_opts = rsync_ssh_opts
        self._remote_out_dir = remote_out_dir.rstrip("/")
        self._local_out_dir = local_out_dir.rstrip("/")
        self._staging_dir = staging_dir
        self._delete_remote = delete_remote

    def translate_output_path(self, remote_path: str) -> str:
        rod = self._remote_out_dir
        if remote_path.startswith(rod + "/") or remote_path == rod:
            rel = remote_path[len(rod):].lstrip("/")
            return os.path.join(self._local_out_dir, rel) if rel else self._local_out_dir
        return remote_path

    def initial_msgs(self) -> Iterator[str]:
        yield f"[GUI] Remote output dir: {self._remote_out_dir}\n"
        yield f"[GUI] Local output dir: {self._local_out_dir}\n"

    def do_intermediate_sync(self) -> Iterator[str]:
        """Rsync remote output dir → local output dir without any cleanup.
        Called periodically while the run is active.
        """
        e_args: List[str] = []
        if self._rsync_ssh_opts:
            e_args = ["-e", "ssh " + " ".join(shlex.quote(o) for o in self._rsync_ssh_opts)]
        cmd = (
            ["rsync", "-az"] + e_args
            + [f"{self._host_spec}:{self._remote_out_dir}/",
               f"{self._local_out_dir}/"]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            yield "[GUI] Periodic sync: output synced locally.\n"
        else:
            yield (
                f"[GUI] ⚠ Periodic sync failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}\n"
            )

    def finalize_msgs(self) -> Iterator[str]:
        yield "[GUI] Syncing output files from remote...\n"
        e_args: List[str] = []
        if self._rsync_ssh_opts:
            e_args = ["-e", "ssh " + " ".join(shlex.quote(o) for o in self._rsync_ssh_opts)]
        cmd = (
            ["rsync", "-az"] + e_args
            + [f"{self._host_spec}:{self._remote_out_dir}/",
               f"{self._local_out_dir}/"]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            yield "[GUI] Output synced locally.\n"
        else:
            yield (
                f"[GUI] ⚠ rsync failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}\n"
            )

        # Always remove hidden staging dir
        rm_staging = (
            ["ssh"] + self._rsync_ssh_opts
            + [self._host_spec, f"rm -rf {shlex.quote(self._staging_dir)}"]
        )
        subprocess.run(rm_staging, capture_output=True, text=True)

        if self._delete_remote:
            yield "[GUI] Deleting remote output directory...\n"
            rm_cmd = (
                ["ssh"] + self._rsync_ssh_opts
                + [self._host_spec, f"rm -rf {shlex.quote(self._remote_out_dir)}"]
            )
            rm = subprocess.run(rm_cmd, capture_output=True, text=True)
            if rm.returncode == 0:
                yield "[GUI] Remote output directory deleted.\n"
            else:
                yield (
                    f"[GUI] ⚠ Could not delete remote output: "
                    f"{rm.stderr.strip() or rm.stdout.strip()}\n"
                )


class ExtractorHandle:
    """RunHandle variant for spatial_extractor: separate stdout and stderr."""

    def __init__(self, proc: subprocess.Popen, cmd_str: str):
        self.proc = proc
        self.cmd_str = cmd_str

    def stdout_lines(self) -> Iterator[str]:
        for line in self.proc.stdout:
            yield line
        self.proc.wait()

    def stderr_text(self) -> str:
        return self.proc.stderr.read() if self.proc.stderr else ""

    @property
    def returncode(self) -> Optional[int]:
        return self.proc.returncode

    def stop(self):
        try:
            self.proc.terminate()
        except Exception:
            pass


class SSHExtractorHandle(ExtractorHandle):
    """ExtractorHandle that removes the remote temp directory after extraction."""

    def __init__(self, proc, cmd_str, cleanup_ssh_cmd: List[str]):
        super().__init__(proc, cmd_str)
        self._cleanup_ssh_cmd = cleanup_ssh_cmd

    def stdout_lines(self) -> Iterator[str]:
        yield from super().stdout_lines()
        subprocess.run(self._cleanup_ssh_cmd, capture_output=True, text=True)


# ── Base class ────────────────────────────────────────────────────────────────

class RunnerBackend:
    """Abstract base — subclasses implement launch()."""

    def launch(
        self,
        runner_host_path: str,
        project_yaml_host_path: str,
        lib_dirs_host: List[str],
        extra_args: List[str],
    ) -> RunHandle:
        raise NotImplementedError

    def launch_extractor(
        self,
        extractor_host_path: str,
        output_dir_host_path: str,
        extra_args: List[str] = (),
        local_proj_dir: Optional[str] = None,
    ) -> ExtractorHandle:
        raise NotImplementedError

    def get_core_count(self) -> int:
        """Return the number of available CPU cores in the execution environment."""
        return 2

    @staticmethod
    def _lib_flags(paths: List[str]) -> List[str]:
        flags: List[str] = []
        for p in paths:
            flags += ["--library-dir", p]
        return flags


# ── Local backend ─────────────────────────────────────────────────────────────

class LocalBackend(RunnerBackend):
    """Run the runner directly as a subprocess using the current Python interpreter."""

    def launch(self, runner_host_path, project_yaml_host_path,
               lib_dirs_host, extra_args) -> RunHandle:
        import sys
        cmd = (
            [sys.executable, runner_host_path, project_yaml_host_path]
            + self._lib_flags(lib_dirs_host)
            + extra_args
        )
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True, bufsize=1,
            cwd=os.path.dirname(os.path.abspath(project_yaml_host_path)),
        )
        return RunHandle(proc, cmd_str)

    def launch_extractor(self, extractor_host_path, output_dir_host_path,
                         extra_args=(), local_proj_dir=None) -> ExtractorHandle:
        import sys
        cmd = [sys.executable, extractor_host_path, output_dir_host_path] + list(extra_args)
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        return ExtractorHandle(proc, cmd_str)

    def get_core_count(self) -> int:
        return os.cpu_count() or 2


# ── Docker backend ────────────────────────────────────────────────────────────

class DockerBackend(RunnerBackend):
    """
    Run the runner inside a local Docker container via ``docker exec``.

    A bind-mount must exist mapping ``host_root`` on the host to
    ``container_root`` inside the container.  All host paths under
    ``host_root`` are translated to the equivalent container path
    automatically.

    Works on macOS, Linux, and Windows (with Docker Desktop).  On Windows
    the host_root may be given as a Windows path (``C:\\Users\\...``) or
    with forward slashes (``C:/Users/...``); both are normalised correctly.
    On Cygwin, use the native Cygwin path format (``/cygdrive/c/...``).

    Example:
        host_root      = "/Users/you/Simudo"          # Mac/Linux
        host_root      = "C:/Users/you/Simudo"        # Windows / Cygwin
        container_root = "/home/user/simudo"
        container      = "container_name"
        python_cmd     = "python3"
    """

    def __init__(
        self,
        container: str,
        host_root: str,
        container_root: str,
        python_cmd: str = "python3",
        container_simudo_path: str = "",
    ):
        if not host_root:
            raise ValueError(
                "docker_host_root must be set to the host directory that is "
                "bind-mounted into the container.  Configure it in the "
                "Execution settings panel."
            )
        self.container = container
        # Sanitize: strip invisible Unicode format characters that can enter paths
        # via copy-paste (e.g. U+2060 word joiner, U+200B zero-width space, U+FEFF BOM).
        self.host_root = os.path.normpath(_sanitize_path(host_root))
        self.container_root = container_root.rstrip("/")
        self.python_cmd = python_cmd
        self.container_simudo_path = container_simudo_path.strip()

    def _to_container_path(self, host_path: str) -> str:
        host_path = os.path.normpath(os.path.abspath(host_path))
        # Use normcase so drive-letter case differences are ignored on Windows
        # (e.g. C:\foo and c:\foo are the same path on Windows).
        rel = os.path.relpath(
            os.path.normcase(host_path),
            os.path.normcase(self.host_root),
        )
        if rel.startswith(".."):
            raise ValueError(
                f"Path {host_path!r} is outside the bind-mounted tree "
                f"({self.host_root!r}).\n"
                "Check that 'Host root' in the Execution settings matches "
                "the host-side directory of your Docker bind mount."
            )
        # rel uses os.sep (backslash on Windows); convert to POSIX for the container.
        return posixpath.join(self.container_root, rel.replace(os.sep, "/"))

    def _derive_container_runner_path(self, runner_host_path: str) -> str:
        """Return the container-side path for simudo_1d_runner.py.

        If *container_simudo_path* is set (output of
        ``docker exec <container> python3 -c "import simudo; print(simudo.__file__)"``),
        the runner path is derived from it — same logic as the SSH backend.
        This is needed when Simudo is pip-installed inside the container and
        the runner therefore lives outside the bind-mounted tree.

        Otherwise, fall back to translating *runner_host_path* through the
        bind-mount (the original developer-workflow behaviour).
        """
        return self._derive_container_gui_script_path(runner_host_path)

    def launch(self, runner_host_path, project_yaml_host_path,
               lib_dirs_host, extra_args) -> DockerRunHandle:
        runner_c = self._derive_container_runner_path(runner_host_path)
        yaml_c   = self._to_container_path(project_yaml_host_path)
        lib_flags = self._lib_flags(
            [self._to_container_path(d) for d in lib_dirs_host]
        )
        cmd = (
            ["docker", "exec", self.container,
             self.python_cmd, "-u", runner_c, yaml_c]
            + lib_flags
            + extra_args
        )
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        return DockerRunHandle(proc, cmd_str, self.host_root, self.container_root)

    def _derive_container_gui_script_path(self, script_host_path: str) -> str:
        """Derive the container path for any script that lives in the Simudo
        gui/ directory.  Uses container_simudo_path when set (pip-install case),
        otherwise translates through the bind-mount."""
        if self.container_simudo_path:
            script_name = os.path.basename(script_host_path)
            sp = self.container_simudo_path.strip()
            pkg_dir  = posixpath.dirname(sp)
            code_dir = posixpath.dirname(pkg_dir)
            return posixpath.join(code_dir, "gui", script_name)
        return self._to_container_path(script_host_path)

    def launch_extractor(self, extractor_host_path, output_dir_host_path,
                         extra_args=(), local_proj_dir=None) -> ExtractorHandle:
        extractor_c  = self._derive_container_gui_script_path(extractor_host_path)
        output_dir_c = self._to_container_path(output_dir_host_path)
        cmd = (
            ["docker", "exec", self.container,
             self.python_cmd, "-u", extractor_c, output_dir_c]
            + list(extra_args)
        )
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        return ExtractorHandle(proc, cmd_str)

    def get_core_count(self) -> int:
        try:
            result = subprocess.run(
                ["docker", "exec", self.container, "nproc"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return 2


# ── SSH backend ───────────────────────────────────────────────────────────────

class SSHBackend(RunnerBackend):
    """
    Run the runner on a remote Linux machine via SSH.

    Launch workflow:
      1. Create a unique run directory on the remote under remote_work_dir.
      2. rsync the project YAML to that directory.
      3. For each library dir: if it lives under the local simudo tree, translate
         to the corresponding remote path; otherwise rsync it to a .lib_cache
         sub-directory inside the run directory.
      4. Launch: ssh <host> <python_cmd> -u <runner_path> <remote_yaml> [flags]
      5. Stream stdout/stderr back in real time.
      6. After the process exits (finalize_msgs): rsync the whole remote run
         directory back to local_proj_dir so output files appear locally.

    SSH authentication:
      - Preferred: configure the host in ~/.ssh/config (handles auth, port, etc.).
        Set ssh_host to the config alias; leave ssh_user and ssh_identity_file blank.
      - Alternative: set ssh_host to hostname/IP, ssh_user to your username, and
        optionally ssh_identity_file to a private key path.
      - Password authentication is not supported; set up key-based auth first.

    Remote runner discovery:
      Set ssh_simudo_path to the output of:
          python3 -c "import simudo; print(simudo.__file__)"
      run on the remote machine.  The runner path is derived as:
          <simudo_pkg_parent>/gui/simudo_1d_runner.py
    """

    def __init__(
        self,
        host: str,
        user: str = "",
        identity_file: Optional[str] = None,
        remote_work_dir: str = "~/simudo_runs",
        python_cmd: str = "python3",
        simudo_path: str = "",
        delete_remote: bool = False,
    ):
        self.host = host
        self.user = user
        self.identity_file = identity_file
        self.remote_work_dir = remote_work_dir
        self.python_cmd = python_cmd
        self.simudo_path = simudo_path
        self.delete_remote = delete_remote

    # ── SSH/rsync option helpers ───────────────────────────────────────────────

    @property
    def _host_spec(self) -> str:
        if self.user:
            return f"{self.user}@{self.host}"
        return self.host

    @property
    def _ssh_identity_opts(self) -> List[str]:
        """SSH options for the identity file only (used in rsync -e)."""
        if self.identity_file:
            return ["-i", os.path.expanduser(self.identity_file)]
        return []

    @property
    def _ssh_cmd_opts(self) -> List[str]:
        """Full SSH options for the launch command (BatchMode + identity)."""
        return self._ssh_identity_opts + [
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=30",
        ]

    def _rsync_e_opt(self) -> List[str]:
        """The -e flag for rsync if we need custom SSH options."""
        ssh_opts = self._ssh_identity_opts + ["-o", "BatchMode=yes"]
        return ["-e", "ssh " + " ".join(shlex.quote(o) for o in ssh_opts)]

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _run_ssh(self, remote_cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
        cmd = ["ssh"] + self._ssh_cmd_opts + [self._host_spec, remote_cmd]
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )

    def _resolve_remote_tilde(self, path: str) -> str:
        """Replace a leading ~ with the remote user's actual home directory.

        Needed because Python's Path.resolve() does not expand tildes, so any
        path containing ~ must be made absolute before being passed to the
        remote Python process as a command-line argument.
        One extra SSH round-trip (`echo $HOME`) is made the first time this is
        called; the result is cached for the lifetime of this backend instance.
        """
        if not (path == "~" or path.startswith("~/")):
            return path
        if not hasattr(self, "_remote_home"):
            result = self._run_ssh("echo $HOME")
            self._remote_home = result.stdout.strip()
        home = self._remote_home
        if path == "~":
            return home
        return home.rstrip("/") + "/" + path[2:]

    def _rsync_to_remote(self, local_src: str, remote_dest_dir: str, timeout: int = 120):
        """rsync local_src (file or dir/) to remote_dest_dir/."""
        cmd = (
            ["rsync", "-az"] + self._rsync_e_opt()
            + [local_src, f"{self._host_spec}:{remote_dest_dir}/"]
        )
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)

    # ── Path derivation ───────────────────────────────────────────────────────

    def _derive_remote_paths(self) -> tuple[str, str]:
        """
        Return (remote_runner_path, remote_code_dir) derived from self.simudo_path.

        simudo_path is the output of `python3 -c "import simudo; print(simudo.__file__)"`.
        For a dev install at ~/simudo/code/simudo/__init__.py:
            remote_code_dir   = ~/simudo/code/
            remote_runner_path = ~/simudo/code/gui/simudo_1d_runner.py
        """
        if not self.simudo_path:
            raise ValueError(
                "SSH backend: 'Simudo path' is not configured.\n"
                "On the remote machine, run:\n"
                "    python3 -c 'import simudo; print(simudo.__file__)'\n"
                "and paste the result into the 'Simudo path' field in "
                "Execution settings."
            )
        sp = self.simudo_path.strip()
        remote_pkg_dir  = posixpath.dirname(sp)   # …/code/simudo/
        remote_code_dir = posixpath.dirname(remote_pkg_dir)  # …/code/
        remote_runner   = posixpath.join(remote_code_dir, "gui", "simudo_1d_runner.py")
        return remote_runner, remote_code_dir

    def _translate_lib_dirs(
        self,
        lib_dirs_host: List[str],
        staging_dir: str,
    ) -> tuple[List[str], List[tuple[str, str]]]:
        """Map every host library dir to a .lib_cache/<i>/ inside staging_dir."""
        remote_lib_dirs: List[str] = []
        rsync_pairs: List[tuple[str, str]] = []
        for i, lib_dir in enumerate(lib_dirs_host):
            norm = os.path.normpath(os.path.abspath(lib_dir))
            remote_cache = posixpath.join(staging_dir, f".lib_cache/{i}")
            rsync_pairs.append((norm, remote_cache))
            remote_lib_dirs.append(remote_cache)
        return remote_lib_dirs, rsync_pairs

    def _remote_dir_exists(self, remote_path: str) -> bool:
        try:
            self._run_ssh(f"test -d {shlex.quote(remote_path)}", timeout=15)
            return True
        except subprocess.CalledProcessError:
            return False

    def _negotiate_output_dir(
        self, resolved_work_dir: str, local_proj_dir: str
    ) -> str:
        """Find the first out/<date>/X that doesn't exist on either side."""
        today = date.today()
        candidate = "out/" + today.strftime("%Y%b%d") + "/a"
        for _ in range(1000):
            local_path = os.path.join(local_proj_dir, candidate)
            remote_path = posixpath.join(resolved_work_dir, candidate)
            if not os.path.isdir(local_path) and not self._remote_dir_exists(remote_path):
                return candidate
            candidate = _increment_dir(candidate)
        raise RuntimeError("Could not find a free output directory after 1000 attempts.")

    # ── RunnerBackend interface ───────────────────────────────────────────────

    def launch(
        self,
        runner_host_path: str,
        project_yaml_host_path: str,
        lib_dirs_host: List[str],
        extra_args: List[str],
    ) -> SSHRunHandle:
        if not _command_exists("rsync"):
            raise EnvironmentError(
                "rsync is not installed or not on PATH. "
                "SSH execution requires rsync for file transfer."
            )

        remote_runner, _remote_code_dir = self._derive_remote_paths()
        local_proj_dir = os.path.dirname(os.path.abspath(project_yaml_host_path))

        # Resolve remote work dir (expand ~) so Python on the remote sees an
        # absolute path — Path.resolve() does not expand tildes.
        resolved_work_dir = self._resolve_remote_tilde(self.remote_work_dir)

        # ── Negotiate output directory (free on both sides) ───────────────────
        rel_out = self._negotiate_output_dir(resolved_work_dir, local_proj_dir)
        remote_out_abs = posixpath.join(resolved_work_dir, rel_out)
        local_out_abs = os.path.join(local_proj_dir, rel_out)

        # Pre-create local output directory now, before the run starts.
        os.makedirs(local_out_abs, exist_ok=True)

        # ── Hidden staging directory on remote (YAML + lib_cache only) ───────
        ts = int(time.time())
        yaml_stem = os.path.basename(project_yaml_host_path).replace(".yaml", "")
        staging_dir = posixpath.join(resolved_work_dir, f".staging_{ts}_{yaml_stem}")
        self._run_ssh(f"mkdir -p {shlex.quote(staging_dir)}")

        # Verify runner exists on remote.
        try:
            self._run_ssh(f"test -f {shlex.quote(remote_runner)}")
        except subprocess.CalledProcessError:
            raise FileNotFoundError(
                f"Runner not found on remote at: {remote_runner}\n"
                f"Derived from simudo_path={self.simudo_path!r}.\n"
                "Check the 'Simudo path' field in Execution settings."
            )

        # ── Write YAML with absolute remote output path baked in ──────────────
        import yaml as _yaml
        with open(project_yaml_host_path, encoding='utf-8') as f:
            proj_data = _yaml.safe_load(f)
        if "output" not in proj_data:
            proj_data["output"] = {}
        proj_data["output"]["folder"] = remote_out_abs

        yaml_basename = os.path.basename(project_yaml_host_path)
        tmpdir = tempfile.mkdtemp()
        try:
            tmp_yaml = os.path.join(tmpdir, yaml_basename)
            with open(tmp_yaml, "w", encoding='utf-8') as f:
                _yaml.dump(proj_data, f, default_flow_style=False,
                           allow_unicode=True, sort_keys=False)
            self._rsync_to_remote(tmp_yaml, staging_dir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # ── Rsync library dirs to staging .lib_cache ──────────────────────────
        remote_lib_dirs, rsync_pairs = self._translate_lib_dirs(lib_dirs_host, staging_dir)
        for local_dir, remote_dir in rsync_pairs:
            self._run_ssh(f"mkdir -p {shlex.quote(remote_dir)}")
            self._rsync_to_remote(local_dir.rstrip("/") + "/", remote_dir)

        # ── Build and launch SSH command ──────────────────────────────────────
        remote_yaml_path = posixpath.join(staging_dir, yaml_basename)
        lib_flags = self._lib_flags(remote_lib_dirs)

        # --resume tells the runner to use output.folder exactly (no increment),
        # since we already negotiated a free directory on both sides.
        remote_cmd_parts = (
            [self.python_cmd, "-u", remote_runner, remote_yaml_path]
            + lib_flags
            + ["--resume"]
            + extra_args
        )
        remote_cmd = " ".join(shlex.quote(c) for c in remote_cmd_parts)
        ssh_cmd = ["ssh"] + self._ssh_cmd_opts + [self._host_spec, remote_cmd]
        cmd_str = " ".join(shlex.quote(c) for c in ssh_cmd)

        proc = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        return SSHRunHandle(
            proc, cmd_str,
            host_spec=self._host_spec,
            rsync_ssh_opts=self._ssh_identity_opts + ["-o", "BatchMode=yes"],
            remote_out_dir=remote_out_abs,
            local_out_dir=local_out_abs,
            staging_dir=staging_dir,
            delete_remote=self.delete_remote,
        )

    def get_core_count(self) -> int:
        try:
            result = self._run_ssh("nproc", timeout=5)
            return int(result.stdout.strip())
        except Exception:
            pass
        return 2

    def launch_extractor(
        self,
        extractor_host_path: str,
        output_dir_host_path: str,
        extra_args=(),
        local_proj_dir: Optional[str] = None,
    ) -> SSHExtractorHandle:
        """Run the extractor on the remote host.

        Always rsyncs the local output directory to a temporary directory on
        the remote before running the extractor, so the files are guaranteed
        to be present regardless of whether delete_remote was set or whether
        the run was done with a different backend.  The temp directory is
        removed automatically when stdout_lines() drains.
        """
        remote_runner, _ = self._derive_remote_paths()
        remote_extractor = posixpath.join(
            posixpath.dirname(remote_runner), "spatial_extractor.py"
        )

        resolved_work_dir = self._resolve_remote_tilde(self.remote_work_dir)
        ts = int(time.time())
        remote_tmp = posixpath.join(resolved_work_dir, f".extract_{ts}")
        self._run_ssh(f"mkdir -p {shlex.quote(remote_tmp)}")

        # rsync the local output dir contents to the remote temp dir
        local_src = output_dir_host_path.rstrip("/") + "/"
        rsync_cmd = (
            ["rsync", "-az"] + self._rsync_e_opt()
            + [local_src, f"{self._host_spec}:{remote_tmp}/"]
        )
        subprocess.run(rsync_cmd, check=True, capture_output=True, text=True)

        remote_cmd_parts = (
            [self.python_cmd, "-u", remote_extractor, remote_tmp]
            + list(extra_args)
        )
        remote_cmd = " ".join(shlex.quote(c) for c in remote_cmd_parts)
        ssh_cmd = ["ssh"] + self._ssh_cmd_opts + [self._host_spec, remote_cmd]
        cmd_str = " ".join(shlex.quote(c) for c in ssh_cmd)
        proc = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        cleanup_cmd = (
            ["ssh"] + self._ssh_identity_opts + ["-o", "BatchMode=yes"]
            + [self._host_spec, f"rm -rf {shlex.quote(remote_tmp)}"]
        )
        return SSHExtractorHandle(proc, cmd_str, cleanup_cmd)


# ── SSH + Docker backend (stub) ───────────────────────────────────────────────

class SSHDockerBackend(RunnerBackend):
    """SSH to a remote host, then docker exec inside a container there.
    Not yet implemented."""

    def launch(self, runner_host_path, project_yaml_host_path,
               lib_dirs_host, extra_args) -> RunHandle:
        raise NotImplementedError("SSHDockerBackend is not yet implemented.")

    def launch_extractor(self, extractor_host_path, output_dir_host_path,
                         extra_args=()) -> ExtractorHandle:
        raise NotImplementedError("SSHDockerBackend is not yet implemented.")


# ── Factory ───────────────────────────────────────────────────────────────────

def backend_from_profile(profile) -> RunnerBackend:
    """Build the appropriate RunnerBackend from an ExecutionProfile dataclass."""
    t = profile.type
    if t == "local":
        return LocalBackend()
    if t == "docker":
        return DockerBackend(
            container=profile.docker_container,
            host_root=profile.docker_host_root,
            container_root=profile.docker_container_root,
            python_cmd=profile.python_cmd,
            container_simudo_path=getattr(profile, 'docker_simudo_path', ''),
        )
    if t == "ssh":
        return SSHBackend(
            host=profile.ssh_host,
            user=profile.ssh_user,
            identity_file=profile.ssh_identity_file or None,
            remote_work_dir=profile.ssh_remote_work_dir,
            python_cmd=profile.python_cmd,
            simudo_path=profile.ssh_simudo_path,
            delete_remote=profile.ssh_delete_remote_after_sync,
        )
    if t == "ssh+docker":
        return SSHDockerBackend()
    raise ValueError(f"Unknown execution profile type: {t!r}")


# ── Utility ───────────────────────────────────────────────────────────────────

def _increment_dir(cur_dir: str) -> str:
    """Increment the last alphanumeric character of a path string.
    Mirrors simudo_1d_runner._increment_dir so the GUI can replicate the
    runner's directory-naming logic without importing the runner.
    """
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
            return cur_dir[:last_num_pos] + str(last_num + 1) + cur_dir[last_num_pos + num_len:]
        return cur_dir
    else:
        return cur_dir[:last_pos] + chr(ord(last_c) + 1) + cur_dir[last_pos + 1:]


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _quote_remote_path(path: str) -> str:
    """Quote a remote path for embedding in an SSH shell command.

    Preserves a leading `~/` unquoted so the remote shell expands it to $HOME.
    Plain shlex.quote would produce `'~/...'` which suppresses tilde expansion.
    """
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)
