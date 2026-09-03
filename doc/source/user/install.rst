
Installation
&&&&&&&&&&&&

So far we have tested Simudo on Ubuntu Linux 24.04 LTS, which is what our
Docker image is built on, and on macOS and Windows 10/11 via Docker.

Simudo runs on top of FEniCS, the finite element library. FEniCS is **not** a
Python package and must already be present before Simudo is installed: Simudo
itself installs with :command:`pip`, but FEniCS does not — it comes from your
system's package manager, or ready-made inside our Docker image. FEniCS first,
Simudo second.

Installing has **two steps**, and most users want both.

**Step 1 — somewhere for the calculations to run.** Choose one:

`Docker (for macOS and Windows)`_
  The least work, and it works on macOS, Windows and Linux. The image already
  contains FEniCS *and* Simudo, so there is nothing to install and nothing to
  order correctly.

`Ubuntu Linux`_
  Simudo on the machine itself: one :command:`apt` command for FEniCS and the
  other system packages, then one :command:`pip3` command for Simudo.

**Step 2 — the point-and-click interface**, which we recommend: see
`Graphical User Interface (GUI)`_. It runs on your own computer and sends
calculations to whatever you set up in step 1, so do step 1 first.

If you have trouble with installation, please feel free to
`send us an email <mailto:jkrich@uottawa.ca>`_.

Ubuntu Linux
============

First, you need to install the dependencies that are available in
Ubuntu's repositories — this is where FEniCS/dolfin comes from.
Copy-paste this entire command in your terminal, and run it::

  sudo apt install build-essential zip unzip parallel cython3 \
    python3-{argh,atomicwrites,cached-property,dolfin,future,h5py} \
    python3-{matplotlib,meshio,pandas,petsc4py,pint,pprofile,pytest} \
    python3-{scipy,sortedcontainers,sphinx,sphinx-rtd-theme,tabulate} \
    python3-{tqdm,yaml,yamlordereddictloader} python3-pip \
    optipng poppler-utils meshio-tools gmsh

.. note::

   The ``python3-dolfin`` package carries a bug that can affect 2-D
   simulations. If you are only running 1-D simulations, it does not affect
   you. If you want to run 2-D simulations, our Docker image already has the
   fix applied; to patch a native installation yourself, see
   ``docker/README.md`` in the Simudo source.

Now that FEniCS is in place, install the remaining dependencies and Simudo
itself::

  pip3 install suffix_trees mpl_render generic_escape simudo

That's it! Simudo is now installed, with everything needed to run a
simulation and the bundled examples.

To update to the latest version of Simudo later::

  pip3 install --upgrade simudo

Optionally, if you want to build your own 2-D geometry with gmsh, pygmsh or
mshr, add::

  pip3 install 'simudo[mesh]'

Simudo does not require any particular mesh generator, and nothing above needs
one. The 1-D layered devices built by the examples and by the graphical
interface are meshed directly, without gmsh.


Docker (for macOS and Windows)
==============================

We have installed Simudo using Docker on macOS and Windows 10/11 systems,
and similar steps will likely work on Linux, though we have not tested there.

Whatever your system, install `Docker <https://docker.com>`_ (Docker
Desktop recommended).

On Linux it is also possible (and maybe more secure) to use podman
instead of docker.

GNU+Linux
---------

Navigate to a directory where you would like to run Simudo
(home directory not recommended) and run::

  podman pull docker.io/ecdee/simudo:latest

  podman run -ti --rm -v $PWD:/home/user/shared/ --uidmap 0:1:999 --uidmap 1000:0:1 --uidmap 1001:1001:64536 ecdee/simudo:latest

The "--uidmap" stuff is necessary for the permissions to work correctly. If you find the command too long, please note that you can create yourself an alias by adding to your ~/.bashrc something like::

  alias simu='podman run -ti --rm -v $PWD:/home/user/shared/ --uidmap 0:1:999 --uidmap 1000:0:1 --uidmap 1001:1001:64536 ecdee/simudo:latest'

so that you can directly run ``simu``. The podman commands are otherwise identical or very similar to the docker ones.

macOS
-----

Open a Terminal and navigate to a directory where you would like to run Simudo
(home directory not recommended) and run::

  docker pull ecdee/simudo:latest

  docker run -ti --name simudo -v $(pwd):/home/user/simudo/ ecdee/simudo:latest

Windows 10/11
-------------

We have used both `cygwin <https://www.cygwin.com>`_ and Powershell.

If using cygwin, open a cygwin terminal and navigate to a directory where you
would like to run Simudo. The directory should be a subdirectory of your cygwin
root directory, which you can find with::

  cygpath -w ~

Once you are in the desired directory, run::

  cygpath -w $(pwd)

The output could be something like :file:`C:\\cygwin64\\home\\username\\simudo`.
Reformat that output into the form :file:`//c/cygwin64/home/username/simudo`,
which we will call :file:`FOLDERNAME` below. Use, for example, :file:`//d/`
in place of :file:`//c/` if the directory is on :file:`D:\\`.
Create the container by::

  docker pull ecdee/simudo:latest

  docker run -ti --name simudo -v FOLDERNAME:/home/user/simudo/ ecdee/simudo:latest

where :file:`FOLDERNAME` is replaced by whatever folder you found before.

Using Powershell, the equivalent command is::

  docker run -ti  --name simudo -v C:\path\to\directory:/home/user/simudo/ ecdee/simudo:latest

and Powershell allows the linked directory to be anywhere in the
Windows file system.

Inside the docker
-----------------

You should find yourself inside the docker, running Ubuntu 24.04, with Simudo
installed and ready to use. Everything in the directory in which you executed
these commands will be available in the docker at :file:`/home/user/simudo/` and
vice versa.

When you finish your session, you can `exit` from the docker. To
enter it again, you can run::

  docker start simudo

  docker attach simudo

There are many resources about using Docker, and we recommend starting
with the
`FEniCS pages <https://fenics-containers.readthedocs.io/en/latest/>`_.
For example, if you would like to run Jupyter notebooks that execute
inside the docker, you can arrange the required port forwarding by changing
the above :command:`docker run` command to::

  docker run -ti --name simudo \
    -p 127.0.0.1:8888:8888 \
    -v $(pwd):/home/user/simudo/ ecdee/simudo:latest

for macOS. For Windows, replace ``$(pwd)`` with FOLDERNAME.

Graphical User Interface (GUI)
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

The Simudo GUI provides a point-and-click interface for setting up and
running 1D simulations. The GUI runs on your local machine and submits
calculations to a Simudo execution environment (a Docker container,
another machine connected by SSH, or a local installation).

**The GUI is strongly recommended.** It is the easiest way to set up a
simulation, and most users should install it.

Prerequisites
=============

**You do not need to install DOLFIN on the machine running the GUI.**
The GUI is a web front end: it reads and writes project files and launches
the solver in your execution environment (eg, Docker or remote machine).

Install only the GUI's own dependencies on the host computer (not the Docker).
The GUI requires Python 3.10+ with:

- `panel <https://panel.holoviz.org>`_ ≥ 1.4
- `param`
- `numpy`
- `scipy`
- `matplotlib`
- `bokeh`
- `pyyaml`

A recommended way to install these is with conda::

  conda create -n simudo_gui python=3.11 \
    panel param numpy scipy matplotlib bokeh pyyaml -c conda-forge

  conda activate simudo_gui

The GUI code is part of the Simudo package, so you also need a copy of
that package on the host machine. Install Simudo *without* its dependencies — you are
installing it for the interface files, not to run Simudo ::

  pip install --no-deps simudo

This does not pull in FEniCS, petsc4py or anything else from the list at the
top of this page.

.. note::

   **Working from a development version instead of a pip install?** The GUI modules
   import each other by package path (``simudo.gui.model``), so ``simudo``
   has to be importable. From the top directory of your source clone::

     pip install -e . --no-deps

   That makes the clone importable without pulling in FEniCS, and you can then
   launch exactly as described below.

You also need a working Simudo execution environment; that is where Simudo
*is* installed. The easiest option is the Docker image described above. Make
sure Docker is running and the Simudo container exists before launching the
GUI, and note its container name: the GUI asks for it on the Simulation panel
and cannot run anything without it.

.. note::

   If one machine is doing both jobs — for example an Ubuntu workstation where
   you followed `Ubuntu Linux`_ above — you do not need a separate environment.
   Install the graphical interface alongside Simudo instead::

     pip3 install 'simudo[gui]'

Starting the GUI
================

Activate the conda environment and start the GUI **from whatever directory you
want to keep your project files in**::

  conda activate simudo_gui
  cd ~/my-simudo-projects
  python -m simudo.gui --show --port 5006

This will open a browser tab at ``http://localhost:5006/app``.
If ``--show`` does not open a browser automatically, open that
URL yourself in your preferred browser.

The GUI's Open and Save dialogs start in the directory you launched from.

To open a particular project yaml file at startup, pass its path::

  python -m simudo.gui my_device.yaml --show --port 5006

Without one, the GUI reopens your most recent project, or a bundled example on
first run. Every other argument is handed to :command:`panel serve`, so its
options (``--port``, ``--address``, ``--allow-websocket-origin``, ...) all work.

If it exits with ``ModuleNotFoundError: No module named 'simudo'``, then
Simudo is not importable from this environment — either the
:command:`pip install --no-deps simudo` step above was skipped, or it went
into a different environment from the one that is now active.

Once the GUI is running, the :doc:`gui_tutorial` shows you how to build two complete
devices from scratch — a silicon p-n diode and an intermediate-band cell
with a heterojunction — and is the fastest way to learn the interface.

Configuring the execution backend
==================================

The first time you run a simulation you need to tell the GUI how to
reach your Simudo installation. Open the **Simulation** panel and click
**Execution profile**. For a local Docker container:

- **Execution type**: Docker
- **Container name**: the name of your running Docker container
  (e.g. ``simudo``)
- **Host root**: the host-side directory that is mounted into the
  container (the ``-v`` path you used with :command:`docker run`)
- **Container root**: the corresponding path inside the container
  (e.g. ``/home/user/simudo``)

Click **Test connection** to verify that the GUI can reach the container,
then **Save as default** to persist the settings to
:file:`~/.simudo_gui.yaml`.

Setting up for developing/modifying Simudo
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Simudo is developed using `fossil <https://fossil-scm.org>`_. If you would
like to install a development version of simudo, after installing
fossil on your system, you can clone the repository using::

  fossil clone https://secure.hydra.ecd.space/eduard/simudo/ simudo.fossil

  fossil open simudo.fossil

You can then enter the simudo directory and install the local version
for development with::

  pip3 install -e . --break-system-packages

Simudo is then imported from your clone, so edits take effect without
reinstalling.  ``--break-system-packages`` is needed because Ubuntu 24.04 marks
its system Python as externally managed (PEP 668); it does not in fact replace
anything here, since the :command:`apt` packages above already satisfy every
dependency.  Inside a virtualenv or conda environment, drop that flag.

(The older ``python3 setup.py develop --user`` still works but is deprecated by
setuptools.)