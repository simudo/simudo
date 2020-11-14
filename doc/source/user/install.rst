
Installation
&&&&&&&&&&&&

So far we have tested Simudo on Debian 10 (Buster), Ubuntu Linux 19.04
and 20.04, and on macOS via Docker.

Simudo depends on some Python packages, as well as

- FEniCS (finite element library)
- GMSH (mesh generator)

Here are some brief installation instructions for Linux and
Docker-based systems. If you have trouble with installation,
please feel free to `send us an email <mailto:jkrich@uottawa.ca>`_.

Ubuntu Linux
============

First, you need to install the dependencies that are available in
Ubuntu's repositories. Copy-paste this entire command in your
terminal, and run it::

  sudo apt install build-essential zip unzip parallel cython3 \
    python3-{argh,atomicwrites,cached-property,dolfin,future,h5py} \
    python3-{matplotlib,meshio,pandas,petsc4py,pint,pprofile,pytest} \
    python3-{scipy,sortedcontainers,sphinx,sphinx-rtd-theme,tabulate} \
    python3-{tqdm,yaml,yamlordereddictloader} python3-pip \
    optipng poppler-utils meshio-tools gmsh

Finally, install the remaining dependencies as well as Simudo itself
from PyPI::

  pip3 install suffix_trees mpl_render generic_escape simudo

That's it! Simudo is now installed.

To update to the latest version of Simudo on PyPI, use ``pip3``::

  $ pip3 install --upgrade simudo


Docker/macOS
============

We have installed Simudo using Docker on macOS, and the same steps
will likely work on Windows or Linux, though we have not tested there.

Install `Docker <https://docker.com>`_ (Docker Desktop recommended)

Open a Terminal and navigate to a directory where you would like to run Simudo
(home directory not recommended) and run::

  docker pull ecdee/simudo:latest

  docker run -ti --name simudo -v $(pwd):/home/user/simudo/ ecdee/simudo:latest

You should find yourself inside the docker, running Ubuntu 20.04, with Simudo
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

  docker run -ti --name simudo -p 127.0.0.1:8888:8888 \
    -v $(pwd):/home/user/simudo/ ecdee/simudo:latest


Setting up for developing/modifying Simudo
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Simudo is developed using `fossil <https://fossil-scm.org>`_. If you would like to 
install a development version of simudo, after installing 
fossil on your system, you can clone the repository using::

  fossil clone https://secure.hydra.ecd.space/eduard/simudo/ simudo.fossil

  fossil open simudo.fossil

You can then enter the simudo directory and install the local version 
for development with::

  python3 setup.py develop --user

