# Simudo runtime image.
#
# Created by Claude Opus 5 (Anthropic), 2026-08-17.
#
# Ubuntu 24.04 LTS (noble), which carries legacy DOLFIN 2019.2.0 in universe
# (python3-dolfin 2019.2.0~legacy20240219.1c52e83) and is supported until
# April 2029.
#
# DOLFIN is rebuilt from the distribution's own source package with one patch
# applied -- see patches/dolfin-systemassembler-facet-wise-assembly.patch and
# README.md.  Without it, forms that combine an interior facet measure (dS)
# with a subdomain-restricted dx assemble against the wrong cell, giving
# silently wrong answers on some 2D problems rather than an error.
#
# Build (native architecture):
#     docker build -t ecdee/simudo:0.7.0.3 docker/
#
# Build for a different architecture (slow -- emulated):
#     docker buildx build --platform linux/amd64 -t ecdee/simudo:0.7.0.3 docker/
#
# See README.md for publishing a multi-architecture tag.

FROM ubuntu:24.04

# Number of parallel compile jobs for the DOLFIN rebuild.
ARG MAKE_J=4
# What to install Simudo from.  A released version by default; pass e.g.
# --build-arg SIMUDO_SPEC=/wheels/simudo-0.7.0.3-py3-none-any.whl to use a
# locally built wheel (mount it with --build-context or COPY it in).
ARG SIMUDO_SPEC=simudo==0.7.0.3
ARG FOR_DOCS=

ENV DEBIAN_FRONTEND=noninteractive

# Set up apt repository caching
# use `buildah ... --build-arg APT_HTTP_PROXY=http://172.17.0.1:3142`
ARG APT_HTTP_PROXY=
COPY _container.sh /usr/local/bin/
RUN _container.sh apt-upgrade

# ---------------------------------------------------------------------------
# Runtime and build dependencies.
#
# python3-dolfin is installed from apt for its dependency tree (petsc4py,
# ufl_legacy, FIAT, ffc); the C++ library and Python module are then rebuilt
# from source below and installed over it into /usr/local.
# ---------------------------------------------------------------------------
RUN apt-get -qq update && \
    if [ -n "$FOR_DOCS" ]; then DOC_EXTRA="python3-myst-parser texlive-latex-base texlive-latex-recommended texlive-latex-extra dvipng"; else DOC_EXTRA=; fi && \
    apt-get install -y --no-install-recommends \
        sudo iproute2 fossil nano emacs-nox less \
        python3-dev python3-pip \
        build-essential cmake zip unzip parallel cython3 \
        jupyter jupyter-notebook jupyter-nbconvert python3-ipykernel \
        python3-sphinx python3-sphinx-rtd-theme \
        python3-argh python3-atomicwrites python3-cached-property \
        python3-dolfin python3-future python3-h5py \
        python3-matplotlib python3-meshio python3-pandas python3-petsc4py \
        python3-pint python3-pprofile python3-pygmsh python3-pytest \
        python3-scipy python3-sortedcontainers python3-tables \
        python3-tabulate python3-tqdm python3-yaml \
        python3-yamlordereddictloader python3-attr python3-sympy \
        python3-setuptools \
        gmsh meshio-tools optipng poppler-utils $DOC_EXTRA && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Rebuild DOLFIN from the distribution source with the SystemAssembler patch.
#
# The cmake flags mirror the ones Debian builds the package with, so the
# result stays ABI-compatible with the apt-installed dependants above.
# ---------------------------------------------------------------------------
COPY patches/ /usr/local/src/simudo-patches/

RUN if [ -z "$FOR_DOCS" ]; then sed -i 's/^Types: deb$/Types: deb deb-src/' \
        /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get -qq update && \
    apt-get install -y --no-install-recommends dpkg-dev patch && \
    apt-get build-dep -y dolfin && \
    mkdir -p /usr/local/src/dolfin && cd /usr/local/src/dolfin && \
    apt-get source dolfin && \
    ln -s dolfin-*/ src && \
    cd src && \
    patch -p1 < /usr/local/src/simudo-patches/dolfin-systemassembler-facet-wise-assembly.patch && \
    cd .. && mkdir bld && cd bld && \
    cmake -GUnix\ Makefiles \
        -D CMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
        -D CMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON \
        -D CMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
        -D CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON \
        -D FETCHCONTENT_FULLY_DISCONNECTED=ON \
        -D CMAKE_INSTALL_RUNSTATEDIR=/run \
        -D CMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON \
        -D BUILD_SHARED_LIBS:BOOL=ON \
        -D DOLFIN_ENABLE_TRILINOS:BOOL=OFF \
        -D DOLFIN_ENABLE_HDF5:BOOL=ON \
        -D HDF5_C_COMPILER_EXECUTABLE:FILEPATH=/usr/bin/h5pcc \
        -D DOLFIN_ENABLE_PARMETIS:BOOL=OFF \
        -D DOLFIN_ENABLE_SCOTCH:BOOL=ON \
        -D DOLFIN_ENABLE_DOCS:BOOL=OFF \
        -D DOLFIN_ENABLE_MPI:BOOL=ON \
        -D MPIEXEC_PARAMS:STRING=--oversubscribe \
        -D CMAKE_CXX_FLAGS:STRING=-fpermissive \
        -D "DOLFIN_EXTRA_CXX_FLAGS:STRING=-g -O2 -fstack-protector-strong -fstack-clash-protection -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2" \
        ../src && \
    make -j"$MAKE_J" && \
    make install && ldconfig && \
    cd ../src/python && \
    pip3 install --no-index --break-system-packages --no-build-isolation --no-deps . && \
    cd / && rm -rf /usr/local/src/dolfin && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*; fi

# ---------------------------------------------------------------------------
# Simudo itself.  Pure-Python dependencies not packaged by Ubuntu
# (mpl_render, suffix_trees, generic_escape, ...) come from PyPI.
# ---------------------------------------------------------------------------
RUN mkdir -p /input && \
    pip3 install --break-system-packages --find-links /input "$SIMUDO_SPEC" && \
    if [ -n "$FOR_DOCS" ]; then \
        mkdir -p /tmp/simudo-download /opt/simudo && \
        pip3 download --no-deps --no-binary :all: --find-links /input -d /tmp/simudo-download "$SIMUDO_SPEC" && \
        tar --strip-components=1 -xf /tmp/simudo-download/simudo*.tar.gz -C /opt/simudo; \
    fi && \
    rm -rf /root/.cache/pip /tmp/simudo-download

RUN if [ -n "$FOR_DOCS" ]; then cd /opt/simudo && ./build_docs.sh; fi


# ---------------------------------------------------------------------------
# Unprivileged user, with passwordless sudo inside the container.
# ---------------------------------------------------------------------------
RUN useradd -m -s /bin/bash user && \
    echo "user:docker" | chpasswd && \
    echo "user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

WORKDIR /home/user
USER user

# Fail the build if the apt module, rather than the rebuilt one, is what gets
# imported -- that would mean the patch is present in the image but not in use.
RUN if [ -z "$FOR_DOCS" ]; then python3 -c "import dolfin, sys; \
    print('dolfin', dolfin.__version__, 'from', dolfin.__file__); \
    sys.exit(0 if dolfin.__file__.startswith('/usr/local/') else \
        'ERROR: imported the apt-packaged dolfin, not the patched rebuild')"; fi

CMD ["/bin/bash"]
