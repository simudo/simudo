# Simudo Docker image

*Created by Claude Opus 5 (Anthropic), 2026-08-17.*

Builds an Ubuntu 24.04 LTS image with legacy FEniCS/DOLFIN 2019.2.0 and Simudo,
with the DOLFIN patch described below applied.

## The DOLFIN patch — why the image rebuilds DOLFIN

`patches/dolfin-systemassembler-facet-wise-assembly.patch` fixes a bug in
`SystemAssembler::facet_wise_assembly`. When an assembled form contains **both**
an interior facet measure (`dS`) and a **subdomain-restricted** `dx`, DOLFIN
extracts the cell integrals for the two cells sharing a facet but keeps only the
second cell's value, so both cells are assembled with the wrong form when they
lie in different subdomains.

**The failure mode is a silently wrong answer, not an error.** Some 2D Simudo
problems are affected. That is why the patch is applied rather than merely
recommended.

The patch is [pull request 576](https://bitbucket.org/fenics-project/dolfin/pull-requests/576)
against upstream legacy DOLFIN, submitted by this project and not yet reviewed by
the maintainers. It is vendored here so the image does not depend on the PR
landing. It is written against upstream commit `1c52e83`, which is the snapshot
Ubuntu 24.04 packages (`2019.2.0~legacy20240219.1c52e83`), and applies cleanly
there.

## Building

```
docker build -t ecdee/simudo:0.7.1.0 docker/
```

Useful build arguments:

| Argument | Default | Purpose |
|---|---|---|
| `MAKE_J` | `4` | Parallel compile jobs for the DOLFIN rebuild |
| `SIMUDO_SPEC` | `simudo==0.7.1.0` | What `pip install` is given — a PyPI spec, or a path to a wheel inside the build context |

To build against a locally built wheel rather than PyPI, copy it into `docker/`
first and pass its path:

```
cp dist/simudo-0.7.1.0-py3-none-any.whl docker/
docker build --build-arg SIMUDO_SPEC=/wheels/simudo-0.7.1.0-py3-none-any.whl \
             -t ecdee/simudo:0.7.1.0 docker/
```

(adjusting the `COPY` in the Dockerfile to place the wheel at `/wheels/`).

## Installing a Simudo wheel into a built image

```
pip3 install --break-system-packages --no-deps simudo-0.7.1.0-py3-none-any.whl
```

`--no-deps` matters. Most of Simudo's dependencies come from apt in this image,
and Debian-packaged Python modules carry no `RECORD` file, so pip refuses to
touch them:

```
ERROR: Cannot uninstall tqdm 0.0.0, RECORD file not found.
       Hint: The package was installed by debian.
```

Anything that makes pip try to reinstall dependencies — `--force-reinstall`
especially — hits this and aborts the whole install, leaving whatever version
was there before. Install the wheel alone; the dependencies are already present.

## Building for another architecture

The image can be built for a platform other than the host's:

```
docker buildx build --platform linux/amd64 -t ecdee/simudo:0.7.1.0 docker/
```

This works anywhere, including on an Apple-silicon Mac, but the DOLFIN
compilation runs under QEMU emulation and is **much** slower. The image itself
is a genuine `linux/amd64` image and runs at native speed on x86 hardware; only
the build is emulated. Building each architecture natively is therefore worth
it if a machine of that architecture is available.

## Publishing a multi-architecture tag

Users should not have to know their own architecture. Build the two images
separately — natively on each — push them under per-architecture tags, then join
them into one multi-architecture tag:

```
docker buildx imagetools create -t ecdee/simudo:0.7.1.0 \
    ecdee/simudo:0.7.1.0-amd64 \
    ecdee/simudo:0.7.1.0-arm64
```

The joined tag is a *manifest list*: a small index that names one image per
platform. `docker pull ecdee/simudo:0.7.1.0` resolves it to the client's own
architecture and downloads only that image, so the multi-architecture tag costs
users nothing in download size.

### Always verify the published manifest

```
docker manifest inspect ecdee/simudo:0.7.1.0
```

and check that every architecture you meant to publish is listed. **Do not treat
this as optional.** When the two architectures are built separately, a tag whose
second half never got built — a failed job, a pipeline that was not triggered —
still publishes perfectly happily as a single-architecture tag. Nothing errors,
and the omission is invisible until a user on the missing architecture cannot
pull it.

This is not hypothetical. The upstream legacy-FEniCS images
(`ghcr.io/scientificcomputing/fenics`) are built exactly this way: amd64 in
GitHub Actions, arm64 on CircleCI, joined with `buildx imagetools`. Their
`2024-05-28` tag carries both architectures; `2025-10-09` carries only amd64 —
while the arm64 build job is still present and unchanged in their repository.
The arm64 half simply stopped arriving, and the tag shipped anyway.

## Applying the patch without Docker

Users who build their own FEniCS, or who run a distribution-packaged DOLFIN,
need the same fix. On a Debian or Ubuntu system with the source package
available:

```
sudo sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
sudo apt-get update
sudo apt-get build-dep dolfin
apt-get source dolfin
cd dolfin-2019.2.0*/
patch -p1 < /path/to/simudo/docker/patches/dolfin-systemassembler-facet-wise-assembly.patch
mkdir ../bld && cd ../bld
cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo -D CMAKE_CXX_FLAGS=-fpermissive ../dolfin-2019.2.0*/
make -j4 && sudo make install && sudo ldconfig
cd ../dolfin-2019.2.0*/python && pip3 install --no-deps .
```

See the `Dockerfile` for the full set of cmake flags, which mirror Debian's own
build so that the result stays compatible with the packaged dependants.

To check which DOLFIN a Python session is using:

```
python3 -c "import dolfin; print(dolfin.__file__)"
```

A path under `/usr/local` is the rebuilt, patched one; `/usr/lib/python3/…` is
the distribution's unpatched package.

## Building the docs

Something like this:

```sh
buildah bud -f Dockerfile -t simudo-for-docs \
    --build-arg APT_HTTP_PROXY=http://192.168.122.1:3142 \
    --build-arg FOR_DOCS=1 \
    -v "$HOME/path/to/simudo/dist/:/wheels:ro" \
    --build-arg SIMUDO_SPEC=/wheels/simudo-0.7.1.0.tar.gz
```

You can omit the `--build-arg APT_HTTP_PROXY=...` if you don't have apt-cacher-ng set
up. You can omit the last two lines if you are not building from a specific sdist
tarball.

To extract the docs from such an image, run:

```sh
podman run --rm localhost/simudo-for-docs:latest tar -C /opt/simudo/doc/build/html -c . | zstd -5v > simudo-docs.tar.zst
```

To build a local sdist tarball from your current simudo source tree, you can run:

```sh
unshare -nc python3 -m build -x -n -s
```

(The `unshare -nc` part is optional, it's only supported on Linux and it's there to ensure that no network access is available to the build process.)
