#!/bin/bash
set -e
set -o pipefail
set -x
shopt -s nullglob

sudo apt install build-essential quilt cmake devscripts python{,3}-{dev,mpi4py} -t stretch-backports
N=30

do_deb_build() {
    dpkg-buildpackage -us -uc -b -nc -j$j "${extra_opts[@]}"
}

make_stage() {
    for f in */; do
        pushd "$f"
        mk-build-deps
        sudo dpkg -i *-build-deps_*.deb || true # failing is expected
        popd
    done
    sudo apt install --yes -f
    for f in */; do
        pushd "$f"
        do_deb_build
        popd
    done
    rm -f */*-build-deps_*.deb
    sudo dpkg -i *.deb
}

get_git() {
    local target="$1"
    local url="$2"
    local revision="$3"
    local tnew="$1.new"
    for loop in 1 2; do
        {
            pushd "$target" || pushd . && false # maintain directory stack size
            git pull
            git reset --hard "$revision"
        } && popd || {
            popd
            rm -rf "$target"
        }
        if [ "$loop" = 1 ] && ! [ -e "$target" ]; then
            git clone "$url" "$target"
        fi
    done
}

extra_opts=()

# damn it superlu
ln -s /usr/lib/libblas/libblas.so /usr/lib/x86_64-linux-gnu/libblas.so || true
for x in libparmetis.so libmetis.so; do
    ln -s /usr/lib/$x /usr/lib/x86_64-linux-gnu/$x || true
done

pd() {
    mkdir -p "$1"
    pushd "$1"
}

######################################################################

pd s1
apt-get source scotch -t buster
j=1 # this stupid package can't be built in parallel, aaaaaaa
make_stage
j=$N
popd

pd s2
apt-get source superlu-dist -t buster
make_stage
popd

pd s3
apt-get source hypre -t buster
make_stage
popd

pd s4
apt-get source petsc -t buster
make_stage
popd

pd s5
apt-get source slepc -t buster
make_stage
popd

pd s6
apt-get source fiat dijitso instant ufl -t buster
make_stage
popd

pd s7
apt-get source ffc -t buster
make_stage
popd

pd s8
apt-get source petsc4py -t buster
extra_opts=( -d )
make_stage
extra_opts=()
popd

pd s9
apt-get source slepc4py -t buster
extra_opts=( -d )
make_stage
extra_opts=()
popd

pd s10
apt-get source dolfin -t buster
make_stage
popd

pd s11
apt-get source cgal tetgen -t buster
make_stage
popd

pd mshr
get_git mshr https://salsa.debian.org/science-team/fenics/mshr.git \
        d0845666b0d60868796a07a2d77ea5701b84be87
j=1
make_stage
j=$N
popd

