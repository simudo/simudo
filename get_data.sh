#!/bin/bash
set -e

mkdir -p data/sentaurus/
pushd data/sentaurus/

o="tmp.zip"
for s in study{2,3}-{spatial,iv}; do
    mkdir -p "$s"
    pushd "$s"
    rm -f "$o"
    uvf=sentaurus-"$s".zip
    fossil uv export "$uvf" "$o" || {
        wget -O "$o" https://hydra.ecd.space/eduard/microdes/uv/"$uvf"
    }
    unzip -o "$o"
    rm "$o"
    popd
done

popd


