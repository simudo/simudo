#!/bin/bash
set -e

mkdir -p data/sentaurus/
pushd data/sentaurus/

o="tmp.zip"
for s in study{2,3}-{spatial,iv}; do
    mkdir -p "$s"
    pushd "$s"
    rm -f "$o"
    fossil uv export sentaurus-"$s".zip "$o"
    unzip -o "$o"
    rm "$o"
    popd
done

popd


