#!/bin/bash
set -e

# Needs sphinx and myst-parser (the user guide is written in Markdown):
#   pip install sphinx sphinx-rtd-theme myst-parser

#rm -r doc/source/apidoc/
pushd simudo
python3 -m sphinx.ext.apidoc -o ../doc/source/apidoc/ . memoize misc problemdata solution test trash
popd

make -C doc html

# webfsd -p 8777 -r doc/build/html/
