#!/bin/bash

cd "$(dirname "$BASH_SOURCE")"

rsync ./ --include '*.py' \
      --include '/*.sh' \
      --exclude '/out/**' --exclude '*.pyc' --exclude '__pycache__' \
      --exclude /doc/build \
      --include /simudo/'**' --include /doc/'**' --include /pint/'**' \
      --include '*/' --exclude '*' -avr pl:j/
