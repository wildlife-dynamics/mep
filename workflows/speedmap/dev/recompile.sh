#!/bin/bash

flags=$*

# (re)initialize dot executable to ensure graphviz is available
pixi run --manifest-path pixi.toml --locked -e default dot -c

echo "recompiling spec.yaml with flags '--clobber ${flags}'"

command="pixi run --manifest-path pixi.toml --locked -e default \
ecoscope-workflows compile --spec spec.yaml --clobber ${flags}"

exec $command
