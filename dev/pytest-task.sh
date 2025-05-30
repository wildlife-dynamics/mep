#!/bin/bash

python_version=$1

manifest_path=src/ecoscope-workflows-ext-mep/pyproject.toml

pixi update --manifest-path $manifest_path

command="pixi run \
--manifest-path ${manifest_path} \
--environment test-py${python_version} \
pytest src/ecoscope-workflows-ext-mep/tests -vv"

shift 1
if [ -n "$*" ]; then
    extra_args=$*
    command="$command $extra_args"
fi

echo "Running command: $command"
eval $command
