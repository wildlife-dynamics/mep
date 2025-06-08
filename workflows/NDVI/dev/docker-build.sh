#!/bin/bash

name=$1

command="pixi run \
--manifest-path ecoscope-workflows-${name}-workflow/pixi.toml \
--locked --environment default \
docker-build"

eval $command
