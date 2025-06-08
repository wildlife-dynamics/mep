#!/bin/bash

name=$1
results_dir=.ecoscope-workflows-tmp/${name}-workflow-results

docker run --name ${name} -d -v $(pwd)/${results_dir}:/workflow/results -e PORT=4000 -p 4000:4000 ecoscope-workflows-${name}-workflow
