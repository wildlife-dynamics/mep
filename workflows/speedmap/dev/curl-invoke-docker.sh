#!/bin/bash

set -e

execution_mode=$1
mock_io=true
port=4000
results_url=/workflow/results  # must be consistent with volume mount set in docker-run.sh
params=$(yq -c '.test1.params' test-cases.yaml)

curl --fail-with-body \
-X POST "http://localhost:${port}/?execution_mode=${execution_mode}&mock_io=${mock_io}&results_url=${results_url}" \
-H "Content-Type: application/json" \
-d '{"params": '"${params}"'}'
