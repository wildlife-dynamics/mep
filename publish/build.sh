#!/bin/bash

RECIPES=(
    "release/ecoscope-workflows-ext-mep"
)
project_root=$(pwd)/src/ecoscope-workflows-ext-mep
export HATCH_VCS_VERSION=$(cd $project_root && hatch version)
echo "HATCH_VCS_VERSION=$HATCH_VCS_VERSION"


echo "Building recipes: ${RECIPES[@]}"

pixi clean cache --yes

# Setup output directory
OUTPUT_DIR="/tmp/ecoscope-workflows-custom/release/artifacts"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory exists, cleaning up old files..."
    # Delete all files starting with ecoscope-workflows-ext-mep or ecoscope-workflows-ext-custom
    rm -f "$OUTPUT_DIR"/noarch/ecoscope-workflows-ext-mep*
    rm -f "$OUTPUT_DIR"/noarch/ecoscope-workflows-ext-custom*
    # Delete repodata.json if it exists
    rm -f "$OUTPUT_DIR/noarch/repodata.json"
else
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

for rec in "${RECIPES[@]}"; do
    echo "Building $rec"
    rattler-build build \
    --recipe $(pwd)/publish/recipes/${rec}.yaml \
    --output-dir /tmp/ecoscope-workflows-custom/release/artifacts \
    --channel https://prefix.dev/ecoscope-workflows \
    --channel conda-forge
done
