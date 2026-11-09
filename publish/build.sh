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

# Playwright's browser, so ext-mep's video export never downloads one at run time. Versioned with the
# Playwright driver, not with ext-mep, so it is only built when that version isn't on the custom channel
# yet (--skip-existing all). Workflows lock for every platform below, so each needs a package; all are
# cross-built here (the recipe repackages Google's prebuilt Chrome for Testing zips). Tests run for the
# native target only: the Windows test is a cmd script that can't run on the Linux builder.
CHROMIUM_TARGETS=(linux-64 linux-aarch64 osx-arm64 osx-64 win-64)
for target in "${CHROMIUM_TARGETS[@]}"; do
    echo "Building playwright-chromium-headless-shell for $target"
    rattler-build build \
    --recipe $(pwd)/publish/recipes/release/playwright-chromium-headless-shell/recipe.yaml \
    --target-platform "$target" \
    --skip-existing all \
    --test native \
    --output-dir /tmp/ecoscope-workflows-custom/release/artifacts \
    --channel https://prefix.dev/ecoscope-workflows-custom \
    --channel conda-forge
done
