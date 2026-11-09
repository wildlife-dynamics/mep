#!/bin/bash

for file in /tmp/ecoscope-workflows-custom/release/artifacts/**/*.conda; do
    # --skip-existing: packages versioned independently of ext-mep (the Chromium build) may already be there.
    rattler-build upload prefix --skip-existing -c ecoscope-workflows-custom "$file"
done
