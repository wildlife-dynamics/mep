# playwright-chromium-headless-shell: point Playwright at the browser bundled in this env,
# unless the user already chose a browsers path.
if [ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" ]; then
    export PLAYWRIGHT_BROWSERS_PATH="${CONDA_PREFIX}/share/ms-playwright"
    export _PW_CHS_SET_BROWSERS_PATH=1
fi
# Linux: chrome-headless-shell has fontconfig statically linked with /etc/fonts as its
# config dir; point it at the env's fonts.conf (fontconfig + fonts-conda-ecosystem) so
# text renders in bare containers without system fonts.
if [ "$(uname -s 2>/dev/null)" = "Linux" ] && [ -z "${FONTCONFIG_FILE:-}" ] \
    && [ -f "${CONDA_PREFIX}/etc/fonts/fonts.conf" ]; then
    export FONTCONFIG_FILE="${CONDA_PREFIX}/etc/fonts/fonts.conf"
    export _PW_CHS_SET_FONTCONFIG_FILE=1
fi
