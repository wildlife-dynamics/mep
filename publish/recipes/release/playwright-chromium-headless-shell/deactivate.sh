# playwright-chromium-headless-shell: undo activate.sh (only what it set).
if [ -n "${_PW_CHS_SET_BROWSERS_PATH:-}" ]; then
    unset PLAYWRIGHT_BROWSERS_PATH
    unset _PW_CHS_SET_BROWSERS_PATH
fi
if [ -n "${_PW_CHS_SET_FONTCONFIG_FILE:-}" ]; then
    unset FONTCONFIG_FILE
    unset _PW_CHS_SET_FONTCONFIG_FILE
fi
