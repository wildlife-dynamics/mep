@REM playwright-chromium-headless-shell: point Playwright at the browser bundled in this env,
@REM unless the user already chose a browsers path.
@if not defined PLAYWRIGHT_BROWSERS_PATH (
    @set "PLAYWRIGHT_BROWSERS_PATH=%CONDA_PREFIX%\share\ms-playwright"
    @set "_PW_CHS_SET_BROWSERS_PATH=1"
)
