@REM playwright-chromium-headless-shell: undo activate.bat (only if it set the variable).
@if defined _PW_CHS_SET_BROWSERS_PATH (
    @set "PLAYWRIGHT_BROWSERS_PATH="
    @set "_PW_CHS_SET_BROWSERS_PATH="
)
