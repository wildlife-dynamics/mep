@echo on
setlocal

:: Zip layout: chrome-headless-shell-win64\{chrome-headless-shell.exe,...}
:: tar.exe (bsdtar) ships with Windows 10+ and extracts zip files.
tar -xf "%SRC_DIR%\chrome-headless-shell.zip" -C "%SRC_DIR%"
if errorlevel 1 exit 1

set "DEST=%PREFIX%\share\ms-playwright\chromium_headless_shell-%PW_REVISION%"
if not exist "%DEST%" mkdir "%DEST%"
if errorlevel 1 exit 1
xcopy /E /I /Q /Y "%SRC_DIR%\chrome-headless-shell-%PW_CFT_PLATFORM%" "%DEST%\chrome-headless-shell-%PW_CFT_PLATFORM%"
if errorlevel 1 exit 1

:: Marker Playwright writes after a successful download.
type nul > "%DEST%\INSTALLATION_COMPLETE"

for %%D in (activate deactivate) do (
    if not exist "%PREFIX%\etc\conda\%%D.d" mkdir "%PREFIX%\etc\conda\%%D.d"
    copy /Y "%RECIPE_DIR%\%%D.bat" "%PREFIX%\etc\conda\%%D.d\%PKG_NAME%_%%D.bat"
    if errorlevel 1 exit 1
    copy /Y "%RECIPE_DIR%\%%D.sh" "%PREFIX%\etc\conda\%%D.d\%PKG_NAME%_%%D.sh"
    if errorlevel 1 exit 1
)
