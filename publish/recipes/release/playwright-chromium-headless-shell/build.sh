#!/usr/bin/env bash
set -euxo pipefail

# Zip layout: chrome-headless-shell-<cft_platform>/{chrome-headless-shell,...}
unzip -q -o "${SRC_DIR}/chrome-headless-shell.zip" -d "${SRC_DIR}"

# Same directory name Playwright's registry uses: <browser name with _>-<revision>
dest="${PREFIX}/${PW_INSTALL_DIR}/chromium_headless_shell-${PW_REVISION}"
mkdir -p "${dest}"
cp -R "${SRC_DIR}/chrome-headless-shell-${PW_CFT_PLATFORM}" "${dest}/"
chmod +x "${dest}/chrome-headless-shell-${PW_CFT_PLATFORM}/chrome-headless-shell"

# Marker Playwright writes after a successful download; with it present
# `playwright install chromium-headless-shell` treats the browser as installed.
touch "${dest}/INSTALLATION_COMPLETE"

for d in activate deactivate; do
    mkdir -p "${PREFIX}/etc/conda/${d}.d"
    cp "${RECIPE_DIR}/${d}.sh" "${PREFIX}/etc/conda/${d}.d/${PKG_NAME}_${d}.sh"
    cp "${RECIPE_DIR}/${d}.bat" "${PREFIX}/etc/conda/${d}.d/${PKG_NAME}_${d}.bat"
done
