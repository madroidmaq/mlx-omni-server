#!/usr/bin/env bash
set -euo pipefail

# Builds an offline wheelhouse for macOS arm64 (Apple Silicon) for mlx-omni-server
# Usage:
#   bash scripts/build_macos_arm64_wheelhouse.sh            # uses default PKG_VERSION
#   PKG_VERSION=0.5.1 bash scripts/build_macos_arm64_wheelhouse.sh

PKG_VERSION=${PKG_VERSION:-0.5.1}
ART_DIR=${ART_DIR:-./artifacts}
WH_DIR="$ART_DIR/wheelhouse"

mkdir -p "$WH_DIR"

# Ensure python3.11 + pip available
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 is required in the environment (try: nix develop)" >&2
  exit 1
fi

python3.11 -m pip install --upgrade pip

# Download macOS arm64 wheels without installing
# We allow this to partially succeed to gather as many wheels as possible
python3.11 -m pip download \
  --platform macosx_12_0_arm64 \
  --only-binary=:all: \
  --implementation cp \
  --python-version 311 \
  --abi cp311 \
  "mlx-omni-server==${PKG_VERSION}" \
  -d "$WH_DIR" || true

# Write helper installer script for macOS host
cat > "$ART_DIR/install_mlx_omni_offline.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEEL_DIR="$SCRIPT_DIR/wheelhouse"
PY=python3.11
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3.11 required on macOS host" >&2
  exit 1
fi
"$PY" -m pip install --upgrade pip
"$PY" -m pip install --no-index --find-links "$WHEEL_DIR" mlx-omni-server==${PKG_VERSION}
echo "Installed. Run: mlx-omni-server --host 0.0.0.0 --port 10240"
EOF
chmod +x "$ART_DIR/install_mlx_omni_offline.sh"

# README for artifacts
cat > "$ART_DIR/README_ARTIFACTS.md" <<EOF
# MLX Omni Server macOS (arm64) Wheelhouse

This directory contains pre-downloaded macOS arm64 wheels for mlx-omni-server and its dependencies.
Use this on an Apple Silicon Mac (Python 3.11).

Offline install on macOS:

  bash install_mlx_omni_offline.sh
  mlx-omni-server --host 0.0.0.0 --port 10240

If any dependency was missing a macOS wheel during the offline step, install online on macOS:

  pip install mlx-omni-server==${PKG_VERSION}

EOF

# Package into a tarball for convenience
TARBALL="$ART_DIR/mlx-omni-macos-arm64-wheelhouse.tar.gz"
rm -f "$TARBALL"
# Create tarball that contains the contents of artifacts/
( cd "$(dirname "$ART_DIR")" && tar -czf "$(basename "$TARBALL")" "$(basename "$ART_DIR")" )

echo "Artifacts prepared in $ART_DIR"
