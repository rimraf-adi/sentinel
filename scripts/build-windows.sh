#!/usr/bin/env bash
set -euo pipefail

# Cross-compile sentinel.exe (x86_64 Windows) from macOS or Linux.
# Supports MSVC via `cargo-xwin` or GNU via `mingw-w64` (`x86_64-w64-mingw32-gcc`).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="release"
TOOLCHAIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      PROFILE="debug"
      shift
      ;;
    --release)
      PROFILE="release"
      shift
      ;;
    --msvc)
      TOOLCHAIN="msvc"
      shift
      ;;
    --gnu)
      TOOLCHAIN="gnu"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/build-windows.sh [--release|--debug] [--msvc|--gnu]

Options:
  --release   Build optimized release binary (default)
  --debug     Build debug binary
  --msvc      Force x86_64-pc-windows-msvc via cargo-xwin
  --gnu       Force x86_64-pc-windows-gnu via mingw-w64
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$TOOLCHAIN" ]]; then
  if command -v cargo-xwin >/dev/null 2>&1; then
    TOOLCHAIN="msvc"
  elif command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
    TOOLCHAIN="gnu"
  else
    echo "Error: no Windows cross-linker found on PATH." >&2
    echo "Install one of:" >&2
    echo "  brew install mingw-w64          # for x86_64-pc-windows-gnu (recommended on macOS)" >&2
    echo "  cargo install cargo-xwin        # for x86_64-pc-windows-msvc" >&2
    exit 1
  fi
fi

# Build embedded web UI unless explicitly skipped
if [[ "${SENTINEL_SKIP_UI:-0}" != "1" ]]; then
  if [[ -f "ui/package.json" ]]; then
    if ! command -v npm >/dev/null 2>&1; then
      echo "Error: npm not found (required to bundle UI assets; pass SENTINEL_SKIP_UI=1 to skip)." >&2
      exit 1
    fi
    echo "Building embedded web UI..."
    (cd ui && ([[ -d node_modules ]] || npm ci) && npm run build)
  fi
fi

PROFILE_FLAG=()
if [[ "$PROFILE" == "release" ]]; then
  PROFILE_FLAG=(--release)
fi

if [[ "$TOOLCHAIN" == "msvc" ]]; then
  TARGET="x86_64-pc-windows-msvc"
  rustup target add "$TARGET" >/dev/null 2>&1 || true
  echo "Building Sentinel for $TARGET ($PROFILE) via cargo-xwin..."
  cargo xwin build "${PROFILE_FLAG[@]}" --target "$TARGET" -p sentinel-cli
else
  TARGET="x86_64-pc-windows-gnu"
  rustup target add "$TARGET" >/dev/null 2>&1 || true
  echo "Building Sentinel for $TARGET ($PROFILE) via mingw-w64..."
  cargo build "${PROFILE_FLAG[@]}" --target "$TARGET" -p sentinel-cli
fi

OUT_BIN="target/$TARGET/$PROFILE/sentinel.exe"
if [[ ! -f "$OUT_BIN" ]]; then
  echo "Error: expected binary not found at $OUT_BIN" >&2
  exit 1
fi

echo ""
echo "Built: $OUT_BIN ($(du -h "$OUT_BIN" | awk '{print $1}'))"
