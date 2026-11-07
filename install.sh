#!/usr/bin/env bash
# Sentinel Installer
# Single command installer for Sentinel ML Training Supervisor & Operations Plane
# Usage: curl -fsSL https://raw.githubusercontent.com/rimraf-adi/sentinel/main/install.sh | bash

set -euo pipefail

BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
echo "  ___ ___ _  _ _____ ___ _  _ ___ _     "
echo " / __| __| \| |_   _|_ _| \| | __| |    "
echo " \__ \ _|| .  | | |  | || .  | _|| |__  "
echo " |___/___|_|\_| |_| |___|_|\_|___|____| "
echo -e "${RESET}"
echo -e "${BOLD}Sentinel ML Training Supervisor & Operations Plane${RESET}"
echo "Installing the single-binary ML training plane..."
echo ""

OS="$(uname -s)"
ARCH="$(uname -m)"

# Determine installation directory
BIN_DIR="${HOME}/.cargo/bin"
if [ ! -d "${BIN_DIR}" ]; then
  BIN_DIR="${HOME}/.local/bin"
  mkdir -p "${BIN_DIR}"
fi

# Check for Rust / Cargo
if command -v cargo >/dev/null 2>&1; then
  echo -e "${CYAN}→${RESET} Found Cargo. Installing latest Sentinel binary from repository..."
  cargo install --git https://github.com/rimraf-adi/sentinel.git sentinel-cli --locked --force
elif command -v curl >/dev/null 2>&1; then
  echo -e "${YELLOW}→${RESET} Cargo not found. Fetching pre-compiled release for ${OS} (${ARCH})..."
  # If GitHub release asset is available, download to BIN_DIR
  TEMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TEMP_DIR}"' EXIT
  
  # Fallback to installing Rust or building via cargo
  echo -e "${YELLOW}Notice:${RESET} Rust toolchain is recommended for local native optimizations."
  if command -v rustup >/dev/null 2>&1; then
    rustup default stable
    cargo install --git https://github.com/rimraf-adi/sentinel.git sentinel-cli --locked --force
  else
    echo -e "${RED}Error:${RESET} Cargo is required to build Sentinel for your machine architecture (${ARCH})."
    echo "Install Rust via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
  fi
else
  echo -e "${RED}Error:${RESET} Neither cargo nor curl available."
  exit 1
fi

# Verify executable
SENTINEL_BIN="${BIN_DIR}/sentinel"
if [ ! -f "${SENTINEL_BIN}" ] && command -v sentinel >/dev/null 2>&1; then
  SENTINEL_BIN="$(command -v sentinel)"
fi

if [ -f "${SENTINEL_BIN}" ]; then
  # On macOS, ensure ad-hoc code signature is valid
  if [ "${OS}" = "Darwin" ] && command -v codesign >/dev/null 2>&1; then
    codesign -s - -f "${SENTINEL_BIN}" >/dev/null 2>&1 || true
  fi
  chmod +x "${SENTINEL_BIN}"
fi

echo ""
echo -e "${GREEN}${BOLD}✓ Sentinel successfully installed!${RESET}"
if command -v sentinel >/dev/null 2>&1; then
  echo -e "  Version: ${BOLD}$(sentinel --version)${RESET}"
else
  echo -e "  Location: ${BOLD}${SENTINEL_BIN}${RESET}"
  echo -e "  ${YELLOW}Note:${RESET} Ensure ${BIN_DIR} is in your PATH:"
  echo -e "        export PATH=\"${BIN_DIR}:\$PATH\""
fi

echo ""
echo -e "${BOLD}Quickstart Workflow:${RESET}"
echo "  1. Initialize repository:    sentinel init"
echo "  2. Verify environment:       sentinel doctor"
echo "  3. Start control dashboard:  sentinel dev          # http://localhost:2512"
echo "  4. Install model template:   sentinel add vit-tiny"
echo "  5. Launch training run:      sentinel run vision/vit-tiny@v1"
echo ""
echo -e "Hub & Community: ${CYAN}https://github.com/rimraf-adi/sentinel-hub${RESET}"
