#!/bin/bash
set -e

WORKSPACE_DIR="$(pwd)"
SENTINEL_BIN="$WORKSPACE_DIR/target/debug/sentinel"
TEST_PROJ="/tmp/sentinel_lifecycle_e2e"

echo "🧪 [E2E] Starting Full Sentinel Lifecycle Integration Test..."

# Clean old run if exists
rm -rf "$TEST_PROJ"

# 1. Scaffolding: Create new project
echo "Step 1: Scaffolding new project 'lifecycle-demo'..."
cd /tmp
"$SENTINEL_BIN" new "$TEST_PROJ"

cd "$TEST_PROJ"
test -f sentinel.yaml || { echo "❌ Missing sentinel.yaml"; exit 1; }
test -f sentinel.lock || { echo "❌ Missing sentinel.lock"; exit 1; }
test -f pyproject.toml || { echo "❌ Missing pyproject.toml"; exit 1; }
echo "✅ Project files successfully created."

# 2. Template management
echo "Step 2: Listing templates..."
"$SENTINEL_BIN" template list

echo "Step 3: Creating custom template..."
"$SENTINEL_BIN" template new "Transfer Learning" --domain vision --extends "vision/image-classification@v1"
test -f templates/transfer-learning.yaml || { echo "❌ Failed to create custom template"; exit 1; }

echo "Step 4: Validating custom template..."
"$SENTINEL_BIN" template check templates/transfer-learning.yaml
echo "✅ Custom template verified."

# 3. Export standalone script
echo "Step 5: Exporting standalone pure bare PyTorch script..."
"$SENTINEL_BIN" export script --template vision/image-classification@v1 --output run_standalone.py
test -f run_standalone.py || { echo "❌ Failed to export standalone script"; exit 1; }
echo "✅ Standalone script exported."

# 4. Run standalone PyTorch training under uv
echo "Step 6: Executing exported script with uv run..."
cd "$WORKSPACE_DIR"
uv run python "$TEST_PROJ/run_standalone.py"
echo "✅ Standalone training executed and decreased loss."

# 5. Clean up
rm -rf "$TEST_PROJ"

echo "🎉 [E2E] All lifecycle integration steps PASSED cleanly!"
