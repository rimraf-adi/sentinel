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

# 2. Component management
echo "Step 2: Listing components..."
"$SENTINEL_BIN" component list

echo "Step 3: Creating custom component..."
"$SENTINEL_BIN" component new "Transfer Learning" --domain vision --extends "vision/image-classification@v1"
test -f components/transfer-learning.yaml || { echo "❌ Failed to create custom component"; exit 1; }

echo "Step 4: Validating custom component..."
"$SENTINEL_BIN" component check components/transfer-learning.yaml
echo "✅ Custom component verified."

# 3. Export standalone script
echo "Step 5: Exporting standalone pure bare PyTorch script..."
"$SENTINEL_BIN" export script --component vision/image-classification@v1 --output run_standalone.py
test -f run_standalone.py || { echo "❌ Failed to export standalone script"; exit 1; }
echo "✅ Standalone script exported."

# 4. Run standalone PyTorch training under uv
echo "Step 6: Executing exported script with uv run..."
cd "$WORKSPACE_DIR"
uv run python "$TEST_PROJ/run_standalone.py"
echo "✅ Standalone training executed and decreased loss."

# 5. Graph components: inspect, compose, train in the foreground
echo "Step 7: Graph components (show / compose / run)..."
cd "$TEST_PROJ"
mkdir -p data
python3 - <<'PY'
import math, random
random.seed(0)
with open("data/demo.csv", "w") as f:
    f.write("x1,x2,label\n")
    for k in range(3):
        for i in range(120):
            r = i / 120
            t = k * 4 + 4 * r + random.gauss(0, 0.2)
            f.write(f"{r * math.sin(t):.4f},{r * math.cos(t):.4f},{k}\n")
PY
"$SENTINEL_BIN" component blocks | grep -q "lstm" || { echo "❌ block catalog missing lstm"; exit 1; }
"$SENTINEL_BIN" component show tabular/ann-classifier@v1 | grep -q "head.out" || { echo "❌ show printed no shapes"; exit 1; }
"$SENTINEL_BIN" component compose sequence/lstm-classifier@v1 -o lstm.py --set hidden=16
grep -q "nn.LSTM" lstm.py || { echo "❌ compose did not emit an LSTM"; exit 1; }
if "$SENTINEL_BIN" run tabular/ann-classifier@v1 --set nope=1 >/dev/null 2>&1; then echo "❌ unknown hyperparameter was accepted"; exit 1; fi

if python3 -c "import torch" 2>/dev/null; then
  OUT=$("$SENTINEL_BIN" run tabular/ann-classifier@v1 --set epochs=120)
  echo "$OUT" | tail -2
  ACC=$(echo "$OUT" | grep test_acc | tail -1 | awk '{print $NF}')
  python3 -c "import sys; sys.exit(0 if float('$ACC') > 0.6 else 1)" || { echo "❌ ANN did not learn (test_acc=$ACC)"; exit 1; }
  test -n "$(ls mlruns/0 2>/dev/null)" || { echo "❌ run was not tracked in mlruns/"; exit 1; }
  echo "✅ Graph component trained and tracked (test_acc=$ACC)."
else
  echo "⚠️  torch not available for python3; skipping the training steps"
fi

# 6. Daemon: embedded UI, API, persistence across a restart
echo "Step 8: Daemon smoke test..."
PORT=2597
"$SENTINEL_BIN" dev --port $PORT >"$TEST_PROJ/daemon.log" 2>&1 &
DAEMON=$!
trap 'kill $DAEMON 2>/dev/null || true' EXIT
for i in $(seq 1 40); do curl -sf "localhost:$PORT/api/health" >/dev/null && break; sleep 0.25; done
curl -sf "localhost:$PORT/api/health" >/dev/null || { echo "❌ daemon did not start"; cat "$TEST_PROJ/daemon.log"; exit 1; }
curl -sf "localhost:$PORT/" | grep -q 'id="root"' || { echo "❌ embedded React UI not served"; exit 1; }
test "$(curl -s "localhost:$PORT/api/blocks" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)))')" -ge 20 || { echo "❌ /api/blocks too small"; exit 1; }
curl -s -X POST "localhost:$PORT/api/graph/validate" -H 'content-type: application/json' \
  -d '{"component_id":"tabular/ann-classifier@v1"}' | grep -q '"ok":true' || { echo "❌ validate failed"; exit 1; }
curl -s -X POST "localhost:$PORT/api/graph/validate" -H 'content-type: application/json' \
  -d '{"component_id":"tabular/ann-classifier@v1","overrides":{"hidden":1}}' | grep -q '"ok":false' || { echo "❌ out-of-range override accepted"; exit 1; }
test "$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/api/tracking/runs/..%2F..%2Fetc")" = "404" || { echo "❌ path traversal not rejected"; exit 1; }

if python3 -c "import torch" 2>/dev/null; then
  JOB=$(curl -s -X POST "localhost:$PORT/api/graph/run" -H 'content-type: application/json' \
    -d '{"component_id":"tabular/ann-classifier@v1","overrides":{"epochs":30}}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["spec"]["id"])')
  for i in $(seq 1 60); do
    ST=$(curl -s "localhost:$PORT/api/jobs/$JOB" | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')
    [ "$ST" = "completed" ] || [ "$ST" = "failed" ] && break; sleep 0.5
  done
  [ "$ST" = "completed" ] || { echo "❌ daemon job ended as '$ST'"; exit 1; }
  kill $DAEMON; wait $DAEMON 2>/dev/null || true
  "$SENTINEL_BIN" dev --port $PORT >>"$TEST_PROJ/daemon.log" 2>&1 &
  DAEMON=$!
  for i in $(seq 1 40); do curl -sf "localhost:$PORT/api/health" >/dev/null && break; sleep 0.25; done
  curl -s "localhost:$PORT/api/jobs/$JOB" | grep -q '"status":"completed"' || { echo "❌ job history lost after restart"; exit 1; }
  echo "✅ Daemon served the UI, ran a job, and kept its history across a restart."
fi
kill $DAEMON 2>/dev/null || true

# 7. Clean up
cd "$WORKSPACE_DIR"
rm -rf "$TEST_PROJ"

echo "🎉 [E2E] All lifecycle integration steps PASSED cleanly!"
