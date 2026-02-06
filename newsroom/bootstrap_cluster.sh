#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$BASE_DIR/config.json}"
CIDR="${2:-}"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[bootstrap] Config: $CONFIG"
if [[ -n "$CIDR" ]]; then
  echo "[bootstrap] CIDR: $CIDR"
fi

AUTO_ARGS=(--config "$CONFIG")
if [[ -n "$CIDR" ]]; then
  AUTO_ARGS+=(--cidr "$CIDR")
fi

python3 "$BASE_DIR/auto_configure_cluster.py" "${AUTO_ARGS[@]}"
bash "$BASE_DIR/check_lm_cluster.sh" "$CONFIG"

# Reinicio limpio del pipeline para aplicar rutas nuevas inmediatamente.
pkill -f "$BASE_DIR/runner.py --config $CONFIG" >/dev/null 2>&1 || true
pkill -f "$BASE_DIR/ai_publisher.py --config $CONFIG" >/dev/null 2>&1 || true
sleep 1

if ! pgrep -f "$BASE_DIR/daemon.sh" >/dev/null 2>&1; then
  nohup /bin/bash "$BASE_DIR/daemon.sh" >> "$LOG_DIR/daemon.bootstrap.out.log" 2>> "$LOG_DIR/daemon.bootstrap.err.log" &
  echo "[bootstrap] daemon iniciado."
else
  echo "[bootstrap] daemon ya estaba activo."
fi

# Activa/renueva sync horario a GitHub.
bash "$BASE_DIR/setup_github_sync_launchd.sh" >/dev/null

echo "[bootstrap] Listo."
echo "[bootstrap] Revisa: $LOG_DIR/publisher.log"
