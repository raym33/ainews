#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="/Users/c/Library/LaAurora/web"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

REPO_DIR="${GITHUB_REPO_DIR:-$WEB_DIR}"
BRANCH="${GITHUB_BRANCH:-main}"
REMOTE_URL="${GITHUB_REMOTE_URL:-}"
COMMIT_PREFIX="${GITHUB_COMMIT_PREFIX:-chore(news): snapshot}"

if [[ -f "$BASE_DIR/github-sync.env" ]]; then
  # shellcheck disable=SC1090
  source "$BASE_DIR/github-sync.env"
  REPO_DIR="${GITHUB_REPO_DIR:-$REPO_DIR}"
  BRANCH="${GITHUB_BRANCH:-$BRANCH}"
  REMOTE_URL="${GITHUB_REMOTE_URL:-$REMOTE_URL}"
  COMMIT_PREFIX="${GITHUB_COMMIT_PREFIX:-$COMMIT_PREFIX}"
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

privacy_guard() {
  if [[ "${ALLOW_PRIVATE_STRINGS:-0}" == "1" ]]; then
    return 0
  fi
  local tmp
  tmp="$(mktemp)"
  local failed=0

  # Detecta URLs locales/privadas o secretos comunes en los archivos versionados.
  if git grep -nE "(https?://(10\\.|192\\.168\\.|172\\.(1[6-9]|2[0-9]|3[0-1])\\.)|:1234/v1|ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})" -- . >"$tmp" 2>/dev/null; then
    if [[ -s "$tmp" ]]; then
      log "ERROR: bloqueado por guardia de privacidad. Revisa coincidencias:"
      cat "$tmp"
      failed=1
    fi
  fi

  rm -f "$tmp"
  if [[ "$failed" -ne 0 ]]; then
    log "Sugerencia: elimina esos datos o exporta ALLOW_PRIVATE_STRINGS=1 bajo tu responsabilidad."
    exit 1
  fi
}

if [[ ! -d "$REPO_DIR" ]]; then
  log "ERROR: no existe GITHUB_REPO_DIR=$REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"

if [[ ! -d ".git" ]]; then
  log "Inicializando repo git en $REPO_DIR"
  git init -b "$BRANCH"
fi

if [[ -n "$REMOTE_URL" ]]; then
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE_URL"
  else
    git remote add origin "$REMOTE_URL"
  fi
fi

if [[ ! -f ".gitignore" ]]; then
  cat > .gitignore <<'EOF'
.DS_Store
Thumbs.db
EOF
fi

git add -A

privacy_guard

if git diff --cached --quiet; then
  log "Sin cambios para subir."
  exit 0
fi

LATEST_TITLE="$(jq -r '.breaking // empty' data/articles.json 2>/dev/null || true)"
STAMP="$(date '+%Y-%m-%d %H:%M:%S')"
MSG="$COMMIT_PREFIX $STAMP"
if [[ -n "$LATEST_TITLE" ]]; then
  MSG="$MSG | $LATEST_TITLE"
fi

git commit -m "$MSG" >/dev/null
log "Commit creado."

if ! git remote get-url origin >/dev/null 2>&1; then
  log "WARN: no hay remote origin. Define GITHUB_REMOTE_URL para activar push."
  exit 0
fi

git fetch origin "$BRANCH" --depth=1 >/dev/null 2>&1 || true
if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  if ! git pull --rebase origin "$BRANCH"; then
    log "WARN: fallo rebase; se aborta rebase y se mantiene commit local."
    git rebase --abort >/dev/null 2>&1 || true
  fi
fi

git push -u origin "$BRANCH"
log "Push completado a origin/$BRANCH."
