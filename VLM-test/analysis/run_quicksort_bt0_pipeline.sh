#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

QS_ROOT="${QS_ROOT:-}"
GT_SCENES="${GT_SCENES:-}"
OUT_ROOT="${OUT_ROOT:-}"

WORKBOOK="${WORKBOOK:-}"
CONCURRENCY="${CONCURRENCY:-8}"
RESTARTS="${RESTARTS:-10}"
BT_RATIO_ALPHA="${BT_RATIO_ALPHA:-0}"
RUNS_FILE="${RUNS_FILE:-}"
MAX_SCENES="${MAX_SCENES:-}"
SKIP_DONE="${SKIP_DONE:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_PIVOT="${SKIP_PIVOT:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  QS_ROOT=/path/to/source/results \
  GT_SCENES=/path/to/gt/scenes \
  OUT_ROOT=/path/to/recon/output \
  bash VLM-test/analysis/run_quicksort_bt0_pipeline.sh

Required env vars:
  QS_ROOT       quick-sort source results root; each run must contain summary.json + scenes/
  GT_SCENES     GT scene json directory for reconstruction evaluation
  OUT_ROOT      reconstruction output root

Optional env vars:
  WORKBOOK      pivot workbook path; default: $OUT_ROOT/pivots/quicksort_bt0_recon_pivot.xlsx
  CONCURRENCY   xargs worker count; default: 8
  RESTARTS      solver restarts; default: 10
  BT_RATIO_ALPHA solver bt_ratio_alpha override; default: 0
  RUNS_FILE     manifest file; one "source_attempt/canonical_run" per line
  MAX_SCENES    optional per-run scene cap for smoke tests
  SKIP_DONE     1 to skip runs whose recon/summary.json already exists; default: 1
  FORCE_RERUN   1 to ignore existing recon/summary.json; default: 0
  SKIP_PIVOT    1 to skip final pivot workbook generation; default: 0
  DRY_RUN       1 to print the resolved plan without executing it; default: 0
EOF
}

fail() {
  echo "[reconstruct-quicksort] $*" >&2
  exit 1
}

require_dir() {
  local path="$1"
  local name="$2"
  [[ -n "$path" ]] || fail "Missing required env var: $name"
  [[ -d "$path" ]] || fail "$name does not exist or is not a directory: $path"
}

normalize_bool() {
  case "$1" in
    1|true|TRUE|yes|YES) echo "1" ;;
    0|false|FALSE|no|NO) echo "0" ;;
    *) fail "Expected boolean-like value for flag, got: $1" ;;
  esac
}

require_int() {
  local value="$1"
  local name="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "$name must be a non-negative integer, got: $value"
}

require_float() {
  local value="$1"
  local name="$2"
  [[ "$value" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || fail "$name must be numeric, got: $value"
}

discover_runs() {
  find "$QS_ROOT" -type f -name 'summary.json' -print \
    | while IFS= read -r summary; do
        local run_dir
        run_dir="$(dirname "$summary")"
        if [[ -d "$run_dir/scenes" ]]; then
          python3 - "$QS_ROOT" "$run_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
run_dir = Path(sys.argv[2]).resolve()
print(run_dir.relative_to(root).as_posix())
PY
        fi
      done \
    | LC_ALL=C sort -u
}

load_runs() {
  local output_file="$1"

  if [[ -n "$RUNS_FILE" ]]; then
    [[ -f "$RUNS_FILE" ]] || fail "RUNS_FILE does not exist: $RUNS_FILE"
    sed \
      -e 's/[[:space:]]*#.*$//' \
      -e 's/^[[:space:]]*//' \
      -e 's/[[:space:]]*$//' \
      -e '/^[[:space:]]*$/d' \
      "$RUNS_FILE" >"$output_file"
  else
    discover_runs >"$output_file"
  fi

  [[ -s "$output_file" ]] || fail "No runs selected. Check QS_ROOT or RUNS_FILE."
}

run_output_dir() {
  local run="$1"
  local tag="${run%%/*}"
  local canon="${run##*/}"
  printf '%s/%s__%s' "$OUT_ROOT" "$tag" "$canon"
}

run_one() {
  local run="$1"
  local source_dir="$QS_ROOT/$run"
  local output_dir
  output_dir="$(run_output_dir "$run")"

  [[ -d "$source_dir" ]] || fail "Source run directory not found: $source_dir"
  [[ -f "$source_dir/summary.json" ]] || fail "Missing summary.json in source run: $source_dir"
  [[ -d "$source_dir/scenes" ]] || fail "Missing scenes/ in source run: $source_dir"

  if [[ "$FORCE_RERUN" != "1" && "$SKIP_DONE" == "1" && -f "$output_dir/${run##*/}/recon/summary.json" ]]; then
    echo "[reconstruct-quicksort] skip completed run: $run"
    return 0
  fi

  echo "[reconstruct-quicksort] processing run: $run"

  local -a cmd=(
    uv run python VLM-test/analysis/reconstruct_quicksort_orders.py
    --results-dir "$source_dir"
    --scenes-dir "$GT_SCENES"
    --output-dir "$output_dir"
    --bt-ratio-alpha "$BT_RATIO_ALPHA"
    --restarts "$RESTARTS"
  )

  if [[ -n "$MAX_SCENES" ]]; then
    cmd+=(--max-scenes "$MAX_SCENES")
  fi

  "${cmd[@]}"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  require_dir "$QS_ROOT" "QS_ROOT"
  require_dir "$GT_SCENES" "GT_SCENES"
  [[ -n "$OUT_ROOT" ]] || fail "Missing required env var: OUT_ROOT"

  CONCURRENCY="${CONCURRENCY//[[:space:]]/}"
  RESTARTS="${RESTARTS//[[:space:]]/}"
  BT_RATIO_ALPHA="${BT_RATIO_ALPHA//[[:space:]]/}"
  MAX_SCENES="${MAX_SCENES//[[:space:]]/}"

  require_int "$CONCURRENCY" "CONCURRENCY"
  [[ "$CONCURRENCY" != "0" ]] || fail "CONCURRENCY must be >= 1"
  require_int "$RESTARTS" "RESTARTS"
  if [[ -n "$MAX_SCENES" ]]; then
    require_int "$MAX_SCENES" "MAX_SCENES"
    [[ "$MAX_SCENES" != "0" ]] || fail "MAX_SCENES must be >= 1 when set"
  fi
  require_float "$BT_RATIO_ALPHA" "BT_RATIO_ALPHA"

  SKIP_DONE="$(normalize_bool "$SKIP_DONE")"
  FORCE_RERUN="$(normalize_bool "$FORCE_RERUN")"
  SKIP_PIVOT="$(normalize_bool "$SKIP_PIVOT")"
  DRY_RUN="$(normalize_bool "$DRY_RUN")"

  if [[ "$FORCE_RERUN" == "1" ]]; then
    SKIP_DONE="0"
  fi

  if [[ -z "$WORKBOOK" ]]; then
    WORKBOOK="$OUT_ROOT/pivots/quicksort_bt0_recon_pivot.xlsx"
  fi

  local run_list
  run_list="$(mktemp)"
  trap 'rm -f "$run_list"' EXIT
  load_runs "$run_list"

  local run_count
  run_count="$(wc -l <"$run_list" | tr -d ' ')"

  echo "[reconstruct-quicksort] repo_root=$REPO_ROOT"
  echo "[reconstruct-quicksort] qs_root=$QS_ROOT"
  echo "[reconstruct-quicksort] gt_scenes=$GT_SCENES"
  echo "[reconstruct-quicksort] out_root=$OUT_ROOT"
  echo "[reconstruct-quicksort] workbook=$WORKBOOK"
  echo "[reconstruct-quicksort] concurrency=$CONCURRENCY restarts=$RESTARTS bt_ratio_alpha=$BT_RATIO_ALPHA"
  if [[ -n "$RUNS_FILE" ]]; then
    echo "[reconstruct-quicksort] runs_file=$RUNS_FILE"
  else
    echo "[reconstruct-quicksort] runs_file=<auto-discovered>"
  fi
  if [[ -n "$MAX_SCENES" ]]; then
    echo "[reconstruct-quicksort] max_scenes=$MAX_SCENES"
  fi
  echo "[reconstruct-quicksort] skip_done=$SKIP_DONE force_rerun=$FORCE_RERUN skip_pivot=$SKIP_PIVOT dry_run=$DRY_RUN"
  echo "[reconstruct-quicksort] selected_runs=$run_count"
  sed 's/^/  - /' "$run_list"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[reconstruct-quicksort] dry run only; exiting before reconstruction"
    exit 0
  fi

  mkdir -p "$OUT_ROOT/pivots"

  export QS_ROOT GT_SCENES OUT_ROOT RESTARTS BT_RATIO_ALPHA MAX_SCENES SKIP_DONE FORCE_RERUN
  export -f fail run_output_dir run_one

  cd "$REPO_ROOT"
  xargs -P "$CONCURRENCY" -I{} bash -lc 'run_one "$@"' _ {} <"$run_list"

  if [[ "$SKIP_PIVOT" == "1" ]]; then
    echo "[reconstruct-quicksort] skip pivot workbook generation"
    exit 0
  fi

  uv run --with openpyxl python VLM-test/analysis/pivot_quicksort_recon_results.py \
    --recon-root "$OUT_ROOT" \
    --source-results-root "$QS_ROOT" \
    --output-xlsx "$WORKBOOK"
}

main "$@"
