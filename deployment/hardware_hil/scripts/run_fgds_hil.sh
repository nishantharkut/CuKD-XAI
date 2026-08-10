#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 PORT GENERATED_DIR BUNDLE_DIR RESULT_DIR HOST_ENVIRONMENT_JSON" >&2
  exit 2
fi

PORT=$1
GENERATED_DIR=$2
BUNDLE_DIR=$3
FINAL_RESULT_DIR=$4
HOST_ENVIRONMENT_JSON=$5
RESULT_DIR="${FINAL_RESULT_DIR}.in_progress"
REFERENCE_CSV="${GENERATED_DIR}/hil_reference_predictions.csv"
VECTORS_CSV="${GENERATED_DIR}/hil_replay_vectors.csv"

if [[ -e "${FINAL_RESULT_DIR}" || -e "${RESULT_DIR}" ]]; then
  echo "refusing to overwrite existing or in-progress result directory: ${FINAL_RESULT_DIR}" >&2
  exit 3
fi
if [[ ! -f "${HOST_ENVIRONMENT_JSON}" ]]; then
  echo "host environment evidence does not exist: ${HOST_ENVIRONMENT_JSON}" >&2
  exit 4
fi
RUN_SCRIPT_SHA256=$(sha256sum -- "$0" | awk '{print $1}')
HOST_ENVIRONMENT_SHA256=$(sha256sum -- "${HOST_ENVIRONMENT_JSON}" | awk '{print $1}')

preserve_failed_run() {
  local status=${1:-$?}
  trap - ERR INT TERM HUP
  if [[ -d "${RESULT_DIR}" ]]; then
    local failed_dir="${FINAL_RESULT_DIR}.failed.$(date -u +%Y%m%dT%H%M%SZ).$$.${RANDOM}"
    mv -- "${RESULT_DIR}" "${failed_dir}"
  echo "FG-DS HIL failed; partial evidence preserved at: ${failed_dir}" >&2
  fi
  exit "${status}"
}
trap 'preserve_failed_run $?' ERR
trap 'preserve_failed_run 130' INT
trap 'preserve_failed_run 143' TERM
trap 'preserve_failed_run 129' HUP

mkdir -p "${RESULT_DIR}"
cp -- "${HOST_ENVIRONMENT_JSON}" "${RESULT_DIR}/host_environment.json"
if [[ "$(sha256sum -- "${RESULT_DIR}/host_environment.json" | awk '{print $1}')" != "${HOST_ENVIRONMENT_SHA256}" ]]; then
  echo "copied host environment evidence changed during capture" >&2
  exit 5
fi

run_stage() {
  local name=$1
  local count=$2
  local serial_timeout=$3
  local wall_timeout=$4
  local limit_args=()
  if [[ "${count}" -ne 56301 ]]; then
    limit_args=(--limit "${count}")
  fi

  timeout --foreground --signal=TERM --kill-after=30s "${wall_timeout}" \
    python deployment/hardware_hil/host/stream_vectors_fgds_strict.py \
    --port "${PORT}" \
    --generated-dir "${GENERATED_DIR}" \
    --bundle-dir "${BUNDLE_DIR}" \
    --vectors-csv "${VECTORS_CSV}" \
    --output-csv "${RESULT_DIR}/${name}_mcu.csv" \
    --summary-json "${RESULT_DIR}/${name}_sequence.json" \
    --timeout "${serial_timeout}" \
    "${limit_args[@]}"

  timeout --foreground --signal=TERM --kill-after=30s 30m \
    python deployment/hardware_hil/host/verify_results_fgds_strict.py \
    --mcu-csv "${RESULT_DIR}/${name}_mcu.csv" \
    --sequence-json "${RESULT_DIR}/${name}_sequence.json" \
    --generated-dir "${GENERATED_DIR}" \
    --bundle-dir "${BUNDLE_DIR}" \
    --reference-csv "${REFERENCE_CSV}" \
    --expected-count "${count}" \
    --output-json "${RESULT_DIR}/${name}_metrics.json"
}

run_stage smoke_10 10 2.0 10m
run_stage validation_1000 1000 2.0 30m
run_stage full_56301 56301 5.0 6h

python - "${RESULT_DIR}" "$0" "${RUN_SCRIPT_SHA256}" "${PORT}" "${GENERATED_DIR}" "${BUNDLE_DIR}" "${HOST_ENVIRONMENT_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
script = Path(sys.argv[2]).resolve()
run_script_sha256 = sys.argv[3]
serial_port = sys.argv[4]
generated_dir = Path(sys.argv[5]).resolve()
bundle_dir = Path(sys.argv[6]).resolve()
host_environment_sha256 = sys.argv[7]
manifest = root / "strict_hil_completion_manifest.json"
expected_stage_counts = {"smoke_10": 10, "validation_1000": 1000, "full_56301": 56301}
expected_names = {
    f"{stage}_{suffix}"
    for stage in expected_stage_counts
    for suffix in ["mcu.csv", "sequence.json", "metrics.json"]
}
expected_names.add("host_environment.json")
actual_names = {path.name for path in root.iterdir() if path.is_file() and path != manifest}
if actual_names != expected_names:
    raise RuntimeError(
        f"strict HIL stage inventory differs: missing={sorted(expected_names - actual_names)}, "
        f"unexpected={sorted(actual_names - expected_names)}"
    )
files = []
for path in sorted(root.iterdir()):
    if not path.is_file() or path == manifest:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest})
temporary = manifest.with_suffix(".json.tmp")
temporary.write_text(json.dumps({
    "status": "complete",
    "protocol_id": "fgds_seed42_strict_hil_three_stage_v1",
    "required_stages": expected_stage_counts,
    "run_script_path_recorded": str(script),
    "run_script_sha256_at_start": run_script_sha256,
    "run_script_sha256_at_completion": hashlib.sha256(script.read_bytes()).hexdigest(),
    "serial_endpoint_recorded": serial_port,
    "generated_dir_recorded": str(generated_dir),
    "bundle_dir_recorded": str(bundle_dir),
    "host_environment_sha256": host_environment_sha256,
    "execution_origin_boundary": (
        "The script records an operator-selected serial endpoint and verifies the "
        "firmware identity response. It does not cryptographically attest that the "
        "endpoint was a particular physical board."
    ),
    "file_count_excluding_manifest": len(files),
    "files": files,
}, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, manifest)
PY

mv -- "${RESULT_DIR}" "${FINAL_RESULT_DIR}"
trap - ERR INT TERM HUP
echo "FG-DS HIL completed: ${FINAL_RESULT_DIR}"
