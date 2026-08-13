#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GENERATED_DIR BUNDLE_DIR CONNECTION_JSON FINAL_RESULT_DIR" >&2
  exit 2
fi

SCRIPT_PATH=$(realpath -- "$0")
REPO_ROOT=$(realpath -- "$(dirname -- "${SCRIPT_PATH}")/../../..")
GENERATED_DIR=$(realpath -- "$1")
BUNDLE_DIR=$(realpath -- "$2")
SOURCE_CONNECTION_JSON=$(realpath -- "$3")
FINAL_RESULT_DIR=$(realpath -m -- "$4")
RESULT_DIR="${FINAL_RESULT_DIR}.in_progress"
REFERENCE_CSV="${GENERATED_DIR}/hil_reference_predictions.csv"
VECTORS_CSV="${GENERATED_DIR}/hil_replay_vectors.csv"
CONNECTION_JSON="${RESULT_DIR}/connection.json"

if [[ ! -d "${GENERATED_DIR}" || ! -d "${BUNDLE_DIR}" ]]; then
  echo "generated and bundle directories must already exist" >&2
  exit 3
fi
if [[ ! -f "${SOURCE_CONNECTION_JSON}" ]]; then
  echo "connection evidence does not exist: ${SOURCE_CONNECTION_JSON}" >&2
  exit 4
fi
if [[ -e "${FINAL_RESULT_DIR}" || -e "${RESULT_DIR}" ]]; then
  echo "refusing to overwrite existing or in-progress result: ${FINAL_RESULT_DIR}" >&2
  exit 5
fi

preserve_failed_run() {
  local status=${1:-$?}
  trap - ERR INT TERM HUP
  if [[ -d "${RESULT_DIR}" ]]; then
    local failed_dir="${FINAL_RESULT_DIR}.failed.$(date -u +%Y%m%dT%H%M%SZ).$$.${RANDOM}"
    mv -- "${RESULT_DIR}" "${failed_dir}"
    echo "Wireless HIL failed; partial evidence preserved at: ${failed_dir}" >&2
  fi
  exit "${status}"
}
trap 'preserve_failed_run $?' ERR
trap 'preserve_failed_run 130' INT
trap 'preserve_failed_run 143' TERM
trap 'preserve_failed_run 129' HUP

mkdir -p -- "${RESULT_DIR}"
cp -- "${SOURCE_CONNECTION_JSON}" "${CONNECTION_JSON}"
if [[ "$(sha256sum -- "${SOURCE_CONNECTION_JSON}" | awk '{print $1}')" != \
      "$(sha256sum -- "${CONNECTION_JSON}" | awk '{print $1}')" ]]; then
  echo "connection evidence changed while copying into the run" >&2
  preserve_failed_run 6
fi

cd -- "${REPO_ROOT}"

timeout --foreground --signal=TERM --kill-after=15s 5m \
  python -m deployment.wireless_hil.host.preflight_wireless_hil \
  --generated-dir "${GENERATED_DIR}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --connection-json "${CONNECTION_JSON}" \
  --output-json "${RESULT_DIR}/preflight.json" \
  --timeout 1.0 \
  --max-attempts 3

run_stage() {
  local stage_name=$1
  local replay_wall_timeout=$2

  timeout --foreground --signal=TERM --kill-after=30s "${replay_wall_timeout}" \
    python -m deployment.wireless_hil.host.stream_vectors_udp \
    --generated-dir "${GENERATED_DIR}" \
    --bundle-dir "${BUNDLE_DIR}" \
    --vectors-csv "${VECTORS_CSV}" \
    --connection-json "${CONNECTION_JSON}" \
    --stage-name "${stage_name}" \
    --output-csv "${RESULT_DIR}/${stage_name}_mcu.csv" \
    --summary-json "${RESULT_DIR}/${stage_name}_sequence.json" \
    --timeout 1.0 \
    --max-attempts 3

  timeout --foreground --signal=TERM --kill-after=30s 2h \
    python -m deployment.wireless_hil.host.verify_results_udp \
    --mcu-csv "${RESULT_DIR}/${stage_name}_mcu.csv" \
    --sequence-json "${RESULT_DIR}/${stage_name}_sequence.json" \
    --connection-json "${CONNECTION_JSON}" \
    --generated-dir "${GENERATED_DIR}" \
    --bundle-dir "${BUNDLE_DIR}" \
    --reference-csv "${REFERENCE_CSV}" \
    --stage-name "${stage_name}" \
    --output-json "${RESULT_DIR}/${stage_name}_metrics.json"
}

run_stage smoke_10 20m
run_stage validation_1000 2h
run_stage full_56301 24h

python -m deployment.wireless_hil.host.complete_wireless_run \
  --result-dir "${RESULT_DIR}" \
  --generated-dir "${GENERATED_DIR}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --connection-json "${CONNECTION_JSON}" \
  --run-script "${SCRIPT_PATH}"

mv -- "${RESULT_DIR}" "${FINAL_RESULT_DIR}"
trap - ERR INT TERM HUP
echo "FG-DS Wi-Fi UDP HIL completed: ${FINAL_RESULT_DIR}"
