#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${DATA_ROOT}/pusht_multitask.zip}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/pusht_multitask}"
DRIVE_ID="14VqUC_LL411o9F_qdjVZlgiRBjZknw01"

ensure_data1_path() {
    local variable_name="$1"
    local path_value="${!variable_name}"
    if [[ "${path_value}" != /* ]]; then
        path_value="${REPO_ROOT}/${path_value}"
        printf -v "${variable_name}" '%s' "${path_value}"
    fi
    local resolved_path
    if [[ -e "${path_value}" || -L "${path_value}" ]]; then
        resolved_path="$(readlink -f "${path_value}")"
    else
        resolved_path="$(readlink -f "$(dirname "${path_value}")")"
    fi
    case "${resolved_path}" in
        /data1/*)
            ;;
        *)
            echo "Data path must resolve under /data1: ${path_value}" >&2
            exit 2
            ;;
    esac
}

for variable_name in DATA_ROOT ARCHIVE_PATH DATASET_PATH; do
    ensure_data1_path "${variable_name}"
done

mkdir -p "${DATA_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "Downloading official PushT-M archive to ${ARCHIVE_PATH}"
    "${PYTHON_BIN}" -m gdown "${DRIVE_ID}" -O "${ARCHIVE_PATH}"
else
    echo "Using existing archive: ${ARCHIVE_PATH}"
fi

unzip -t "${ARCHIVE_PATH}" >/dev/null
if [[ ! -d "${DATASET_PATH}/data" || ! -d "${DATASET_PATH}/meta" ]]; then
    echo "Extracting PushT-M archive into ${DATA_ROOT}"
    unzip -q "${ARCHIVE_PATH}" -d "${DATA_ROOT}"
fi

required_paths=(
    "${DATASET_PATH}/data/img"
    "${DATASET_PATH}/data/state"
    "${DATASET_PATH}/data/action"
    "${DATASET_PATH}/meta/episode_ends"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "PushT-M archive is missing expected path: ${required_path}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" - "${DATASET_PATH}" <<'PY'
import sys
import zarr

group = zarr.open(sys.argv[1], mode="r")
expected = {
    "data/img": (35927, 96, 96, 3),
    "data/state": (35927, 5),
    "data/action": (35927, 2),
    "meta/episode_ends": (247,),
}
for key, shape in expected.items():
    actual = tuple(group[key].shape)
    if actual != shape:
        raise SystemExit(f"Unexpected PushT-M shape for {key}: {actual}; expected {shape}")
print("PushT-M dataset shape check: PASS")
PY

echo "PushT-M dataset ready: ${DATASET_PATH}"
echo "Resolved storage: $(readlink -f "${DATASET_PATH}")"
