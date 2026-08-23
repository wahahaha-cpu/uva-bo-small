#!/usr/bin/env bash
set -euo pipefail

# Thin entry point for the released VLA-Adapter LIBERO-Long-Pro evaluation.
VLA_ADAPTER_ROOT="${VLA_ADAPTER_ROOT:-/home/jinboning/project/VLA-Adapter}"
launcher="$VLA_ADAPTER_ROOT/scripts/eval_official_libero10_pro_baseline.sh"
if [[ ! -x "$launcher" ]]; then
  echo "VLA-Adapter Pro launcher is missing or not executable: $launcher" >&2
  exit 2
fi
exec "$launcher" "$@"
