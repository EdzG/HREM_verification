#!/usr/bin/env bash
set -euo pipefail
# Optional quick GPU visibility check (non-fatal)
if [ -x /opt/rocm/bin/rocminfo ]; then
  /opt/rocm/bin/rocminfo >/dev/null 2>&1 || true
fi
exec "$@"
