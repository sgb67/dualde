#!/usr/bin/env bash
# Install dualde with caches on scratch, not $HOME.
#   ./scripts/install.sh              # core
#   ./scripts/install.sh notebooks    # + Jupyter
#   ./scripts/install.sh notebooks evo
# DUALDE_CACHE=/path/on/scratch ./scripts/install.sh notebooks

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/env.sh"

VENV="${ROOT}/.venv"
PYTHON="${PYTHON:-python3}"
EXTRAS=("$@")

"${PYTHON}" -m venv "${VENV}"

cat > "${VENV}/pip.conf" <<EOF
[global]
cache-dir = ${PIP_CACHE_DIR}
EOF

ACTIVATE="${VENV}/bin/activate"
if ! grep -q "dualde caches" "${ACTIVATE}"; then
  cat >> "${ACTIVATE}" <<EOF

# dualde caches (do not use \$HOME)
export DUALDE_CACHE="${DUALDE_CACHE}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export TMPDIR="${TMPDIR}"
export TEMP="${TEMP}"
export TMP="${TMP}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export TORCH_HOME="${TORCH_HOME}"
export MPLCONFIGDIR="${MPLCONFIGDIR}"
export PYTHONNOUSERSITE=1
EOF
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

SPEC="${ROOT}"
if ((${#EXTRAS[@]})); then
  joined=$(printf ",%s" "${EXTRAS[@]}")
  SPEC="${ROOT}[${joined:1}]"
fi

python -m pip install -U pip
python -m pip install -e "${SPEC}"

if ((${#EXTRAS[@]})) && printf '%s\n' "${EXTRAS[@]}" | grep -qx notebooks; then
  python -m ipykernel install --sys-prefix --name dualde --display-name dualde
fi

echo "Installed into ${VENV}"
echo "Caches: ${DUALDE_CACHE}"
echo "Activate with:  source ${VENV}/bin/activate"
