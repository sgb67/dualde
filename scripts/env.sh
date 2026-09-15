# Source from the dualde clone (or any cwd):  . scripts/env.sh
# Override the location with DUALDE_CACHE=/path/on/scratch
#
# Does not change $HOME. Redirects pip, tmp, Hugging Face, and Torch caches.

_dualde_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DUALDE_CACHE="${DUALDE_CACHE:-${_dualde_root}/.cache}"
case "${DUALDE_CACHE}" in
  "${HOME}"|"${HOME}"/*)
    echo "DUALDE_CACHE cannot be under \$HOME (${DUALDE_CACHE})" >&2
    unset _dualde_root
    return 1 2>/dev/null || exit 1
    ;;
esac
mkdir -p "${DUALDE_CACHE}/pip" "${DUALDE_CACHE}/tmp" \
  "${DUALDE_CACHE}/huggingface" "${DUALDE_CACHE}/torch" \
  "${DUALDE_CACHE}/matplotlib"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DUALDE_CACHE}/pip}"
export TMPDIR="${TMPDIR:-${DUALDE_CACHE}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DUALDE_CACHE}}"
export HF_HOME="${HF_HOME:-${DUALDE_CACHE}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${DUALDE_CACHE}/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DUALDE_CACHE}/matplotlib}"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset _dualde_root
