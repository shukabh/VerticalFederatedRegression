#!/usr/bin/env bash
# Build the vfr_native Python extension against an installed OpenFHE (default /opt/openfhe-1.5.1).
# Usage: implementation/native/build.sh [OPENFHE_PREFIX]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${1:-${OPENFHE_PREFIX:-/opt/openfhe-1.5.1}}"
PY="${PYTHON:-python3}"
EXT="$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PYINC="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PBINC="$($PY -c 'import pybind11; print(pybind11.get_include())')"
INC="$PREFIX/include/openfhe"
OUT="$HERE/../vfr/vfr_native$EXT"
g++ -O3 -march=native -std=c++17 -fopenmp -fPIC -shared \
    -Wno-deprecated-declarations \
    -I"$PYINC" -I"$PBINC" -I"$INC" -I"$INC/core" -I"$INC/pke" -I"$INC/binfhe" -I"$INC/cereal" \
    "$HERE/vfr_native.cpp" -o "$OUT" \
    -L"$PREFIX/lib" -lOPENFHEpke -lOPENFHEcore -lOPENFHEbinfhe -lcrypto -lquadmath \
    -Wl,--disable-new-dtags -Wl,-rpath,"$PREFIX/lib"
echo "built $OUT"
