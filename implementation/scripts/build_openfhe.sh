#!/usr/bin/env bash
# Build and install OpenFHE 1.5.1 (the version used by this implementation) from source.
#   scripts/build_openfhe.sh [PREFIX=/opt/openfhe-1.5.1] [NATIVE_SIZE=64]
# NATIVE_SIZE=128 builds the 128-bit native-integer variant used only for the scale-option benchmark.
set -euo pipefail
PREFIX="${1:-/opt/openfhe-1.5.1}"
NATIVE="${2:-64}"
SRC="${SRC:-$(mktemp -d)/openfhe-development}"
if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch v1.5.1 https://github.com/openfheorg/openfhe-development "$SRC"
fi
mkdir -p "$SRC/build$NATIVE" && cd "$SRC/build$NATIVE"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
         -DBUILD_BENCHMARKS=OFF -DBUILD_EXTRAS=OFF -DWITH_OPENMP=ON -DNATIVE_SIZE="$NATIVE"
make -j"$(nproc)"
make install
echo "installed OpenFHE 1.5.1 (NATIVE_SIZE=$NATIVE) to $PREFIX"
