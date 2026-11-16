#!/bin/sh
# Build the C++ core for the browser: robotnav/assets/robotnav_core.wasm (the viewer carries it inside the page).
# Needs wasi-sdk (https://github.com/WebAssembly/wasi-sdk); point WASI_SDK at its folder.
#   WASI_SDK=/opt/wasi-sdk tools/build_wasm.sh [out.wasm]
# The same sources as the native build, the same -ffp-contract=off, no threads and no exceptions. cpp/wasm/test.mjs
# then checks it against cpp/tests/golden.digest:  node cpp/wasm/test.mjs
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$here/robotnav/assets/robotnav_core.wasm}
: "${WASI_SDK:?point WASI_SDK at a wasi-sdk folder}"
"$WASI_SDK/bin/clang++" --target=wasm32-wasi -std=c++17 -O2 -ffp-contract=off -fno-exceptions -fno-rtti \
  -DROBOTNAV_NO_THREADS -mexec-model=reactor -Wl,--no-entry -Wl,--strip-all -Wl,--initial-memory=33554432 -Wl,--max-memory=268435456 \
  -I"$here/cpp/include" -o "$out" \
  "$here/cpp/src/map.cpp" "$here/cpp/src/planners.cpp" "$here/cpp/src/replan.cpp" "$here/cpp/src/airspace.cpp" \
  "$here/cpp/src/sensing.cpp" "$here/cpp/src/digest.cpp" "$here/cpp/wasm/api.cpp"
echo "wrote $out ($(wc -c < "$out") bytes)"
