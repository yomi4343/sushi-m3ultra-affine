# Source application and independent checks

Use new isolated local checkouts. The commands below are review instructions; a clean combined source build has been performed locally using existing offline dependencies. Cross-machine reproduction and new full-model/GPU qualification remain unverified. Already-local fixed Git objects were used to check patch application. No dependency or tool installation is part of the performed verification.

Requirements follow the pinned Sushi build: Apple Silicon macOS, Xcode/Metal SDK 26.2+, CMake, an appropriate Zig 0.17 development toolchain, and the pinned project's existing dependencies. Do not change system wired-memory/security settings. The exact machine-local dependency cache and binary install names used by the earlier campaign are deliberately excluded.

```sh
# Extract the archive first. Record the absolute extracted package directory.
SUSHI_PATCH_DIR="$PWD/payload"
git clone --recurse-submodules https://github.com/beamivalice/sushi.git sushi-review
cd sushi-review
git checkout 711572e91c491b50b973e57045b8d4a7f5137764
git submodule update --init --recursive
git -C lib/mlx-src checkout d73eb752ef2e6288fd95b032c0bff0a15a4a9e93
git -C lib/mlxc-src checkout 56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6
git -C lib/mlx-src apply --check "$SUSHI_PATCH_DIR/patches/mlx/0001-bk32.patch"
git -C lib/mlx-src apply "$SUSHI_PATCH_DIR/patches/mlx/0001-bk32.patch"
git -C lib/mlxc-src apply "$SUSHI_PATCH_DIR/patches/mlx-c/0002-sg2.patch"
git -C lib/mlxc-src apply "$SUSHI_PATCH_DIR/patches/mlx-c/0003-aligned32-tail16.patch"
git -C lib/mlxc-src apply "$SUSHI_PATCH_DIR/patches/mlx-c/0004-indexed-input.patch"
git -C lib/mlxc-src apply "$SUSHI_PATCH_DIR/patches/mlx-c/0005-stock-qfrag.patch"
git apply "$SUSHI_PATCH_DIR/patches/sushi/0006-prefill-expert-reduce.patch"
git apply "$SUSHI_PATCH_DIR/patches/sushi/0007-complete-generation-metric.patch"
```

Inspect each diff against `SERIES.json`. Check each patch before its apply when repeating the procedure. Patches 0002–0005 share the MLX-C tree and are cumulative in that order. Patch 0006 has no server diagnostic amendment dependency. Patch 0007 includes the new helper in the patch itself.

Once dependencies and the trusted Zig toolchain are already available, the pinned upstream build routes are:

```sh
./scripts/build-mlx.sh
.zig-toolchain/zig build -Doptimize=ReleaseFast -j8
.zig-toolchain/zig test src/decode_metric_boundary.zig
```

Use a fresh stage: the upstream build script's version stamp is based on commit IDs and can skip a previously staged tree despite uncommitted patches. Confirm the new executable links the newly built MLX and MLX-C, and record library/source hashes. Do not copy over a production binary. A successful patch application and helper test are not a full engine build or a GPU parity test. The clean MLX/MLX-C/Sushi build and targeted CPU tests have passed locally. Numeric GPU tests and controlled public-corpus qualification remain pending for this clean binary. Read CPU-VALIDATION.md for the test-root import workaround needed to avoid zero-test filtered runs.

## Synthetic numeric fixtures

`tests/` contains trusted source-only boundary fixtures used in the campaign. Seeds and random tensors are synthetic. Portable copies change only CLI argument guards and the required actual-library path comparison; no numerical algorithm is changed. They have not been freshly run against the extracted package.

Build C++ fixtures against the matching source headers and newly staged libraries, for example:

```sh
c++ -std=c++20 -O2 -Ilib/mlx-src -Ilib/mlxc-src -Ilib/mlx/include \
  "$SUSHI_PATCH_DIR/tests/bk32-boundaries.cpp" -Llib/mlx/lib \
  -Wl,-rpath,"$PWD/lib/mlx/lib" -lmlx -lmlxc -o bk32-boundaries
```

Use the same compilation pattern with each other `.cpp` file. `bk32-boundaries` takes no argument. `sg2-capi-boundaries` takes the `tests` directory containing `body.metal`; `sg2-factory-guard` takes the `body.metal` file. `aligned32-capi-boundaries` and `indexed-capi-boundaries` take the actual freshly linked MLX-C dylib path. `qfrag-capi-boundaries` takes the `tests` directory and that dylib path. Keep the actual-library identity checks enabled. Reducer synthetic and dispatch tests are included in patch 0006 in `transformer.zig`; use the pinned project's test build after preparing the full engine dependencies.

Run one GPU fixture or full-model workload at a time. Preflight the GPU lock/processes, disk and memory; retain a 16 GiB disk floor and stop each run at 300 seconds. Set an output cap appropriate to the test and stop owned servers at safe boundaries. This package contains no raw trace capture instructions.

## Performance protocol

All six retained flags default off. Enable them only in the isolated qualified target experiment:

```sh
export SUSHI_GEMM_PREFILL_BK32=1
export SUSHI_MTP_QMV8_SG2=1
export SUSHI_MOE_ALIGNED32_TAIL16=1
export SUSHI_MOE_INDEXED_INPUT=1
export SUSHI_QSA_QFRAG_BF16=1
export SUSHI_PREFILL_EXPERT_REDUCE=1
```

SG2 and QSA substitutions require their exact Sushi source/header/signatures; an enabled mismatched signature deliberately raises an error. Shape/graph admission checks otherwise retain ordinary paths. Do not enable these flags globally for unrelated engines or models. Recheck numerical equality/state/output parity and dispatch boundaries before measuring on any changed source, library, GPU or checkpoint.

Replicate the explicit settings in `BENCHMARKS.json` with the pinned engine's CLI/env interfaces. Use synthetic public native-tool, correction and state fixtures. Benchmark ordinary decode and exact MTP separately. Use controlled full-length warmup with fixed request order to condition compilation and adaptive history, then new cold prefixes with prefix entries0. Record actual token counts, compile misses0 and cached tokens0. Compare matched BASE/DIAG/DIAG/BASE or control/candidate pairs without treating diagnostics as optimizations. Keep private raw evidence local and publish only separately reviewed summaries.

Exact reproduction of the private corpus is intentionally unavailable. A public-corpus study should supply its own shareable prompt generator and a fresh result table. The 1000/80 goal and broad generalization must be judged on repeated corrected measurements, not the legacy tick-only decode numbers.
