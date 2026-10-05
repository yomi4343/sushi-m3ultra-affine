# Clean source build and CPU verification

The fixed MLX and MLX-C sources were freshly configured/built/installed into an isolated stage, then Sushi was freshly built in ReleaseFast against that pair. MLX took about120 seconds, MLX-C about4 seconds, and Sushi about164 seconds. All stayed within the300-second guard and disk/memory/output reserves. Existing trusted offline dependencies, Xcode/AppleClang21, the existing Zig toolchain, the upstream tracked Jinja static archive and installed WebP were used. No dependency download or software installation occurred. This verifies one local source build, not cross-machine portability or deterministic binary reproduction.

`BUILD-VALIDATION.json` records pins, source patch order, binary/library hashes and actual completed stages. `--version` and `--guest-manifest` succeeded and the loader identified the new staged MLX/MLX-C/JACCL libraries. These commands perform the upstream Metal device discovery and process-local allocator policy; they load no model, start no server and run no inference workload. No system wired/security limits changed.

The targeted native CPU cases passed: metric boundary7, reducer default-off/admission1, Ngram malformed-header3, synthetic8KB BF16 Ngram mmap/pool gather1, Qwen config parsing1. Four auxiliary EXL3 test-artifact passes are separate from these13 targeted cases. New GPU numeric parity, full-model performance/quality and a corrected full-suite repeat remain unverified for this clean binary. The saved benchmark rates belong to the earlier experimental binary.

## Avoid zero-test filtered runs

At the fixed upstream revision, `src/tests.zig` imports most modules inside an unnamed `test` block. Applying a name filter can prune that block and its import seam, leaving no selected tests despite exit0. The first such runs are excluded from the test count. The supplied `tests/publication_cpu_tests.zig` imports the relevant modules unconditionally and runs their existing tests; it does not add replacement tests or change inference code.

In a fresh isolated review checkout with the completed staged build, copy this root into `src/`, temporarily change only the test root path in `build.zig`, and execute the targeted existing cases:

```sh
cp "$SUSHI_PATCH_DIR/tests/publication_cpu_tests.zig" src/publication_cpu_tests.zig
python3 - <<'PY'
from pathlib import Path
p = Path('build.zig')
backup = Path('build.zig.cpu-verification-original')
assert not backup.exists()
t = p.read_text()
assert t.count('b.path("src/tests.zig")') == 1
backup.write_text(t)
p.write_text(t.replace('b.path("src/tests.zig")', 'b.path("src/publication_cpu_tests.zig")'))
PY
export SUSHI_WIRED=off
export SUSHI_CACHE_LIMIT=1073741824
.zig-toolchain/zig build test -Doptimize=ReleaseSafe -j8 --summary all \
  '-Dtest-filter=prefill expert reduction: defaultOFF and restricted dispatch'
.zig-toolchain/zig build test -Doptimize=ReleaseSafe -j8 --summary all \
  '-Dtest-filter=ngram table header:'
.zig-toolchain/zig build test -Doptimize=ReleaseSafe -j8 --summary all \
  '-Dtest-filter=ngram table: a bf16 .bin gathers rows'
.zig-toolchain/zig build test -Doptimize=ReleaseSafe -j8 --summary all \
  '-Dtest-filter=parseConfigFromJson: qwen4_exp (Qwen3.8-Flash-Next) reads'
.zig-toolchain/zig test src/decode_metric_boundary.zig
python3 - <<'PY'
from pathlib import Path
Path('build.zig').write_text(Path('build.zig.cpu-verification-original').read_text())
PY
```

The inference executable was built before this test-only harness change. Restore the build script afterward; do not count an empty filtered runner as a passed predicate test. Keep raw loader/build logs local and publish only reviewed source and aggregate results.
