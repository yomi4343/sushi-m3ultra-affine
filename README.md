# Sushi opt-in M3 Ultra source patches

This is a public experimental source patch kit, prepared locally on 2026-10-05 JST. It contains six retained performance changes and one reporting correction. The performance flags are off unless explicitly set to `1`. The package does not contain weights, n-gram tables, compiled libraries, model prompts or outputs, request state, private logs, credentials, or raw traces.

`SERIES.json` is the exact patch order, repository mapping and fixed source commit list. `MANIFEST.json` lists every payload file and its SHA-256. The patch files use relative paths and apply with ordinary `git apply`; no unsafe-path option is needed.

| Repository | Patch | Purpose |
| --- | --- | --- |
| MLX | 0001-bk32 | BF16 dense prefill GEMM BK32, restricted to qualified M3 Ultra architecture, shapes and widths |
| MLX-C | 0002-sg2 | Restricted small-row, 8-bit MTP projection kernel; strict source and signature checks |
| MLX-C | 0003-aligned32-tail16 | Grouped expert rows in aligned 32-row blocks, 16-row tail path |
| MLX-C | 0004-indexed-input | Reuse original input rows for admitted expert gate/up Gather graphs; requires aligned32 flag |
| MLX-C | 0005-stock-qfrag | QSA BF16 query fragment, with strict source/header/signature checks |
| Sushi | 0006-prefill-expert-reduce | Fuse inverse permutation and weighted expert reduction for qualified BF16 prefill |
| Sushi | 0007-complete-generation-metric | Complete post-prefill reporting clock; internal tick accounting is preserved |

The first six patches preserve the retained local patch bytes. Patch 0007 is a new extraction of the verified reporting change onto the clean Sushi pin: it replaces diagnostic phase hooks with timestamps at the same prefill/terminal seams and includes the exact tested `decode_metric_boundary.zig` helper. It adds no diagnostic ledger or GPU synchronization. A clean combined MLX/MLX-C/Sushi source build and targeted native CPU tests have now passed on the same Mac. Full-model/GPU parity and performance have not been rerun for this clean binary; see BUILD-VALIDATION.json and CPU-VALIDATION.md.

The proposed publication excludes all diagnostic observers and ledgers, dense BF16 memoization, capture evaluation, command-buffer experiments, head/body overlap, terminal predraft changes, other rejected kernels and unfinished prototypes. No full experimental source tree is shipped.

## Measurements and limitations

See `BENCHMARKS.json`, `BENCHMARKS.md`, and `BUILD-VALIDATION.json`. The sole full-model qualification target was an original pre-weighted Qwen3.8-Flash-Next-Uncensored mixed 3/4/8 affine group-size-64 checkpoint on an M3 Ultra with 96 GB memory. These results do not qualify other models, quantizations, Apple GPUs or production deployment.

The latest corrected exact-MTP control is **76.9759 tokens/s**, with 256 published completions over mean 3325.7176 ms in two runs. Its prompt is 8130 tokens. Its cold-prefill rates are 926.419 and 926.174 tokens/s. Separate earlier normal-decode 32K controls reached 972.709 and 975.208 cold-prefill tokens/s. They are different experiments and cannot be combined into a simultaneous result. The 1000-prefill / 80-decode target is not robustly met.

Earlier broad qualification supports the retained prefill gains, but used a legacy decode metric that missed a final interval. Old decode rates must not be advertised as corrected performance. There is no corrected full-suite repeat or fresh full-model qualification of this clean package. Private prompts are excluded, so the exact private corpus is not reproducible from this package. Public synthetic numeric fixtures are included for independent correctness checks; performance on a new public corpus requires new measurement.

## Build, checks and measurement

Follow `REPRODUCE.md` in fresh isolated checkouts with an already configured trusted toolchain. No install, build, GPU launch or publication is performed by this package. Preserve all existing production installations and model files. Keep all licenses and notices with copies of the changed source.

## Attribution and model boundary

Sushi's own code is MIT; its existing third-party portions retain their licenses. The complete Sushi NOTICE and Apache-2.0 text are included unchanged, together with Sushi, MLX and MLX-C MIT license texts. MLX-derived kernel source retains its Apple copyright comments. Attribution and exact pinned-source links are in `LICENSES.md`.

Engine-source licensing does not establish permission to distribute a model. No model payload is part of this proposal. A separately inspected model repository's actual LICENSE was Qwen Community License 1.0 while its card claimed Apache-2.0; The proposal follows the actual Qwen license with preserved notices; the card mismatch alone is not treated as a permanent distribution prohibition. Exact training ancestry and complete independent conversion reconstruction remain disclosed provenance limits. This package does not grant model redistribution rights or legal clearance.
