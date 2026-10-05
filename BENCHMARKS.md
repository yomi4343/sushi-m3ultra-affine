# Retained measurements

All rates below are local saved evidence, not a new run of the extracted package. Hardware: Apple M3 Ultra, 96 GB. Model: original pre-weighted mixed 3/4/8 affine group-size-64 Qwen3.8-Flash-Next-Uncensored only. Cache reuse and measured compile misses were zero in the measured steady-state controls. Cold prefix means no prefix-cache reuse; first library compilation was conditioned before measurement.

| Evidence group | Scope | Supported result |
| --- | --- | --- |
| Five combined performance patches | 24 isolated servers, 72 cells, 36 matched pairs | Prefill gain range 4.476–10.555%; minimum candidate 915.323 tokens/s; single observation per version/mode/content/order, no statistical adoption claim |
| Added prefill reducer | 48 servers, 144 cells, 72 pairs, 36 strata | All 36 strata positive median prefill gain; 69 positive pairs, 3 negative; candidate minimum 922.3 tokens/s; all 72 candidate cells below 1000 |
| Latest corrected exact-MTP controls | Two controls, prompt 8130, full 256 completions, standard order | 3322.017667 / 3329.417500 ms; 256 divided by mean wall = 76.9758687 tokens/s; cold prefill 926.419 / 926.174 tokens/s |
| Separate saved normal-decode 32K controls | Prompt 31920, different experiment | Cold prefill 972.709 / 975.208 tokens/s; this says nothing about corrected exact-MTP throughput in the latest controls |

The broad suites used the old sum-of-ticks decode denominator. The corrected denominator includes the terminal interval previously missed (roughly 37–40 ms in the diagnosed requests). Therefore historical decode numbers are omitted. Normal decode and exact MTP are distinct modes; cached reuse is distinct from cold prefill.

The corrected reporting convention is **all published completion tokens / awake wall from prefill-end to terminal decision**. A first completion produced during prefill remains in the existing numerator convention; the helper separately exposes the post-prefill token count. Un-emitted EOS and discarded speculative suffixes are excluded. The end boundary precedes final statistics/persistence cleanup. This is not end-to-end client latency or throughput under concurrent interleaving. Streaming in-progress reports can still fall back to tick time until the terminal boundary exists.

Latest configuration: KV8, prefill chunk4096, prefix entries0, prefix memory4GB, prefix disk off, SSM stride4096/max16 requested but effective stride0 with prefix caching disabled, context32768, exact adaptive MTP cap3, EV seed on, lookup/PLD/cost-table planning/persistence off, allocator cache1GiB, wired residency off, MLX max ops50/max MB50. Six retained performance flags were on; diagnostic controls were off. A prior practical chunk8192/prefix4-entry configuration is a separate campaign setup, not this measurement configuration.

There is no corrected full-suite repeat, no fresh full-model/GPU qualification of the clean-package binary, and no cross-model or cross-machine qualification. Private fixtures and raw evidence are excluded; this package supplies aggregate facts and synthetic correctness sources. Public performance replication must record its own corpus, tokenizer/version, actual counts, order, adaptive history conditioning, warmup, cache/compile status, command and source/binary/library hashes. A single favorable run cannot establish the 1000/80 goal.
