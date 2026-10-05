# Licenses and attribution

These are source-code terms observed in the fixed source objects; this document is not a legal opinion.

| Source | Fixed revision | Included notices |
| --- | --- | --- |
| [Sushi](https://github.com/beamivalice/sushi/tree/711572e91c491b50b973e57045b8d4a7f5137764) | 711572e91c491b50b973e57045b8d4a7f5137764 | `licenses/SUSHI-LICENSE`, `licenses/SUSHI-NOTICE`, `licenses/SUSHI-LICENSE-APACHE-2.0` |
| [MLX](https://github.com/ml-explore/mlx/tree/d73eb752ef2e6288fd95b032c0bff0a15a4a9e93) | d73eb752ef2e6288fd95b032c0bff0a15a4a9e93 | `licenses/MLX-LICENSE`; Apple copyright comments retained in modified and embedded source |
| [MLX-C](https://github.com/ml-explore/mlx-c/tree/56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6) | 56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6 | `licenses/MLXC-LICENSE` |

The MIT files require retention of their copyright and permission notices. Sushi's LICENSE explicitly preserves third-party terms; its full NOTICE is included unchanged rather than discarding attributions for inherited portions. Existing Apache-licensed attribution includes MTPLX by Youssof Altoukhi and its credited upstreams. The reducer patch extends existing Sushi code and its reduction implementation; this package does not replace the upstream license of that code. Embedded MLX Metal code in the MLX-C performance patches remains Apple-origin code under its own MIT notice. Retain these files and existing source comments when distributing substantial portions.

No model weights, quantized banks, tables, tokenizer or model card are included. Model licenses and exact derivative ancestry require a separate review before any model distribution. The previously inspected Qwen model LICENSE/card disagreement is unresolved for that separate payload and is not converted into an Apache-only claim here. There is no claim of permission to publish any model or private benchmark corpus.

No new contributor identity or copyright owner is invented. Before destination-specific publication, the owner should choose the contribution terms and retain upstream notices, check any destination contribution/DCO/CLA requirements, and review the exact archive. No PR description or commit message is supplied in this package.
