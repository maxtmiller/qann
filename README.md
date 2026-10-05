# QANN

**Q**uantized **A**pproximate **N**earest **N**eighbors: a C++20 vector search library (Flat, IVF, product quantization, exact re-ranking) with Python bindings.

> **Alpha:** under active development. The API may change between releases, and indexes cannot be saved or loaded yet.

```bash
pip install qann
```

Wheels are published for Linux (x86_64, aarch64) and macOS (Apple Silicon, Intel) on Python 3.9 to 3.13. x86_64 builds require AVX2 and FMA. To build from source instead, run `pip install .` in a checkout; this needs CMake 3.20+, a C++20 compiler and, on Linux, OpenBLAS (optional but much faster training).

```python
import numpy as np, qann

x = np.random.rand(10_000, 128).astype(np.float32)
ivf = qann.IVFIndex(128, nlist=100, nprobe=10)
ivf.enable_pq(16)
ivf.train(x, seed=1)
refine = qann.RefineIndex(ivf, k_factor=10)
refine.add(x)
ids, dists = refine.batch_query(x[:5], 10)   # (5, 10) arrays
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for design, benchmarks and build details.
