# QaNN

QaNN (**Q**uantized  **N**earest **N**eighbors) is a C++20 vector search library (Flat, IVF, product quantization, exact re-ranking) with Python bindings.

```bash
pip install .
```

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
