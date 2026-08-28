# eindex

einops-style notation for tensor **indexing**, originally by [Callum McDougall](https://www.perfectlynormal.co.uk/blog-eindex).
This is the [ARENA](https://github.com/callummcdougall/ARENA_3.0) fork: **same API, same patterns, same
results** — but the pattern is parsed once and the call runs at the speed of a hand-written `torch.gather`.

Install:

```
pip install git+https://github.com/ARENA-education/eindex.git
```

Import and use exactly as before:

```python
from eindex import eindex

output = eindex(logprobs, labels, "batch seq [batch seq]")
```

which sets the elements of the 2D `output` tensor as follows:

```
output[batch, seq] = logprobs[batch, seq, labels[batch, seq]]
```

See the accompanying [Colab notebook](https://colab.research.google.com/drive/1KbuRsoKTMrgjtOQgUDeam8GWX0k1YzmO?usp=sharing)
(or [Callum's blog](https://www.perfectlynormal.co.uk/blog-eindex)) for the pattern language, and `demo.ipynb` in this repo.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/indexing.png" width="320">

## Pattern language (unchanged)

| pattern | meaning |
| --- | --- |
| `"batch seq [batch seq]"` | `out[b,s] = arr[b, s, idx[b,s]]` |
| `"batch seq [batch seq] -> seq batch"` | same, output transposed |
| `"batch seq [batch seq] [batch seq]"` with two index tensors | `out[b,s] = arr[b, s, i1[b,s], i2[b,s]]` |
| `"batch seq [batch seq 0] [batch seq 1]"` with one `(b,s,2)` tensor | `out[b,s] = arr[b, s, idx[b,s,0], idx[b,s,1]]` |
| `"batch [batch] d_vocab"` | `out[b,v] = arr[b, idx[b], v]` (index a middle axis) |
| `"batch [batch seqQ k]"` | `out[b,q,k] = arr[b, idx[b,q,k]]` (brackets introduce output axes) |
| `"batch seq [batch seq+1]"` | `out[b,s] = arr[b, s, idx[b,s+1]]`, output has `seq-1` positions |
| `"b s [b s k2] b s [b s k1] -> b s k2 k1"` | **new**: repeated bare axes index the diagonal — `out[b,s,k2,k1] = jac[b, s, oi[b,s,k2], b, s, ii[b,s,k1]]` |

The last row is [upstream issue #4](https://github.com/callummcdougall/eindex/issues/4), which the original
implementation raised on. Numpy arrays are accepted in and returned out. `eindex(..., verbose=True)` prints
the inferred axis sizes and output shape.

## What's different in this fork

**Speed.** The original re-parsed and re-validated the pattern string on every call, ran
`torch.tensor(shape).prod().item()` device-sync asserts, and indexed with a Python *list* (torch's slow,
deprecated non-tuple path). `eindex` here compiles each pattern once into a closure of native torch ops
(a plain `torch.gather` for single-bracket patterns, a broadcast tuple index otherwise) and caches it.
Measured on the two call sites that sit inside ARENA training loops, µs per call:

| | `"env time [env time] -> env time"` on `(4, 128, 2)` — CPU | same — CUDA (A40) | `"b s [b s+1]"` on real GPT-2-small logits `(3, 402, 50257)` — CUDA |
| --- | ---: | ---: | ---: |
| original `eindex` 0.1.1 | 111 | 420 | 240 |
| this `eindex` (drop-in) | 11 | 17 | 36 |
| `compile_eindex(...)` closure | 11 | 16 | — |
| hand-written `torch.gather` | 7 | 12 | 28 |

Run `python bench/bench_eindex.py` for the full size sweep (it writes plots to `bench/`, which are
git-ignored — this repo stays image-free).

**`compile_eindex`.** For a hot loop, or to hand a pure closure to `torch.compile`:

```python
from eindex import compile_eindex

pick = compile_eindex("batch [batch]")     # parse once ...
for _ in range(steps):
    child = pick(node_child, action)       # ... call at torch.gather speed
```

The closures are `torch.compile(fullgraph=True)`-clean (no graph breaks); they don't call
`torch.compile` themselves. `compile_eindex(pattern, validate=False)` skips the argument checks
(~3 µs) if you really need it.

**Errors.** Shape / pattern mistakes raise `eindex.EindexError` with a message that names the
problem (which axis, which sizes). It subclasses both `ValueError` and `AssertionError`, so code
written against the original's `except AssertionError` keeps working.

**Dependencies.** `torch` and `numpy` only (`einops` is no longer required).

## Tests

```
pip install -e ".[test]"
pytest
```

The suite checks every documented pattern, every pattern used in the ARENA course, and a random-shape
fuzz against an unmodified copy of the original implementation (`tests/reference/`), plus the
diagonal case against a ground-truth loop, error messages, caching and `torch.compile` cleanliness.
