"""Tests for the compile-once `eindex` against the original implementation (vendored, unmodified, in
`tests/reference/` -- eindex-callum 0.1.1) and against ground-truth loops.

Covers: the six blog examples (basic, reorder, integer slots, multiple tensors, mid-axis index,
bracket-introduced axes, offsets), numpy I/O, every pattern used in the ARENA course, a random-shape
fuzzer, the issue-#4 repeated-axis / diagonal case (the reference raises; we check against a loop),
error messages (and their `AssertionError` compatibility), caching, `verbose`, and torch.compile.

    pip install -e ".[test]" && pytest
"""

import contextlib
import io

import numpy as np
import pytest
import torch
from reference import eindex as ref_eindex  # the original (tests/reference, see conftest.py)

from eindex import EindexError, compile_eindex, eindex
from eindex.indexing import _compiled

G = torch.Generator().manual_seed(0)


def _rn(*shape):
    return torch.randn(*shape, generator=G)


def _ri(hi, shape):
    return torch.randint(0, hi, shape, generator=G)


def _same(a, b):
    return a.shape == b.shape and torch.equal(a, b)


# --------------------------------------------------------------------------------------------------
# The documented examples (https://www.perfectlynormal.co.uk/blog-eindex)
# --------------------------------------------------------------------------------------------------


def test_example1_basic():
    lp, lab = _rn(32, 5, 100), _ri(100, (32, 5))
    assert _same(eindex(lp, lab, "batch seq [batch seq]"), ref_eindex(lp, lab, "batch seq [batch seq]"))


@pytest.mark.parametrize("p", ["batch seq [batch seq] -> seq batch", "batch seq [batch seq] -> batch seq"])
def test_example1_reorder(p):
    lp, lab = _rn(32, 5, 100), _ri(100, (32, 5))
    assert _same(eindex(lp, lab, p), ref_eindex(lp, lab, p))


def test_example2a_integer_slots():
    lp = _rn(32, 5, 100, 50)
    lab = torch.stack([_ri(100, (32, 5)), _ri(50, (32, 5))], dim=-1)
    p = "batch seq [batch seq 0] [batch seq 1]"
    assert _same(eindex(lp, lab, p), ref_eindex(lp, lab, p))


def test_example2b_multiple_tensors():
    lp, l1, l2 = _rn(32, 5, 100, 50), _ri(100, (32, 5)), _ri(50, (32, 5))
    p = "batch seq [batch seq] [batch seq]"
    assert _same(eindex(lp, l1, l2, p), ref_eindex(lp, l1, l2, p))


def test_example3_mid_axis_index():
    lp, lab = _rn(32, 5, 100), _ri(5, (32,))
    p = "batch [batch] d_vocab"
    assert _same(eindex(lp, lab, p), ref_eindex(lp, lab, p))


def test_example4_bracket_introduces_axes():
    arr, idx = _rn(32, 7), _ri(7, (32, 4, 3))
    p = "batch [batch seqQ k]"
    assert _same(eindex(arr, idx, p), ref_eindex(arr, idx, p))


def test_example5_offset():
    lp, tok = _rn(32, 5, 100), _ri(100, (32, 5))
    p = "batch seq [batch seq+1]"
    out = eindex(lp, tok, p)
    assert out.shape == (32, 4)
    assert _same(out, ref_eindex(lp, tok, p))
    assert _same(out, eindex(lp[:, :-1], tok[:, 1:], "batch seq [batch seq]"))


def test_numpy_io():
    lp = np.random.randn(8, 5, 20).astype(np.float32)
    lab = np.random.randint(0, 20, (8, 5))
    out = eindex(lp, lab, "batch seq [batch seq]")
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, ref_eindex(lp, lab, "batch seq [batch seq]"))


# --------------------------------------------------------------------------------------------------
# Every pattern used in the ARENA course material
# --------------------------------------------------------------------------------------------------

ARENA_PATTERNS = [
    ("batch [batch] -> batch", lambda: (_rn(64, 7), [_ri(7, (64,))])),
    ("b s [b s+1]", lambda: (_rn(8, 16, 50), [_ri(50, (8, 16))])),
    ("game pos row col [game pos row col]", lambda: (_rn(4, 10, 8, 8, 3), [_ri(3, (4, 10, 8, 8))])),
    ("env time [env time] -> env time", lambda: (_rn(4, 128, 2), [_ri(2, (4, 128))])),
    ("s [s] s_new -> s s_new", lambda: (_rn(16, 4, 16), [_ri(4, (16,))])),
]


@pytest.mark.parametrize("p,make", ARENA_PATTERNS, ids=[p for p, _ in ARENA_PATTERNS])
def test_arena_course_patterns(p, make):
    arr, idx = make()
    assert _same(eindex(arr, *idx, p), ref_eindex(arr, *idx, p))


# --------------------------------------------------------------------------------------------------
# Fuzz over random shapes
# --------------------------------------------------------------------------------------------------

FUZZ_PATTERNS = {
    "batch seq [batch seq]": lambda B, S, V: (_rn(B, S, V), [_ri(V, (B, S))]),
    "batch seq [batch seq] -> seq batch": lambda B, S, V: (_rn(B, S, V), [_ri(V, (B, S))]),
    "batch seq [batch seq] [batch seq]": lambda B, S, V: (_rn(B, S, V, V), [_ri(V, (B, S)), _ri(V, (B, S))]),
    "batch seq [batch seq 0] [batch seq 1]": lambda B, S, V: (_rn(B, S, V, V), [_ri(V, (B, S, 2))]),
    "batch seq [batch seq+1]": lambda B, S, V: (_rn(B, S, V), [_ri(V, (B, S))]),
    "batch [batch] d": lambda B, S, V: (_rn(B, S, V), [_ri(S, (B,))]),
    "batch [batch seqQ k]": lambda B, S, V: (_rn(B, V), [_ri(V, (B, S, 3))]),
}


@pytest.mark.parametrize("p", list(FUZZ_PATTERNS))
def test_fuzz_random_shapes(p):
    make = FUZZ_PATTERNS[p]
    for _ in range(40):
        B, S, V = int(_ri(8, (1,)) + 1), int(_ri(8, (1,)) + 2), int(_ri(12, (1,)) + 1)
        arr, idx = make(B, S, V)
        assert _same(eindex(arr, *idx, p), ref_eindex(arr, *idx, p)), (p, B, S, V)


# --------------------------------------------------------------------------------------------------
# Issue #4: repeated bare axis (diagonal). The reference raises on this; check against a loop.
# --------------------------------------------------------------------------------------------------


def test_issue4_repeated_axis_diagonal():
    b, s, k, feat = 2, 3, 5, 7
    jac = _rn(b, s, feat, b, s, feat)
    oi, ii = _ri(feat, (b, s, k)), _ri(feat, (b, s, k))
    gt = torch.empty(b, s, k, k)
    for bb in range(b):
        for ss in range(s):
            for k2 in range(k):
                for k1 in range(k):
                    gt[bb, ss, k2, k1] = jac[bb, ss, oi[bb, ss, k2], bb, ss, ii[bb, ss, k1]]
    p = "b s [b s k2] b s [b s k1] -> b s k2 k1"
    assert torch.equal(eindex(jac, oi, ii, p), gt)
    with pytest.raises(AssertionError):  # the reference can't do this
        ref_eindex(jac, oi, ii, p)


def test_repeated_axis_plain_diagonal():
    # out[i, k] = A[i, i, idx[i, k]]
    n, m, k = 6, 9, 4
    A, idx = _rn(n, n, m), _ri(m, (n, k))
    gt = torch.stack([A[i, i, idx[i]] for i in range(n)])
    assert torch.equal(eindex(A, idx, "i i [i k]"), gt)


# --------------------------------------------------------------------------------------------------
# Fast path, caching, compile-cleanliness, verbose
# --------------------------------------------------------------------------------------------------


def test_gather_fast_path_matches_native():
    nc, a = _ri(50, (256, 7)), _ri(7, (256,))
    assert torch.equal(eindex(nc, a, "batch [batch]"), nc.gather(1, a.unsqueeze(1)).squeeze(1))
    lp, lab = _rn(32, 5, 100), _ri(100, (32, 5))
    assert torch.equal(eindex(lp, lab, "batch seq [batch seq]"), lp.gather(2, lab.unsqueeze(2)).squeeze(2))


def test_pattern_cached():
    _compiled.cache_clear()
    a, i = _ri(7, (4, 7)), _ri(7, (4,))
    eindex(a, i, "batch [batch]")
    eindex(a, i, "batch [batch]")
    assert _compiled.cache_info().hits >= 1, "repeated pattern should hit the compile cache"


def test_torch_compile_clean():
    nc, a = _ri(7, (64, 7)), _ri(7, (64,))
    f = compile_eindex("batch [batch]")
    assert torch.equal(torch.compile(f, fullgraph=True)(nc, a), f(nc, a))  # fullgraph errors on any break
    lp, tk = _rn(8, 5, 10), _ri(10, (8, 5))
    g = compile_eindex("batch seq [batch seq+1]")
    assert torch.equal(torch.compile(g, fullgraph=True)(lp, tk), g(lp, tk))


def test_compile_eindex_validate_flag():
    lp, lab = _rn(4, 5, 10), _ri(10, (4, 5))
    for validate in (True, False):
        f = compile_eindex("batch seq [batch seq]", validate=validate)
        assert torch.equal(f(lp, lab), ref_eindex(lp, lab, "batch seq [batch seq]"))


def test_verbose():
    lp, tok = _rn(4, 5, 10), _ri(10, (4, 5))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = eindex(lp, tok, "batch seq [batch seq+1]", verbose=True)
    assert out.shape == (4, 4)
    assert "output shape" in buf.getvalue()


# --------------------------------------------------------------------------------------------------
# Errors: helpful messages, raised as EindexError (a ValueError AND an AssertionError, so code written
# against the original's `except AssertionError` keeps working)
# --------------------------------------------------------------------------------------------------

BAD_CALLS = {
    "array ndim mismatch": (lambda: eindex(_rn(4, 5, 6), _ri(6, (4, 5)), "batch [batch seq]"), "3 dimensions"),
    "index ndim mismatch": (lambda: eindex(_rn(4, 5, 6), _ri(6, (4,)), "batch seq [batch seq]"), "Bracket #0"),
    "contradictory sizes": (
        lambda: eindex(_rn(4, 5, 6), _ri(6, (4, 7)), "batch seq [batch seq]"),
        "Incompatible sizes",
    ),
    "offset too large": (
        lambda: eindex(_rn(4, 5, 6), _ri(6, (4, 5)), "batch seq [batch seq+5]"),
        "too large",
    ),
    "bad -> names": (lambda: eindex(_rn(4, 5, 6), _ri(6, (4, 5)), "batch seq [batch seq] -> batch foo"), "'->'"),
    "too many tensors": (
        lambda: eindex(_rn(4, 5, 6), _ri(6, (4, 5)), _ri(6, (4, 5)), "batch seq [batch seq]"),
        "2 index tensors",
    ),
    "no brackets": (lambda: eindex(_rn(4, 5), _ri(4, (4,)), "batch seq"), "nothing to index"),
    "slot out of range": (lambda: eindex(_rn(4, 5, 6), _ri(5, (4, 1)), "batch [batch 0] [batch 1]"), "out of range"),
    "unbalanced brackets": (lambda: eindex(_rn(4, 5), _ri(5, (4,)), "batch [batch"), "Unbalanced"),
}


@pytest.mark.parametrize("name", list(BAD_CALLS))
def test_errors_are_helpful(name):
    fn, needle = BAD_CALLS[name]
    with pytest.raises(EindexError, match=needle):
        fn()
    with pytest.raises(AssertionError):  # backwards compatible with the original
        fn()
    with pytest.raises(ValueError):
        fn()


def test_type_errors():
    with pytest.raises(TypeError):
        eindex(_rn(4, 5), _ri(5, (4,)))  # no pattern
    with pytest.raises(TypeError):
        eindex(_rn(4, 5), _ri(5, (4,)), "b [b]", foo=1)  # unknown kwarg


def test_demo_notebook_error_cells():
    # exactly what demo.ipynb does: catch AssertionError and print the message
    lp, lab = _rn(32, 5, 100), _ri(100, (33, 5))
    with pytest.raises(AssertionError, match="Incompatible sizes"):
        eindex(lp, lab, "batch seq [batch seq]")
    with pytest.raises(AssertionError, match="2 terms"):
        eindex(lp, _ri(100, (32, 5)), "batch [batch seq]")


def test_generic_path_plan_cache_and_shape_change():
    # multi-bracket -> generic path; the per-shape plan must not leak between shapes or devices
    p = "batch seq [batch seq] [batch seq]"
    for B, S, V in [(4, 5, 6), (3, 7, 6), (4, 5, 6)]:
        lp, l1, l2 = _rn(B, S, V, V), _ri(V, (B, S)), _ri(V, (B, S))
        assert _same(eindex(lp, l1, l2, p), ref_eindex(lp, l1, l2, p))
    with pytest.raises(EindexError):
        eindex(_rn(4, 5, 6, 6), _ri(6, (4, 5)), _ri(6, (4, 9)), p)


def test_generic_path_torch_compile_clean():
    lp, l1, l2 = _rn(4, 5, 6, 6), _ri(6, (4, 5)), _ri(6, (4, 5))
    g = compile_eindex("batch seq [batch seq] [batch seq]")
    assert torch.equal(torch.compile(g, fullgraph=True)(lp, l1, l2), g(lp, l1, l2))
    jac = _rn(2, 3, 7, 2, 3, 7)
    oi, ii = _ri(7, (2, 3, 5)), _ri(7, (2, 3, 5))
    h = compile_eindex("b s [b s k2] b s [b s k1] -> b s k2 k1")
    assert torch.equal(torch.compile(h, fullgraph=True)(jac, oi, ii), h(jac, oi, ii))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_generic_path_cuda_matches_cpu():
    p = "batch seq [batch seq 0] [batch seq 1]"
    lp, lab = _rn(4, 5, 6, 6), _ri(6, (4, 5, 2))
    cpu = eindex(lp, lab, p)
    assert torch.equal(eindex(lp.cuda(), lab.cuda(), p).cpu(), cpu)
    assert torch.equal(eindex(lp, lab, p), cpu)  # cpu plan still intact after the cuda call


def test_error_message_shows_inferred_sizes_like_original():
    # the original's message: "... inferred dimension sizes are 'batch=32 seq=5 [batch=33 seq=5]'"
    lp, lab = _rn(32, 5, 100), _ri(100, (33, 5))
    with pytest.raises(EindexError, match=r"'batch=32 seq=5 \[batch=33 seq=5\]'"):
        eindex(lp, lab, "batch seq [batch seq]")
    with pytest.raises(AssertionError, match=r"'batch=32 seq=5 \[batch=33 seq=5\]'"):
        ref_eindex(lp, lab, "batch seq [batch seq]")


def test_one_tensor_shared_across_brackets_like_original():
    lp, lab = _rn(8, 5, 9, 9), _ri(9, (8, 5))
    p = "batch seq [batch seq] [batch seq]"
    assert _same(eindex(lp, lab, p), ref_eindex(lp, lab, p))
    assert _same(eindex(lp, lab, p), eindex(lp, lab, lab, p))


def test_whitespace_in_pattern():
    lp, lab = _rn(8, 5, 9), _ri(9, (8, 5))
    ref = ref_eindex(lp, lab, "batch seq [batch seq]")
    patterns = [
        "batch  seq [batch seq]",
        "batch\tseq [batch\tseq]",
        "  batch seq [batch seq]  ",
        "batch seq [ batch seq ]",
    ]
    for p in patterns:
        assert _same(eindex(lp, lab, p), ref), repr(p)
