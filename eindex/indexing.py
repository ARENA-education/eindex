"""Fast, compile-once `eindex` (einops-style tensor indexing).

Drop-in for the original `eindex` (https://www.perfectlynormal.co.uk/blog-eindex): same signature,
same pattern grammar, same results -- but the pattern is parsed **once** into a closure of native
torch ops, so calling it in a hot loop costs the same as a hand-written `torch.gather`.

    from eindex import eindex, compile_eindex

    out = eindex(logprobs, labels, "batch seq [batch seq]")   # drop-in; pattern compiled once + cached
    pick = compile_eindex("batch [batch]")                    # or compile explicitly for a hot loop
    out = pick(arr, idx)

Supported grammar (a superset of the original's):
  - bare axes (kept in the output)       "batch"
  - bracketed indexed axes               "[batch seq]"          (each bracket consumes one index tensor)
  - multiple index tensors               "... [batch seq] [batch seq]"   (1:1 with the brackets)
  - single tensor, integer-slot brackets "... [batch seq 0] [batch seq 1]"
  - offsets                              "[batch seq+1]"        (autoregressive; shrinks that axis)
  - "-> ..." output reorder
  - numpy arrays in / out (converted at the boundary)
  - repeated bare axes -> diagonal, e.g. "b s [b s k2] b s [b s k1] -> b s k2 k1" on a (b,s,f,b,s,f)
    jacobian gives out[b,s,k2,k1] = jac[b,s,oi[b,s,k2],b,s,ii[b,s,k1]]. The original raised on this
    (github.com/callummcdougall/eindex/issues/4).
Whether brackets share one index tensor (integer-slot case) or map one-each (multi-tensor case) is
decided by the number of index tensors passed -- exactly as the original does.

Validation: shape/pattern mismatches raise `EindexError` (a `ValueError` *and* an `AssertionError`, for
compatibility with the original) with a message in the spirit of the original. It uses only Python-int
comparisons (no `.item()` device syncs), so the closures stay `torch.compile(fullgraph=True)`-clean and
the check runs once per distinct shape signature (memoised), so a fixed-shape loop pays ~0.5 us per call.

Why the original is ~30-50x slower (profiled on CPU, "batch [batch]", B=4096): not the regex parse
(~3 us) but the `torch.tensor(shape).prod().item()` device-sync asserts, unconditional error-string
building per axis, and indexing with a Python *list* (`arr[full_idx]` -- torch's slow, deprecated
non-tuple path: ~450 us vs ~30 us for `gather`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor

Array = Union[Tensor, np.ndarray]


class EindexError(ValueError, AssertionError):
    """Raised on a pattern / shape mismatch. Subclasses `AssertionError` too, because the original
    `eindex` raised `AssertionError` and existing code (e.g. `demo.ipynb`) catches that."""


# ---------------------------------------------------------------------------------------------------
# Pattern parsing (done once per pattern, in `compile_eindex`)
# ---------------------------------------------------------------------------------------------------


def _split_axes(lhs: str) -> list[str]:
    """'batch seq [batch seq]' -> ['batch', 'seq', '[batch seq]'] (bracket-aware split on spaces)."""
    parts, buf, depth = [], "", 0
    for ch in lhs.strip():
        if ch == "[":
            depth += 1
            buf += ch
        elif ch == "]":
            depth -= 1
            buf += ch
        elif ch == " " and depth == 0:
            if buf:
                parts.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        parts.append(buf)
    if depth != 0:
        raise EindexError(f"Unbalanced brackets in pattern {lhs!r}")
    return parts


def _parse_entry(tok: str) -> tuple[str, int, bool]:
    """'seq' -> ('seq', 0, False); 'seq+1' -> ('seq', 1, False); '0' -> ('0', 0, True) (integer slot)."""
    if tok.isdigit():
        return (tok, 0, True)
    name, _, off = tok.partition("+")
    if not name or (off and not off.isdigit()):
        raise EindexError(f"Bad axis token {tok!r} (expected 'name', 'name+k' or an integer slot)")
    return (name, int(off) if off else 0, False)


def _np_wrap(run: Callable) -> Callable:
    """Allow numpy arrays in/out (convert at the boundary), like the original."""

    def f(arr: Array, *idx: Array) -> Array:
        np_in = isinstance(arr, np.ndarray)
        a = torch.as_tensor(arr) if np_in else arr
        ii = [torch.as_tensor(x) if isinstance(x, np.ndarray) else x for x in idx]
        out = run(a, *ii)
        return out.numpy() if np_in else out

    return f


# ---------------------------------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------------------------------


def compile_eindex(pattern: str, verbose: bool = False, validate: bool = True) -> Callable:
    """Parse `pattern` once; return `f(arr, *index_tensors)` that indexes `arr` with no re-parsing.

    verbose=True prints the inferred per-axis sizes and output shape on each call (like the original's
    `verbose`); it bypasses the gather fast path so the sizes are available to print.
    validate=False skips the (cheap, shape-only) argument checks; errors then surface as raw torch
    indexing errors.
    """
    lhs, arrow, rhs = pattern.partition("->")
    if arrow and not rhs.strip():
        raise EindexError(f"Pattern {pattern!r} has '->' but nothing after it")

    # per arr-axis token: ("bare", name, offset) | ("idx", [(name, offset, is_digit), ...], bracket_i)
    axis_tokens: list[tuple] = []
    n_brackets = 0
    for part in _split_axes(lhs):
        if part.startswith("["):
            entries = [_parse_entry(t) for t in part[1:-1].split()]
            if not entries:
                raise EindexError(f"Empty brackets in pattern {pattern!r}")
            axis_tokens.append(("idx", entries, n_brackets))
            n_brackets += 1
        else:
            name, offset, _ = _parse_entry(part)
            axis_tokens.append(("bare", name, offset))
    if n_brackets == 0:
        raise EindexError(f"Pattern {pattern!r} has no [indexed] axes -- nothing to index")
    n_arr_axes = len(axis_tokens)

    def _entries(tok):
        return [(tok[1], tok[2], False)] if tok[0] == "bare" else tok[1]

    # max offset per name -> how much that output axis shrinks
    offset_size: dict[str, int] = {}
    for tok in axis_tokens:
        for name, off, is_digit in _entries(tok):
            if not is_digit:
                offset_size[name] = max(offset_size.get(name, 0), off)

    inferred_axes: list[str] = []
    for tok in axis_tokens:
        for name, _off, is_digit in _entries(tok):
            if not is_digit and name not in inferred_axes:
                inferred_axes.append(name)
    if rhs.strip():
        out_axes = rhs.split()
        if sorted(out_axes) != sorted(inferred_axes):
            raise EindexError(
                "The dimensions after '->' don't match the inferred output dimensions of your indexing operation."
                f"\nInferred output dimensions: {inferred_axes}\nYour '->' dimensions: {out_axes}"
            )
    else:
        out_axes = inferred_axes
    out_pos = {nm: k for k, nm in enumerate(out_axes)}
    nout = len(out_axes)
    has_offset = any(v > 0 for v in offset_size.values())
    has_digit = any(tok[0] == "idx" and any(e[2] for e in tok[1]) for tok in axis_tokens)

    idx_axes = [ax for ax, tok in enumerate(axis_tokens) if tok[0] == "idx"]
    bare_names = [tok[1] for tok in axis_tokens if tok[0] == "bare"]

    # (bracket_i, entry_pos, name, is_digit) for every bracket entry: used by the shape checks
    bracket_shapes = [(tok[2], len(tok[1])) for tok in axis_tokens if tok[0] == "idx"]

    validated: set = set()  # shape signatures already checked: a loop with fixed shapes validates once

    def _check(arr: Tensor, idx_tensors: tuple) -> None:
        """Shape-only validation (Python ints; no device syncs), memoised on the shapes involved."""
        if len(idx_tensors) == 1:
            key = (arr.shape, idx_tensors[0].shape)
        else:
            key = (arr.shape, *[t.shape for t in idx_tensors])
        if key in validated:
            return
        _check_full(arr, idx_tensors)
        if len(validated) > 64:  # unbounded growth guard (e.g. a new shape every call)
            validated.clear()
        validated.add(key)

    def _check_full(arr: Tensor, idx_tensors: tuple) -> None:
        if arr.ndim != n_arr_axes:
            raise EindexError(
                f"Invalid indices: pattern {pattern!r} has {n_arr_axes} terms but the array to index into has "
                f"{arr.ndim} dimensions (shape {tuple(arr.shape)})."
            )
        n_idx = len(idx_tensors)
        if n_idx == 0:
            raise EindexError("You need to pass at least one index tensor.")
        if n_idx != 1 and n_idx != n_brackets:
            raise EindexError(
                f"Pattern {pattern!r} has {n_brackets} bracketed groups but you passed {n_idx} index tensors "
                "(pass one tensor per group, or a single shared tensor with integer slots)."
            )
        if n_idx == 1 and n_brackets > 1 and not has_digit:
            raise EindexError(
                f"Pattern {pattern!r} has {n_brackets} bracketed groups but only one index tensor was passed; "
                "either pass one tensor per group, or use integer slots like '[batch seq 0] [batch seq 1]'."
            )
        multi = n_idx > 1
        for bi, n_entries in bracket_shapes:
            t = idx_tensors[bi if multi else 0]
            if t.ndim != n_entries:
                raise EindexError(
                    f"Bracket #{bi} in pattern {pattern!r} has {n_entries} terms but the corresponding index "
                    f"tensor has {t.ndim} dimensions (shape {tuple(t.shape)})."
                )
        size: dict[str, int] = {}
        for ax, tok in enumerate(axis_tokens):
            if tok[0] == "bare":
                pairs = ((tok[1], arr.shape[ax]),)
            else:
                t = idx_tensors[tok[2] if multi else 0]
                pairs = tuple((e[0], t.shape[p]) for p, e in enumerate(tok[1]) if not e[2])
            for name, n in pairs:
                prev = size.setdefault(name, n)
                if prev != n:
                    raise EindexError(
                        f"Contradictory sizes for dimension {name!r} in pattern {pattern!r}: got {prev} and {n} "
                        f"(array shape {tuple(arr.shape)}, index shapes {[tuple(t.shape) for t in idx_tensors]})."
                    )
        if has_digit:
            t = idx_tensors[0]
            for tok in axis_tokens:
                if tok[0] == "idx":
                    for p, (name, _o, is_digit) in enumerate(tok[1]):
                        if is_digit and int(name) >= t.shape[p]:
                            raise EindexError(
                                f"Integer slot {name} in pattern {pattern!r} is out of range for index tensor axis "
                                f"{p} of size {t.shape[p]}."
                            )

    # Fast path: one bracket, no integer slots, no verbose, and both the bare axes and the bracket's
    # names are exactly the output axes -> a plain torch.gather along the bracket axis. Offsets in the
    # bracket ("[batch seq+1]") are handled by slicing views of `arr` / `idx` first, so this still
    # costs one gather (this is the next-token-logprob pattern, which sits in training loops).
    bare_offsets_zero = all(tok[2] == 0 for tok in axis_tokens if tok[0] == "bare")
    if (
        n_brackets == 1
        and not has_digit
        and not verbose
        and bare_offsets_zero
        and bare_names == out_axes
        and [e[0] for e in axis_tokens[idx_axes[0]][1]] == out_axes
    ):
        gather_dim = idx_axes[0]
        # arr: shrink each bare axis whose name carries an offset (slice 0 : size-m); idx: slice each
        # axis `off : off-m` (None when off == m). Built once here; `slice(0, None)` is a no-op view.
        arr_sl = tuple(
            slice(0, -offset_size[tok[1]] if tok[0] == "bare" and offset_size.get(tok[1], 0) else None)
            for tok in axis_tokens
        )
        idx_sl = tuple(
            slice(off, (off - offset_size[name]) if off != offset_size[name] else None)
            for name, off, _d in axis_tokens[gather_dim][1]
        )
        if not has_offset:
            arr_sl = idx_sl = None

        def _gather(arr: Tensor, idx: Tensor) -> Tensor:
            if arr_sl is not None:
                arr, idx = arr[arr_sl], idx[idx_sl]
            return arr.gather(gather_dim, idx.unsqueeze(gather_dim)).squeeze(gather_dim)

        if validate:

            def run(arr: Tensor, *idx: Tensor) -> Tensor:
                _check(arr, idx)
                return _gather(arr, idx[0])

        else:

            def run(arr: Tensor, idx: Tensor) -> Tensor:
                return _gather(arr, idx)

        return _np_wrap(run)

    def run(arr: Tensor, *idx_tensors: Tensor) -> Tensor:
        if validate:
            _check(arr, idx_tensors)
        multi = len(idx_tensors) > 1  # one tensor per bracket vs one shared tensor with integer slots
        size: dict[str, int] = {}
        for ax, tok in enumerate(axis_tokens):
            if tok[0] == "bare":
                size[tok[1]] = arr.shape[ax]
            else:
                t = idx_tensors[tok[2] if multi else 0]
                for pos, (name, _off, is_digit) in enumerate(tok[1]):
                    if not is_digit:
                        size[name] = t.shape[pos]
        true = {nm: size[nm] - offset_size.get(nm, 0) for nm in out_axes}
        if verbose:
            print(
                f"eindex {pattern!r}: sizes "
                + ", ".join(f"{nm}={size[nm]}" for nm in out_axes)
                + " | output shape "
                + str(tuple(true[nm] for nm in out_axes))
            )
        index_arrays = []
        for ax, tok in enumerate(axis_tokens):
            if tok[0] == "bare":
                p = out_pos[tok[1]]
                shp = [true[tok[1]] if j == p else 1 for j in range(nout)]
                index_arrays.append(torch.arange(true[tok[1]], device=arr.device).reshape(shp))
            else:
                t = idx_tensors[tok[2] if multi else 0]
                sl: list = []  # slice / int per index-tensor axis
                for name, off, is_digit in tok[1]:
                    if is_digit:
                        sl.append(int(name))
                    else:
                        os_ = offset_size.get(name, 0)
                        sl.append(slice(off, (off - os_) if off != os_ else None))
                sub = t[tuple(sl)]  # axes = the bracket's non-digit names, in order
                names = [e[0] for e in tok[1] if not e[2]]
                perm = sorted(range(len(names)), key=lambda k: out_pos[names[k]])
                pos2size = {out_pos[nm]: true[nm] for nm in names}
                shp = [pos2size.get(j, 1) for j in range(nout)]
                index_arrays.append(sub.permute(perm).reshape(shp))
        return arr[tuple(index_arrays)]

    return _np_wrap(run)


@lru_cache(maxsize=None)
def _compiled(pattern: str) -> Callable:
    return compile_eindex(pattern)


def eindex(*tensors_and_pattern, **kwargs) -> Array:
    """`eindex(arr, *index_tensors, pattern, verbose=False)` -- einops-style indexing.

    Same call signature as the original `eindex`. The pattern is compiled once (see `compile_eindex`)
    and cached, so repeated calls with the same pattern pay no parse cost.

    Example: `eindex(logprobs, labels, "batch seq [batch seq]")` is
             `output[b, s] = logprobs[b, s, labels[b, s]]`.
    """
    verbose = kwargs.pop("verbose", False)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
    if len(tensors_and_pattern) < 3:
        raise TypeError("eindex needs at least an array, one index tensor, and a pattern string.")
    *tensors, pattern = tensors_and_pattern
    if not isinstance(pattern, str):
        raise TypeError("Last argument must be the pattern string.")
    arr, *index_tensors = tensors
    fn = compile_eindex(pattern, verbose=True) if verbose else _compiled(pattern)
    return fn(arr, *index_tensors)
