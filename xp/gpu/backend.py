"""Optional CUDA acceleration (via CuPy) for the bulk per-page array work in
convert.py: sampling every contour's pixel colors and finding each
contour's dominant color, batched across an entire page's contours at once.

Falls back to plain numpy transparently when CuPy or a working CUDA GPU
isn't available. Everything here is written once against an `xp` module
alias (numpy or cupy) since CuPy's array API mirrors numpy's closely enough
that the same code runs on either.

Why batching matters more than "GPU or not": profiling convert.py showed
the real bottleneck was never the OpenCV image ops (thresholding, contour
finding) -- it was thousands of tiny, independent per-contour Python/numpy
calls. A loop that launches one tiny GPU kernel per contour would be
*slower* than the CPU loop it replaces (kernel-launch latency dominates
work that small). The GPU only pays off once that per-contour work is
reshaped into a handful of large array operations across the whole page's
contours at once -- which is what `batched_dominant_keys` does.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

GPU_AVAILABLE = False
xp = np


def _try_enable_gpu() -> None:
    global GPU_AVAILABLE, xp
    try:
        if "CUDA_PATH" not in os.environ:
            import nvidia.cuda_runtime as _rt  # pip: nvidia-cuda-runtime-cu12

            # This is a namespace package (no __init__.py), so __file__ is
            # None; use __path__ instead.
            os.environ["CUDA_PATH"] = str(Path(list(_rt.__path__)[0]))
        import cupy as cp

        cp.cuda.Device(0).compute_capability  # forces an actual device query
        xp = cp
        GPU_AVAILABLE = True
    except Exception:
        xp = np
        GPU_AVAILABLE = False


_try_enable_gpu()


def room(side=None) -> int:
    """How many bytes a working array may use right now, in bytes.

    The user's card is a 12 GiB laptop part and a page is 173 megapixels, so "it
    fits" is not a property of the algorithm -- it is a property of the page
    in front of it. A spatial search over a thick pen on a dense page asked
    for 5.02 GiB in one allocation and the conversion died with cupy's
    `OutOfMemoryError`, which is not a thing a user should ever be shown.

    So the size of a working array is **measured**, not assumed: what the
    device says is free, less a margin for everything the step still has to
    allocate beside it. On the host this is deliberately generous -- system
    memory is not the thing that was running out -- but still finite, because
    the same search on the host would have asked for the same 5 GiB.
    """
    if side is not None and side is np:
        return 2 << 30
    if not GPU_AVAILABLE:
        return 2 << 30
    try:
        free, _total = xp.cuda.runtime.memGetInfo()
        # Half of what is free. The other half is for the arrays the caller
        # already holds a reference to and the temporaries it makes while it
        # works on this one; a budget that spends everything free is a budget
        # that fails on the next line instead of this one.
        return max(1 << 24, int(free) // 2)
    except Exception:
        return 1 << 28


def to_device(arr):
    return xp.asarray(arr)


def to_host(arr):
    return xp.asnumpy(arr) if GPU_AVAILABLE else np.asarray(arr)


def batched_dominant_keys(keys, group_ids, num_groups: int):
    """`keys` and `group_ids` are equal-length 1-D int arrays on the active
    `xp` backend. Returns a length-`num_groups` int64 array: the most
    frequent `keys[i]` among rows sharing each `group_ids[i]`, or -1 for a
    group with zero rows.

    Implemented as two sort-based grouping passes (no per-group Python
    loop, and no bincount over a wide/sparse value range -- `run_id` below
    is a dense 0..num_runs-1 index by construction, so bincount on it is
    cheap and safe): first group identical (group_id, key) pairs into
    contiguous "runs" and count them, then take a segmented argmax over
    those run counts per group_id.
    """
    n = int(keys.shape[0])
    result = xp.full(num_groups, -1, dtype=xp.int64)
    if n == 0:
        return result

    order = xp.lexsort(xp.stack([keys, group_ids]))  # primary: group_ids
    sorted_gid = group_ids[order]
    sorted_key = keys[order]

    is_new_run = xp.empty(n, dtype=bool)
    is_new_run[0] = True
    is_new_run[1:] = (sorted_gid[1:] != sorted_gid[:-1]) | (sorted_key[1:] != sorted_key[:-1])
    run_id = xp.cumsum(is_new_run) - 1
    num_runs = int(run_id[-1]) + 1

    run_counts = xp.bincount(run_id, minlength=num_runs)
    run_gid = sorted_gid[is_new_run]
    run_key = sorted_key[is_new_run]

    # Segmented argmax: per group_id, the run with the highest count. Sort
    # runs by (group_id primary, -count secondary) so each group's winning
    # run is the first one encountered for that group_id.
    order2 = xp.lexsort(xp.stack([-run_counts, run_gid]))  # primary: run_gid
    sorted_run_gid2 = run_gid[order2]
    is_first_in_gid = xp.empty(len(sorted_run_gid2), dtype=bool)
    is_first_in_gid[0] = True
    is_first_in_gid[1:] = sorted_run_gid2[1:] != sorted_run_gid2[:-1]

    winning_gids = sorted_run_gid2[is_first_in_gid]
    winning_keys = run_key[order2][is_first_in_gid]
    result[winning_gids] = winning_keys
    return result
