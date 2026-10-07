"""multigpu.py (formerly sharded.py) -- run a vmapped batch across all local GPUs, with bounded per-GPU memory.

The modern replacement for the jax.pmap voxel-sharding path (the old
`fitting.fit_batch_sharded`). Uses jax.jit + NamedSharding (GSPMD): the leading BATCH
axis of every input is sharded across devices and XLA partitions the already-vmapped function
automatically -- no separate per-device function, no manual replication of the constants.

WHY NOT pmap: pmap fixes per-device batch = total / n_dev and then OOMs when that quotient is
too large. The fitting sharding benchmark died exactly this way -- at 1 GPU it tried to fit all
65 536 voxels in one shot (23.8 GiB requested, 24 GiB card). Here an OUTER python loop feeds the
batch in super-chunks of `n_dev * per_device_chunk`, so each GPU only ever holds
`per_device_chunk` items at once: peak memory is set by that knob, INDEPENDENT of total batch
size. That is the property the pmap version lacked.

Result is bit-for-bit what `vmapped_fn(*batched)` produces on a single device -- sharding only
partitions an already-independent vmap, and the padding rows added to make the axis divisible by
n_dev are dropped again. Verify with `benchmark_sharding.py --validate`
(or on a GPU-less box: `XLA_FLAGS=--xla_force_host_platform_device_count=4 ... --validate`).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def pick_devices(n_devices=None):
    """The local devices to shard over: all of them, or the first `n_devices`."""
    devs = jax.local_devices()
    return devs if not n_devices else devs[:min(n_devices, len(devs))]


def _n_lead(tree):
    return jax.tree_util.tree_leaves(tree)[0].shape[0]


def _slice0(tree, sl):
    return jax.tree.map(lambda a: a[sl], tree)


def _pad0(tree, pad):                       # repeat the last row `pad` times along axis 0
    return jax.tree.map(lambda a: jnp.concatenate([a, jnp.repeat(a[-1:], pad, axis=0)], axis=0), tree)


def sharded_vmap(vmapped_fn, *batched, n_devices=None, per_device_chunk=1024):
    """Apply an already-vmapped `vmapped_fn(*batched)` across GPUs, chunked for memory.

    vmapped_fn : a function already `jax.vmap`-ed over the leading axis of its arguments
                 (so it maps a whole batch -> a batched result). It may close over any
                 replicated constants (events, spec, M0, bounds); jit bakes those in.
    batched    : one or more pytrees whose every leaf shares the SAME leading (batch) axis N.
                 That axis is the one sharded across devices / looped over in super-chunks.
    per_device_chunk : max items held on ONE device at once -> the memory knob. Total batch
                 size no longer bounds peak memory (this is the whole point vs pmap).

    returns : np.ndarray stacking the per-item results, identical to running vmapped_fn on a
              single device. (Single array output only -- the LM fit returns one (N, P) array.)
    """
    devs = pick_devices(n_devices)
    n_dev = len(devs)
    N = _n_lead(batched[0])
    jfn = jax.jit(vmapped_fn)                # compiled once; XLA caches per input shape

    if n_dev > 1:
        mesh = Mesh(np.array(devs), ("b",))
        shard = NamedSharding(mesh, P("b"))
        group = n_dev * per_device_chunk    # items per outer step (per_device_chunk each)
        align = n_dev                       # batch axis must divide n_dev to shard evenly
    else:
        shard, group, align = None, per_device_chunk, 1   # 1 device: just chunk (bounds memory)

    outs = []
    for s in range(0, N, group):
        piece = [_slice0(a, slice(s, min(s + group, N))) for a in batched]
        pad = (-_n_lead(piece[0])) % align  # pad the last, ragged super-chunk up to a n_dev multiple
        if pad:
            piece = [_pad0(a, pad) for a in piece]
        if shard is not None:
            piece = [jax.device_put(a, shard) for a in piece]
        out = np.asarray(jfn(*piece))
        outs.append(out[:-pad] if pad else out)
    return np.concatenate(outs, axis=0)
