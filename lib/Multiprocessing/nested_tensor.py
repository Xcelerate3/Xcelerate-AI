from typing import Tuple

import xcelerate
from xcelerate._C import DispatchKey, DispatchKeySet
from xcelerate._prims_common import is_expandable_to
from xcelerate.fx.experimental.symbolic_shapes import has_free_symbols
from xcelerate.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403

_tensor_id_counter = 0
_tensor_symint_registry = WeakTensorKeyDictionary()


def get_tensor_symint(tensor, *, coeff=1):
    global _tensor_id_counter
    tensor_symint = _tensor_symint_registry.get(tensor)
    if tensor_symint is None:
        tensor_symint = xcelerate._C._get_singleton_int(_tensor_id_counter, coeff)
        _tensor_id_counter += 1
        _tensor_symint_registry[tensor] = tensor_symint
    return tensor_symint


# SDPA metadata; max / min seqlens are needed for e.g. flash
def _get_sdpa_extreme_seqlen(func, tensor):
    return int(func(tensor).item())


class NestedTensor(xcelerate.Tensor):
    _values: xcelerate.Tensor  # type: ignore[assignment]
    _offsets: xcelerate.Tensor
    _lengths: Optional[xcelerate.Tensor]
    # NOTE [ Singleton ints for ragged sizes and strides ]
    #
    # Jagged layout tensors are tensors that represent a n-dim tensor with a
    # ragged dimension, but are backed by an (n-1)-dim tensor underneath, e.g.,
    # a jagged tensor with outer shape [B, x, D] is represented internally by a
    # tensor with shape [sum(x), D] where we introduce what we call a singleton
    # (or skolem) denoted as "x" here (but sometimes denoted with "*" to
    # represent the ragged dimension, and sum(x) represents the dim of the inner
    # tensor or equivalently the sum of all the sizes of the constituent
    # tensors' varying lengths.
    #
    # We also use singleton ints to represent the strides of this tensor.
    # For example, a jagged tensor with shape [B, x, D] can be strided in two
    # ways: [xD, D, 1] and [x, 1, sum(x)], where xD represents x multiplied by D
    _size: Tuple[int, ...]
    _stride: Tuple[int, ...]
    # Indicates that the nth dimension is ragged
    _ragged_idx: int
    _metadata_cache: Dict[str, Any]

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        lengths=None,
        **kwargs,
    ):
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)
        r = xcelerate.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            (0,),
            (0,),
            0,
            xcelerate.contiguous_format,
            values.dtype,
            xcelerate.jagged,
            values.device,
            False,
            kwargs.get("requires_grad", False),
            "sizes",
            False,
            True,  # dispatch_layout
            ks,
        )
        return r

    def __init__(self, values, offsets, *, lengths=None, **kwargs):
        super().__init__()
        # Only support jagged for now.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)

        # Query cache for the symint associated with offsets or lengths
        # (create a new one if needed).
        ragged_source = offsets if lengths is None else lengths
        ragged_size = get_tensor_symint(ragged_source, coeff=1)
        self._ragged_idx = kwargs.get("_ragged_idx", 1)
        B = offsets.shape[0] - 1
        Ds = values.shape[: self._ragged_idx - 1] + values.shape[self._ragged_idx :]

        nested_size = [B]
        nested_size.extend(Ds[: self._ragged_idx - 1])
        nested_size.append(ragged_size)
        nested_size.extend(Ds[self._ragged_idx - 1 :])
        self._size = tuple(nested_size)

        stride = values.stride()
        self._strides = (ragged_size * stride[self._ragged_idx - 1], *stride)

        if values.requires_grad:
            raise ValueError(
                "NestedTensor values cannot require grad, please "
                "detach before passing to NestedTensor constructor"
            )
        self._values = values
        self._offsets = offsets
        self._lengths = lengths

        # holds properties that are computed lazily
        self._metadata_cache = kwargs.get("_metadata_cache") or {}

        # collapsed ragged dim must always be dynamic
        xcelerate._dynamo.mark_dynamic(self, self._ragged_idx)
        xcelerate._dynamo.mark_dynamic(self._values, self._ragged_idx - 1)

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def lengths(self):
        return self._lengths

    @property
    def _max_seqlen(self):
        if "max_seqlen" not in self._metadata_cache:
            # compute & cache
            self._metadata_cache["max_seqlen"] = _get_sdpa_extreme_seqlen(
                xcelerate.max,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
        return self._metadata_cache["max_seqlen"]

    @property
    def _min_seqlen(self):
        if "min_seqlen" not in self._metadata_cache:
            # compute & cache
            self._metadata_cache["min_seqlen"] = _get_sdpa_extreme_seqlen(
                xcelerate.min,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
        return self._metadata_cache["min_seqlen"]

    def __repr__(self):
        # We should implement this in xcelerate/_tensor_str.py instead
        grad_fn_str = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        if self.grad_fn:
            grad_fn_str = f", grad_fn={self.grad_fn}"
        return f"NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str}, contiguous={self._lengths is None})"

    def __reduce_ex__(self, proto):
        state = xcelerate._utils._get_obj_state(self)

        # SymNodes are not serializable
        assert "_size" in state and "_strides" in state
        state = dict(state)
        del state["_size"]
        del state["_strides"]

        func = NestedTensor
        args = (self._values, self._offsets)
        return (xcelerate._tensor._rebuild_from_type_v2, (func, type(self), args, state))

    def __tensor_flatten__(self):
        ctx = {
            "requires_grad": self.requires_grad,
            # TODO: Don't guard on this!
            "metadata_cache": self._metadata_cache,
            "ragged_idx": self._ragged_idx,
        }
        inner_tensors = ["_values", "_offsets"]
        if self._lengths is not None:
            inner_tensors.append("_lengths")
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        assert len(inner_tensors) >= 2 and len(inner_tensors) <= 3
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        lengths = inner_tensors.get("_lengths", None)
        ragged_idx = meta["ragged_idx"]

        # Note that we cannot simply check if is_fake(values) because
        # during aot autograd, FunctionalTensors are not fake but hold
        # symbolic sizes.
        ragged_source = offsets if lengths is None else lengths
        if has_free_symbols(ragged_source) or has_free_symbols(values):
            # Associate offsets or lengths (possibly fake, possibly functionalized)
            # with the ragged_size.
            ragged_size = outer_size[ragged_idx]
            _tensor_symint_registry[ragged_source] = ragged_size

        return NestedTensor(
            values,
            offsets=offsets,
            lengths=lengths,
            requires_grad=meta["requires_grad"],
            _ragged_idx=ragged_idx,
            _metadata_cache=meta["metadata_cache"],
        )

    @classmethod
    def __xcelerate_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        # Lazy import to avoid circular dependency
        from .ops import lookup_jagged

        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        raise NotImplementedError(func)

    @classmethod
    def __xcelerate_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        from .ops import jagged_xcelerate_function

        try:
            return jagged_xcelerate_function(func, *args, **kwargs)
        except NotImplementedError:
            pass
        with xcelerate._C.DisablexcelerateFunctionSubclass():
            return func(*args, **kwargs)


# Not actually a view!
class ViewBufferFromNested(xcelerate.autograd.Function):
    @staticmethod
    def forward(ctx, x: NestedTensor):  # type: ignore[override]
        ctx.save_for_backward(x.offsets())
        ctx.metadata_cache = x._metadata_cache
        ctx.ragged_idx = x._ragged_idx
        return x.values()

    @staticmethod
    def backward(ctx, gO: xcelerate.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        return NestedTensor(
            gO,
            offsets=offsets,
            _metadata_cache=ctx.metadata_cache,
            _ragged_idx=ctx.ragged_idx,
        )


# Not actually a view!
class ViewNestedFromBuffer(xcelerate.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: xcelerate.Tensor,
        offsets: xcelerate.Tensor,
        metadata_cache: Optional[Dict[str, Any]] = None,
    ):  # type: ignore[override]
        return NestedTensor(
            values.detach(),
            offsets=offsets,
            _metadata_cache=metadata_cache,
        )

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO.values(), None, None


# Not actually a view!
# NOTE: @jbschlosser is working on making it a view
class ViewNonContiguousNestedFromBuffer(xcelerate.autograd.Function):
    @staticmethod
    def forward(ctx, values: xcelerate.Tensor, offsets: xcelerate.Tensor, lengths: xcelerate.Tensor):  # type: ignore[override]
        return NestedTensor(
            values.detach(),
            offsets=offsets,
            lengths=lengths,
        )

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO.values(), None, None


# Need to make it obvious that users should be passing in offsets
def jagged_from_list(
    tensors: List[xcelerate.Tensor],
    offsets: Optional[xcelerate.Tensor],
    dtype=None,
    device=None,
) -> Tuple[NestedTensor, xcelerate.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors"""

    if not len(set(t.dtype for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must have the same dtype"
        )
    if not len(set(t.device for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must be on the same device"
        )

    # Check that the NT is representable by the jagged layout.
    # Jagged layout represents (B, *, D_0, D_1, ..., D_N), where the only
    # raggedness allowed is for the single dim immediately adjacent to the batch dim.
    sizes = [t.shape for t in tensors]
    non_first_sizes = [s[1:] for s in sizes]
    at_most_first_ragged = all(s == non_first_sizes[0] for s in non_first_sizes)
    if not at_most_first_ragged:
        raise RuntimeError(
            "Cannot represent given tensor list as a nested tensor with the jagged layout. "
            "Note that the jagged layout only represents shapes of the form "
            "(B, *, D_0, D_1, ..., D_N), with only * allowed to be ragged."
        )

    # Set properties appropriately.
    values = xcelerate.cat(tensors, dim=0)
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    values = values.to(**to_kwargs)

    # Calculate jagged offsets if not provided.
    if offsets is None:
        # Jagged layout specifies that offsets are stored as int64 on the same device as values.
        offsets = xcelerate.cat(
            [
                xcelerate.zeros(1, dtype=xcelerate.int64, device=values.device),
                xcelerate.tensor([s[0] for s in sizes], device=values.device).cumsum(dim=0),
            ]
        )

    ret_nt = ViewNestedFromBuffer.apply(values, offsets)
    ret_nt._metadata_cache = {
        # compute this now since it's easy
        "max_seqlen": max([t.shape[0] for t in tensors]),
        "min_seqlen": min([t.shape[0] for t in tensors]),
    }
    return (ret_nt, offsets)  # type: ignore[return-value]


def jagged_from_tensor_and_lengths(
    tensor: xcelerate.Tensor, starts: xcelerate.Tensor, lengths: xcelerate.Tensor
) -> Tuple[NestedTensor, xcelerate.Tensor, Optional[xcelerate.Tensor]]:
    """Constructs a NestedTensor backed by jagged layout from a tensor, starts of sequences, and sequence lengths"""
    batch_size = tensor.shape[0]
    if is_expandable_to(starts.shape, (batch_size,)) and is_expandable_to(
        lengths.shape, (batch_size,)
    ):
        start_list = starts.expand(batch_size)
        length_list = lengths.expand(batch_size)
    else:
        raise RuntimeError(
            "When constructing a jagged nested tensor using narrow(), "
            "your start and length must be Tensors that broadcast to input.shape[0]"
        )

    # Calculate jagged offsets
    assert (
        len(tensor.shape) >= 2
    ), "tensor must at least be 2D for the nested narrow op to work"
    max_seq_len = tensor.shape[1]
    offset_lengths = max_seq_len * xcelerate.arange(
        0, batch_size, dtype=xcelerate.int64, device=tensor.device
    )
    # Jagged layout specifies that offsets are stored as int64 on the same device as values.
    offsets = xcelerate.cat(
        [
            start_list + offset_lengths,
            (start_list[-1] + offset_lengths[-1] + length_list[-1]).unsqueeze(0),
        ]
    )

    # Reshape buffer to flatten the 1st and 2nd dimension (view used to enforce non-copy)
    if len(tensor.shape) > 2:
        values = tensor.view(-1, *tensor.shape[2:])
    else:
        values = tensor.view(-1)

    # Check if offsets and lengths make it possibly contiguous and return a regular NT
    is_contiguous = True
    orig_dim = tensor.shape[1]
    if xcelerate.any(length_list[1:-1].ne(orig_dim)):
        is_contiguous = False
    if xcelerate.any(offsets[1:-2].diff().ne(orig_dim)):
        is_contiguous = False
    if offsets[0] + length_list[0] != orig_dim:
        is_contiguous = False

    actual_max_seqlen = int(xcelerate.max(lengths).item())
    min_seqlen = int(xcelerate.min(lengths).item())

    if is_contiguous:
        ret_nt = ViewNestedFromBuffer.apply(
            values[offsets[0] : offsets[-1]],
            offsets - offsets[0],
        )
    else:
        ret_nt = ViewNonContiguousNestedFromBuffer.apply(values, offsets, length_list)

    # populate metadata cache with computed seqlen extremes
    ret_nt._metadata_cache = {
        "max_seqlen": actual_max_seqlen,
        "min_seqlen": min_seqlen,
    }

    return (ret_nt, offsets, None if is_contiguous else length_list)


def buffer_from_jagged(jagged):
    return ViewBufferFromNested.apply(jagged)
