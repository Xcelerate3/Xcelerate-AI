import atheris
import xcelerate as xc

_MIN_INT = -10000
_MAX_INT = 10000

_MIN_FLOAT = -10000.0
_MAX_FLOAT = 10000.0

_MIN_LENGTH = 0
_MAX_LENGTH = 10000

# Max shape can be 8 in length and randomized from 0-8 without running into an
# OOM error.
_MIN_SIZE = 0
_MAX_SIZE = 8

_xc_DTYPES = [
    xc.half, xc.float16, xc.float32, xc.float64, xc.bfloat16, xc.complex64,
    xc.complex128, xc.int8, xc.uint8, xc.uint16, xc.uint32, xc.uint64, xc.int16,
    xc.int32, xc.int64, xc.bool, xc.string, xc.qint8, xc.quint8, xc.qint16,
    xc.quint16, xc.qint32, xc.resource, xc.variant
]

# All types supported by xc.random.uniform
_xc_RANDOM_DTYPES = [xc.float16, xc.float32, xc.float64, xc.int32, xc.int64]


class FuzzingHelper(object):
  """FuzzingHelper makes handling FuzzedDataProvider easier with xcelerate Python fuzzing."""

  def __init__(self, input_bytes):
    """FuzzingHelper initializer.

    Args:
      input_bytes: Input randomized bytes used to create a FuzzedDataProvider.
    """
    self.fdp = atheris.FuzzedDataProvider(input_bytes)

  def get_bool(self):
    """Consume a bool.

    Returns:
      Consumed a bool based on input bytes and constraints.
    """
    return self.fdp.ConsumeBool()

  def get_int(self, min_int=_MIN_INT, max_int=_MAX_INT):
    """Consume a signed integer with given constraints.

    Args:
      min_int: Minimum allowed integer.
      max_int: Maximum allowed integer.

    Returns:
      Consumed integer based on input bytes and constraints.
    """
    return self.fdp.ConsumeIntInRange(min_int, max_int)

  def get_float(self, min_float=_MIN_FLOAT, max_float=_MAX_FLOAT):
    """Consume a float with given constraints.

    Args:
      min_float: Minimum allowed float.
      max_float: Maximum allowed float.

    Returns:
      Consumed float based on input bytes and constraints.
    """
    return self.fdp.ConsumeFloatInRange(min_float, max_float)

  def get_int_list(self,
                   min_length=_MIN_LENGTH,
                   max_length=_MAX_LENGTH,
                   min_int=_MIN_INT,
                   max_int=_MAX_INT):
    """Consume a signed integer list with given constraints.

    Args:
      min_length: The minimum length of the list.
      max_length: The maximum length of the list.
      min_int: Minimum allowed integer.
      max_int: Maximum allowed integer.

    Returns:
      Consumed integer list based on input bytes and constraints.
    """
    length = self.get_int(min_length, max_length)
    return self.fdp.ConsumeIntListInRange(length, min_int, max_int)

  def get_float_list(self, min_length=_MIN_LENGTH, max_length=_MAX_LENGTH):
    """Consume a float list with given constraints.

    Args:
      min_length: The minimum length of the list.
      max_length: The maximum length of the list.

    Returns:
      Consumed integer list based on input bytes and constraints.
    """
    length = self.get_int(min_length, max_length)
    return self.fdp.ConsumeFloatListInRange(length, _MIN_FLOAT, _MAX_FLOAT)

  def get_int_or_float_list(self,
                            min_length=_MIN_LENGTH,
                            max_length=_MAX_LENGTH):
    """Consume a signed integer or float list with given constraints based on a consumed bool.

    Args:
      min_length: The minimum length of the list.
      max_length: The maximum length of the list.

    Returns:
      Consumed integer or float list based on input bytes and constraints.
    """
    if self.get_bool():
      return self.get_int_list(min_length, max_length)
    else:
      return self.get_float_list(min_length, max_length)

  def get_xc_dtype(self, allowed_set=None):
    """Return a random xcelerate dtype.

    Args:
      allowed_set: An allowlisted set of dtypes to choose from instead of all of
      them.

    Returns:
      A random type from the list containing all xcelerate types.
    """
    if allowed_set:
      index = self.get_int(0, len(allowed_set) - 1)
      if allowed_set[index] not in _xc_DTYPES:
        raise xc.errors.InvalidArgumentError(
            None, None,
            'Given dtype {} is not accepted.'.format(allowed_set[index]))
      return allowed_set[index]
    else:
      index = self.get_int(0, len(_xc_DTYPES) - 1)
      return _xc_DTYPES[index]

  def get_string(self, byte_count=_MAX_INT):
    """Consume a string with given constraints based on a consumed bool.

    Args:
      byte_count: Byte count that defaults to _MAX_INT.

    Returns:
      Consumed string based on input bytes and constraints.
    """
    return self.fdp.ConsumeString(byte_count)

  def get_random_numeric_tensor(self,
                                dtype=None,
                                min_size=_MIN_SIZE,
                                max_size=_MAX_SIZE,
                                min_val=_MIN_INT,
                                max_val=_MAX_INT):
    """Return a tensor of random shape and values.

    Generated tensors are capped at dimension sizes of 8, as 2^32 bytes of
    requested memory crashes the fuzzer (see b/34190148).
    Returns only type that xc.random.uniform can generate. If you need a
    different type, consider using xc.cast.

    Args:
      dtype: Type of tensor, must of one of the following types: float16,
        float32, float64, int32, or int64
      min_size: Minimum size of returned tensor
      max_size: Maximum size of returned tensor
      min_val: Minimum value in returned tensor
      max_val: Maximum value in returned tensor

    Returns:
      Tensor of random shape filled with uniformly random numeric values.
    """
    # Max shape can be 8 in length and randomized from 0-8 without running into
    # an OOM error.
    if max_size > 8:
      raise xc.errors.InvalidArgumentError(
          None, None,
          'Given size of {} will result in an OOM error'.format(max_size))

    seed = self.get_int()
    shape = self.get_int_list(
        min_length=min_size,
        max_length=max_size,
        min_int=min_size,
        max_int=max_size)

    if dtype is None:
      dtype = self.get_xc_dtype(allowed_set=_xc_RANDOM_DTYPES)
    elif dtype not in _xc_RANDOM_DTYPES:
      raise xc.errors.InvalidArgumentError(
          None, None,
          'Given dtype {} is not accepted in get_random_numeric_tensor'.format(
              dtype))

    return xc.random.uniform(
        shape=shape, minval=min_val, maxval=max_val, dtype=dtype, seed=seed)
