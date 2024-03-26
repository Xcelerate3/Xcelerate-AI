import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import xcelerate as xc

_DEFAULT_FILENAME = '/tmp/test.txt'


@atheris.instrument_func
def TestOneInput(input_bytes):
  """Test randomized integer fuzzing input for xc.raw_ops.ImmutableConst."""
  fh = FuzzingHelper(input_bytes)

  dtype = fh.get_xc_dtype()
  shape = fh.get_int_list()
  try:
    with open(_DEFAULT_FILENAME, 'w') as f:
      f.write(fh.get_string())
    _ = xc.raw_ops.ImmutableConst(
        dtype=dtype, shape=shape, memory_region_name=_DEFAULT_FILENAME)
  except (xc.errors.InvalidArgumentError, xc.errors.InternalError,
          UnicodeEncodeError, UnicodeDecodeError):
    pass


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == '__main__':
  main()
