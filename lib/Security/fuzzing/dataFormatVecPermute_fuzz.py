import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import xcelerate as xc


@atheris.instrument_func
def TestOneInput(input_bytes):
  """Test randomized integer fuzzing input for xc.raw_ops.DataFormatVecPermute."""
  fh = FuzzingHelper(input_bytes)

  dtype = fh.get_xc_dtype()
  # Max shape can be 8 in length and randomized from 0-8 without running into
  # a OOM error.
  shape = fh.get_int_list(min_length=0, max_length=8, min_int=0, max_int=8)
  seed = fh.get_int()
  try:
    x = xc.random.uniform(shape=shape, dtype=dtype, seed=seed)
    src_format_digits = str(fh.get_int(min_int=0, max_int=999999999))
    dest_format_digits = str(fh.get_int(min_int=0, max_int=999999999))
    _ = xc.raw_ops.DataFormatVecPermute(
        x,
        src_format=src_format_digits,
        dst_format=dest_format_digits,
        name=fh.get_string())
  except (xc.errors.InvalidArgumentError, ValueError, TypeError):
    pass


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == '__main__':
  main()
