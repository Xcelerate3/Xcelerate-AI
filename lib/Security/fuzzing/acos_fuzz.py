import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import xcelerate as xc


def TestOneInput(data):
  """Test randomized fuzzing input for xc.raw_ops.Acos."""
  fh = FuzzingHelper(data)

  # xc.raw_ops.Acos takes xc.bfloat16, xc.half, xc.float32, xc.float64, xc.int8,
  # xc.int16, xc.int32, xc.int64, xc.complex64, xc.complex128, but
  # get_random_numeric_tensor only generates xc.float16, xc.float32, xc.float64,
  # xc.int32, xc.int64
  input_tensor = fh.get_random_numeric_tensor()
  _ = xc.raw_ops.Acos(x=input_tensor)


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == "__main__":
  main()
