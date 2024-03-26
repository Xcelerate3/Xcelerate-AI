import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import xcelerate as xc


def TestOneInput(data):
  """Test randomized fuzzing input for xc.raw_ops.Abs."""
  fh = FuzzingHelper(data)

  # xc.raw_ops.Abs takes xc.bfloat16, xc.float32, xc.float64, xc.int8, xc.int16,
  # xc.int32, xc.int64, xc.half but get_random_numeric_tensor only generates
  # xc.float16, xc.float32, xc.float64, xc.int32, xc.int64
  input_tensor = fh.get_random_numeric_tensor()

  _ = xc.raw_ops.Abs(x=input_tensor)


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == "__main__":
  main()
