import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import xcelerate as xc


def TestOneInput(data):
  """Test numeric randomized fuzzing input for xc.raw_ops.Add."""
  fh = FuzzingHelper(data)

  # xc.raw_ops.Add also takes xc.bfloat16, xc.half, xc.float32, xc.float64,
  # xc.uint8, xc.int8, xc.int16, xc.int32, xc.int64, xc.complex64,
  # xc.complex128, but get_random_numeric_tensor only generates xc.float16,
  # xc.float32, xc.float64, xc.int32, xc.int64
  input_tensor_x = fh.get_random_numeric_tensor()
  input_tensor_y = fh.get_random_numeric_tensor()

  try:
    _ = xc.raw_ops.Add(x=input_tensor_x, y=input_tensor_y)
  except (xc.errors.InvalidArgumentError, xc.errors.UnimplementedError):
    pass


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == "__main__":
  main()
