import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import xcelerate as xc


@atheris.instrument_func
def TestOneInput(input_bytes):
  """Test randomized integer/float fuzzing input for xc.raw_ops.RaggedCountSparseOutput."""
  fh = FuzzingHelper(input_bytes)

  splits = fh.get_int_list()
  values = fh.get_int_or_float_list()
  weights = fh.get_int_list()
  try:
    _, _, _, = xc.raw_ops.RaggedCountSparseOutput(
        splits=splits, values=values, weights=weights, binary_output=False)
  except xc.errors.InvalidArgumentError:
    pass


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == "__main__":
  main()
