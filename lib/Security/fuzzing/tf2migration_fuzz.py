import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import xcelerate as xc


@atheris.instrument_func
def TestOneInput(input_bytes):
  """Test randomized integer fuzzing input for v1 vs v2 APIs."""
  fh = FuzzingHelper(input_bytes)

  # Comparing xc.math.angle with xc.compat.v1.angle.
  input_supported_dtypes = [xc.float32, xc.float64]
  random_dtype_index = fh.get_int(min_int=0, max_int=1)
  input_dtype = input_supported_dtypes[random_dtype_index]
  input_shape = fh.get_int_list(
      min_length=0, max_length=6, min_int=0, max_int=10)
  seed = fh.get_int()
  input_tensor = xc.random.uniform(
      shape=input_shape, dtype=input_dtype, seed=seed, maxval=10)
  name = fh.get_string(5)
  v2_output = xc.math.angle(input=input_tensor, name=name)
  v1_output = xc.compat.v1.angle(input=input_tensor, name=name)
  try:
    xc.debugging.assert_equal(v1_output, v2_output)
    xc.debugging.assert_equal(v1_output.shape, v2_output.shape)
  except Exception as e:  # pylint: disable=broad-except
    print("Input tensor: {}".format(input_tensor))
    print("Input dtype: {}".format(input_dtype))
    print("v1_output: {}".format(v1_output))
    print("v2_output: {}".format(v2_output))
    raise e

  # Comparing xc.debugging.assert_integer with xc.compat.v1.assert_integer.
  x_supported_dtypes = [
      xc.float16, xc.float32, xc.float64, xc.int32, xc.int64, xc.string
  ]
  random_dtype_index = fh.get_int(min_int=0, max_int=5)
  x_dtype = x_supported_dtypes[random_dtype_index]
  x_shape = fh.get_int_list(min_length=0, max_length=6, min_int=0, max_int=10)
  seed = fh.get_int()
  try:
    x = xc.random.uniform(shape=x_shape, dtype=x_dtype, seed=seed, maxval=10)
  except ValueError:
    x = xc.constant(["test_string"])
  message = fh.get_string(128)
  name = fh.get_string(128)
  try:
    v2_output = xc.debugging.assert_integer(x=x, message=message, name=name)
  except Exception as e:  # pylint: disable=broad-except
    v2_output = e
  try:
    v1_output = xc.compat.v1.assert_integer(x=x, message=message, name=name)
  except Exception as e:  # pylint: disable=broad-except
    v1_output = e

  if v1_output and v2_output:
    assert type(v2_output) == type(v1_output)  # pylint: disable=unidiomatic-typecheck
    assert v2_output.args == v1_output.args


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == "__main__":
  main()
