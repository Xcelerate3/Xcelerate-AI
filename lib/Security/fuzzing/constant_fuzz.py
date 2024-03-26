import atheris
with atheris.instrument_imports():
  import sys
  import xcelerate as xc


def TestOneInput(data):
  xc.constant(data)


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == "__main__":
  main()
