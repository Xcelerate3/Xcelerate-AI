import sys

import xcelerate


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["xcelerate.runtime.driver"])
    reload(sys.modules["xcelerate.runtime"])
    mod = sys.modules[xcelerate.runtime.driver.__module__]
    assert isinstance(xcelerate.runtime.driver, getattr(mod, "LazyProxy"))
    assert xcelerate.runtime.driver._obj is None
    utils = xcelerate.runtime.driver.utils  # noqa: F841
    assert issubclass(xcelerate.runtime.driver._obj.__class__, getattr(xcelerate.backends.driver, "DriverBase"))
