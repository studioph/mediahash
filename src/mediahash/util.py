import contextlib
from typing import Unpack

@contextlib.contextmanager
def reraise(
    *errors: Unpack[Exception], as_: Exception = RuntimeError, **exception_args
):
    try:
        yield
    except errors as error:
        raise as_(error, **exception_args) from error