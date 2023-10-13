from contextlib import contextmanager


class Config:
    enable_backprop = True


@contextmanager
def no_grad():
    """The no_grad context manager is defined using the @contextmanager decorator, which allows it to be used as a
    context manager using the with statement. When the no_grad context manager is entered,
    it sets Config.enable_backprop to False, which disables backpropagation. This is useful in situations where we
    want to perform inference on a neural network without updating its weights. After the with block is exited,
    the final block is executed, which sets Config.enable_backprop back to True, re-enabling backpropagation.
    """
    Config.enable_backprop = False
    try:
        yield
    except Exception:
        raise Exception("No grad, can not process.")
    finally:
        Config.enable_backprop = True
