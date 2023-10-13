from tensor import Tensor, Function
import numpy as np


class Reshape(Function):
    def __init__(self, shape):  # target shape
        self.x_shape = None  # original shape
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.reshape(x.data, self.shape)
        return Tensor(y)

    def backward(self, gys):
        gys.data = np.reshape(gys.data, self.x_shape)
        return gys


def reshape(x, shape):
    return Reshape(shape)(x)


class Transpose(Function):

    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = np.transpose(x.data, self.axes)
        return Tensor(y)

    def backward(self, gys):
        if self.axes is None:
            return np.transpose(gys)
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return np.transpose(gys, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class BoradCastTo(Function):
    def __init__(self, shape):
        self.shape = shape
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x.data, self.shape)
        return Tensor(y)

    def backward(self, gys):
        gx = sum_to(gys, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return x
    return BoradCastTo(shape)(x)


def sum_to(x, shape):
    """
    Computes the sum of the input tensor `x` across specified dimensions to match the shape of `shape`.
    The broadcast backpropagation is implemented in the sum_to function.

    Args:
        x (torch.Tensor): The input tensor to be summed.
        shape (tuple): The desired shape of the output tensor.

    Returns:
        torch.Tensor: The tensor obtained by summing `x` across specified dimensions to match the shape of `shape`.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = np.sum(x.data, lead_axis + axis, keepdims=True)
    if lead > 0:
        y = np.squeeze(y, lead_axis)
    return y


class MatMul(Function):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, W: Tensor):
        y = np.dot(x.data, W.data)
        return Tensor(y)

    def backward(self, gys: Tensor):
        x, W = self.xs
        gx = np.dot(gys.data, W.data.T)
        gW = np.dot(x.data.T, gys.data)
        return gx, gW


def matmul(x: Tensor, W: Tensor):
    return MatMul()(x, W)
