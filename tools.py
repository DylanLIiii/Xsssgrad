import numpy as np


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
        y = np.squeeze(y ,lead_axis)
    return y
