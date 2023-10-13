import numpy as np
from function import Transpose, Reshape, BoradCastTo, MatMul
from tensor import Tensor


def test_transpose():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    t = Transpose()
    y = t.forward(x)
    assert np.array_equal(y.data, np.transpose(x))

    gy = np.array([[7, 8], [9, 10], [11, 12]])
    gx = t.backward(gy)
    assert np.array_equal(gx, np.transpose(gy))

    t = Transpose(axes=(1, 0))
    y = t.forward(x)
    assert np.array_equal(y.data, np.transpose(x, axes=(1, 0)))

    gy = np.array([[7, 8], [9, 10]])
    gx = t.backward(gy)
    assert np.array_equal(gx, np.transpose(gy, axes=(1, 0)))

    
def test_reshape():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    r = Reshape((3, 2))
    y = r.forward(x)
    assert np.array_equal(y.data, np.reshape(x, (3, 2)))

    gy = np.array([[7, 8], [9, 10], [11, 12]])
    gx = r.backward(Tensor(gy)).data
    assert np.array_equal(gx, np.reshape(gy, x.shape))

    r = Reshape((2, 3))
    y = r.forward(x)
    assert np.array_equal(y.data, np.reshape(x, (2, 3)))

    gy = np.array([[7, 8, 9], [10, 11, 12]])
    gx = r.backward(Tensor(gy)).data
    assert np.array_equal(gx, np.reshape(gy, x.shape))

    # Test that the original tensor is not modified
    assert np.array_equal(x, np.array([[1, 2, 3], [4, 5, 6]]))

def test_matmul():
    x = np.array([[1, 2], [3, 4]])
    W = np.array([[2, 0], [0, 2]])
    m = MatMul()
    y = m(Tensor(x), Tensor(W))
    assert np.array_equal(y.data, np.dot(x, W))

    gy = np.array([[1, 2], [3, 4]])
    gx, gW = m.backward(Tensor(gy))
    assert np.array_equal(gx, np.dot(gy, W.T))
    assert np.array_equal(gW, np.dot(x.T, gy))