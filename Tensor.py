# import lib
import numpy as np
import weakref
from config import Config
from tools import sum_to


class Tensor:
    def __init__(self, data) -> None:

        self.data = np.asarray(data)  # accept data, we use numpy here
        self.grad = None  # gradient
        self.data = np.atleast_1d(data)
        self._generator = None  # generator is a Function we define
        self.priority = 0
        
        
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, generator):
        self._generator = generator
        self.priority = generator.priority + 1  # if generator is not None else 0

    def backward(self, retain_graph=True, create_graph=False):
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.data))  # auto init grad of last layer

        funcs = []

        # sort 
        func_set = set()

        def add_func(f):
            if f not in func_set:
                func_set.add(f)
                funcs.append(f)
                funcs.sort(key=lambda x: x.priority)  # sort by priority to support multiple backward

        add_func(self._generator)

        while funcs:

            generator = funcs.pop()
            if generator is None: break

            xs, ys = generator.xs, generator.ys
            gys = [y.grad for y in ys]
            if not create_graph:
                Config.enable_backprop = False

            gxs = generator.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(xs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # accumulate gradient when we call multiple backward
                if x.generator is not None:
                    add_func(x.generator)

            if not retain_graph:
                for y in ys:
                    y.grad = None

            Config.enable_backprop = True

    def zero_grad(self):
        self.grad = None

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __pow__(self, power):
        return pow(self, power)

    def __neg__(self):
        return neg(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Tensor({self.data}), Shape({self.data.shape})"


class Function:
    """
    The Function class in your code appears to be an abstract base class (ABC). An ABC is a class that contains one or more abstract methods, and can't be instantiated.
    Abstract methods are methods that have a declaration but do not have an implementation.
    """

    def __call__(self, *xs):

        self.xs = xs
        ys = self.forward(
            *xs)  # #here we directly define the forward of xs, when we call, then we do a forward, in Pytorch,
        # it's the same

        if Config.enable_backprop: # Control the generation of computational graphs
            if not isinstance(ys, tuple): ys = (ys,)
            self.ys = [y for y in ys]
            self.priority = max([x.priority for x in self.xs])

        for y in self.ys:  # always manipulate on defined object
            y.generator = self

        return self.ys if (len(self.ys) > 1) else self.ys[0]  # simple process

    def forward(self, *xs):
        raise NotImplementedError  # this is a common practice in Python when defining methods in a base class that
        # are intended to be overridden in subclasses.

    def backward(self, gys):
        raise NotImplementedError


class Add(Function):

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return Tensor(x0.data + x1.data)

    def backward(self, gys):
        gy0, gy1 = gys, gys
        if self.x0_shape != self.x1_shape and self.x0_shape != (1,):
            gy0 = sum_to(gy0, self.x0_shape)
            gy1 = sum_to(gy1, self.x1_shape)
        return gy0, gy1


class Mul(Function):

    def forward(self, x0, x1):
        return Tensor(x0.data * x1.data)

    def backward(self, gys):
        x0, x1 = self.xs
        return x1 * gys, x0 * gys


class Sub(Function):

    def forward(self, x0, x1):
        return Tensor(x0.data - x1.data)

    def backward(self, gys):
        return gys, -gys


class Div(Function):

    def forward(self, x0, x1):
        return Tensor(x0.data / x1.data)

    def backward(self, gys):
        x0, x1 = self.xs
        gx0 = gys / x1
        gx1 = gys * (-x0 / x1 ** Tensor(2))
        return gx0, gx1


class Neg(Function):

    def forward(self, x0):
        return Tensor(-x0.data)

    def backward(self, gys):
        return -gys


class Pow(Function):

    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return Tensor(x.data ** self.c)

    def backward(self, gys):
        x = self.xs[0]
        c = self.c
        return Tensor(c) * x ** (c - 1) * gys


def add(x0, x1):
    return Add()(x0, x1)


def mul(x0, x1):
    return Mul()(x0, x1)


def sub(x0, x1):
    return Sub()(x0, x1)


def div(x0, x1):
    return Div()(x0, x1)


def neg(x0):
    return Neg()(x0)


def pow(x, c):
    return Pow(c)(x)
