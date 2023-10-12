# import lib 
from typing import Any
import numpy as np

class Tensor: 
    def __init__(self, data) -> None:


        self.data = np.asarray(data)# accept data, we use numpy here
        self.grad = None  # gradient
        self.data = np.atleast_1d(data)
        self._generator = None # generator is a Function we define
        self.priority = 0
        
        @property
        def generator(self):
            return self._generator
        
        @generator.setter
        def generator(self, generator):
            self._generator = generator
            self.priority = generator.priority + 1 #if generator is not None else 0
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = []
        
        # sort 
        func_set = set()
        
        def add_func(f):
            if f not in func_set:
                func_set.add(f)
                funcs.append(f)
                funcs.sort(key=lambda x: x.priority)
                
        add_func(self._generator)
        
        while funcs:
            
            generator = funcs.pop()
            if generator is None: break

            xs, ys = generator.xs, generator.ys
            gys = [o.grad for o in ys]
            gxs = generator.backward(*gys)
            
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(xs, gxs): 
                if x.grad is None: 
                    x.grad = gx
                x.grad = x.grad + gx # accumulate gradient when we call multiple backward
                if x.generator is not None:
                        add_func(x.generator)
                        
        def zero_grad():
            self.grad = None
            '''if self.generator is not None:
                self.generator.zero_grad()'''
                        
                        
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
        ys = self.forward(*xs) ##here we directly define the forward of xs, when we call, then we do a forward, in Pytorch, it's the same
        if not isinstance(ys, tuple):
            ys = (ys,)
        self.ys = ys
        self.priority = max([x.priority for x in self.xs])
        
        
        for y in self.ys: # always maniplate on defined object 
            y.generator = self
        
            
        return self.ys if (len(self.ys)>1) else self.ys[0] #simple process
        
    def forward(self, xs): 
        raise NotImplementedError #his is a common practice in Python when defining methods in a base class that are intended to be overridden in subclasses.
    
    def backward(self, gys):
        raise NotImplementedError
        
class Add(Function):
        
        def forward(self, x0, x1):
            return Tensor(x0.data + x1.data)
        
class Mul(Function):
    
    def forward(self, x0, x1):
        return Tensor(x0.data * x1.data)
    
    def backward(self, gys):
        x0, x1 = self.xs
        return x0.data * gys, x1.data * gys
        
    
def add(x0, x1):
    return Add()(x0, x1)

def mul(x0, x1):
    return Mul()(x0, x1)