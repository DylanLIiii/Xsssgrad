# import lib 
from typing import Any
import numpy as np

class Tensor: 
    def __init__(self, data) -> None:


        self.data = np.asarray(data)# accept data, we use numpy here
        self.grad = None  # gradient
        self.data = np.atleast_1d(data)
        self.generator = None # generator is a Function we define
        
    def backward(self):
        funcs = []
        funcs.append(self.generator)
        
        while funcs:
            
            generator = funcs.pop()
            if generator is None: break

            inputs, outputs = generator.inputs, generator.outputs
            gys = [o.grad for o in outputs]
            gxs = generator.backward(*gys)
            
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for input, gx in zip(inputs, gxs): 
                input.grad = gx
                if input.generator is not None:
                        funcs.append(input.generator)
                        
                        
    def __repr__(self):
        return self.__str__()              
                    
    def __str__(self):
        return f"Tensor({self.data}), Shape({self.data.shape})"
        
        
        
class Function: 
    """
    The Function class in your code appears to be an abstract base class (ABC). An ABC is a class that contains one or more abstract methods, and can't be instantiated. 
    Abstract methods are methods that have a declaration but do not have an implementation.
    """
    
    def __call__(self, *inputs):

        self.inputs = inputs 
        ys = self.forward(*inputs) ##here we directly define the forward of inputs, when we call, then we do a forward, in Pytorch, it's the same
        if not isinstance(ys, tuple):
            ys = (ys,)
        self.outputs = ys
        for y in self.outputs: # always maniplate on defined object 
            y.generator = self
            
        return self.outputs if (len(self.outputs)>1) else self.outputs[0] #simple process
        
    def forward(self, inputs): 
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
        x0, x1 = self.inputs
        return x0.data * gys, x1.data * gys
        
    
def add(x0, x1):
    return Add()(x0, x1)

def mul(x0, x1):
    return Mul()(x0, x1)