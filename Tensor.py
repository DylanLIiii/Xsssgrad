# import lib 
from typing import Any
import numpy as np

class Tensor: 
    def __init__(self, data) -> None:
        """
         Initialize the object with data. This is the method that must be called before any operations that are performed on the object such as __init__ and __setstate__
         
         @param data - The data to be used for the optimizer
         
         We use np.atleast_1d() to make sure the data shape is right
         because when you pass a scaler to np.array(), the shape will be None.
         
         @return None The optimizer is initialized with the data and the
        """

        self.data = np.asarray(data)# accept data, we use numpy here
        self.grad = None  # gradient
        self.data = np.atleast_1d(data)
        self.generator = None #g generator is a Function we define
        
        def __str__(self):
            """
            Return a string representation of the tensor.
            """
            return f"Tensor({self.data}), Shape({self.data.shape})"
        
        def backward(self, grad_in):
            generator = self.generator 
            if generator is not None:
                inputs = generator.inputs
                inputs.grad = generator.backward(grad_in)
                inputs.backward()
            
        
        
class Function: 
    """
    The Function class in your code appears to be an abstract base class (ABC). An ABC is a class that contains one or more abstract methods, and can't be instantiated. 
    Abstract methods are methods that have a declaration but do not have an implementation.
    """
    
    def __call__(self, inputs, *args: Any, **kwds: Any) -> Any:
        """
         Calls the forward function. This is the method that should be called by the user.
         
         @param inputs - The inputs to the network. It should be a list of numpy arrays
         @param args - Positional arguments passed to the network
         @param kwds - Keyword arguments passed to the
        """
        
        self.inputs = inputs 
        y = self.forward(inputs)
        y.generator = self# here we directly define the forward of inputs, when we call, then we do a forward, in Pytorch, it's the same
    
    def forward(self, inputs): 
        """
         Forward function. This is called by the model to get the output of the model.
         
         @param inputs - A list of inputs to the model. Each input is a tuple ( x y)
         
        """
        raise NotImplementedError #his is a common practice in Python when defining methods in a base class that are intended to be overridden in subclasses.
    
    def backward(self, grad_out):
        """
         Backward pass for the layer. This is called in the end of the forward pass to get the gradients for the output layer
         
         @param grad_out - Gradient tensor with respect to
        """
        raise NotImplementedError
        
        
class Square(Function):
    
    def forward(self, inputs): 
        y = self.inputs.data ** 2
        return Tensor(y)
    
    def backward(self, grad_in):
        # grad_in is the grad from last layer
        return 2*self.inputs.data * grad_in