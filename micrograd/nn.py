import random
from micrograd.engine import Value

class Module:
    '''
    Base class for all neural network modules.
    '''
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    '''
    A single neuron with weights and bias.
    It can apply a ReLU activation function if nonlin is True.
    The weights are initialized randomly between -1 and 1, and the bias is initialized to 0.
    The neuron computes the weighted sum of inputs plus bias, and applies the activation function if specified.
    '''

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.w)})"

class Layer(Module):
    '''
    A layer of neurons.
    It contains multiple Neuron instances, each with its own weights and bias.
    The layer can apply a non-linear activation function to each neuron if specified.
    '''

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    '''
    A Multi-Layer Perceptron (MLP) consisting of multiple layers of neurons.
    It takes an input size and a list of output sizes for each layer.
    '''
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"