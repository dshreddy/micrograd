from graphviz import Digraph

class Value:
    '''
    A class that tries to mimic the behavior of a tensor in a deep learning framework.
    It holds a float value, its gradient, and methods for basic arithmetic operations.
    It also has internal variables to keep track of the autograd graph construction.
    This class is used to implement automatic differentiation.
    '''

    def __init__(self, data:float, _label='', _prev=(), _op='') -> None:
        '''
        Initializes a Value instance.
        :param data: The float value to be stored in the Value instance.
        :param _label: A label for the Value instance, used for debugging.
        :param _prev: A tuple of previous Value instances, used to construct the autograd graph.
        :param _op: The operation that produced this Value instance, used for debugging.
        :raises Exception: If data is not a float or if _prev is not a tuple.
        '''
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None

        # internal variables used for autograd graph construction
        self._label = _label
        self._prev = set(_prev)
        self._op = _op
        
    def __repr__(self):
        '''
        Returns a string representation of the Value instance.
        :return: A string representation of the Value instance.
        '''
        return f"Value(data={self.data}, grad={self.grad}, label='{self._label}', prev={self._prev}, op='{self._op}')"
    
    def __add__(self, other):
        '''
        Implements addition for Value instances.
        If the other operand is not a Value instance, it tries to convert it to a Value.
        :param other: The other operand to be added.
        :raises Exception: If the other operand is not a Value instance or a convertible type.
        :return: A new Value instance that is the sum of the two operands.
        '''
        if not isinstance(other, Value):
            if isinstance(other, (int, float)):
                other = Value(other)
            else:
                raise Exception(f"Can't add an instance of type Value with an instance of type {type(other)}")
            
        out = Value(self.data+other.data, _prev=(self, other), _op='+')

        def backward():
            '''
            Backward pass for the addition operation.
            This method computes the gradients of the operands with respect to the output.
            It adds the gradient of the output to the gradients of the operands.
            '''
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = backward

        return out
    
    def __mul__(self, other):
        '''
        Implements multiplication for Value instances.
        If the other operand is not a Value instance, it tries to convert it to a Value.
        :param other: The other operand to be multiplied.
        :raises Exception: If the other operand is not a Value instance or a convertible type.
        :return: A new Value instance that is the product of the two operands.
        '''
        if not isinstance(other, Value):
            if isinstance(other, (int, float)):
                other = Value(other)
            else:
                raise Exception(f"Can't multiply an instance of type Value with an instance of type {type(other)}")
            
        out = Value(self.data * other.data, _prev=(self, other), _op='*')

        def backward():
            '''
            Backward pass for the multiplication operation.
            This method computes the gradients of the operands with respect to the output.
            It multiplies the gradient of the output by the other operand's data and vice versa.
            '''
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward

        return out
    
    def __pow__(self, power):
        '''
        Implements exponentiation for Value instances.
        If the power is not an integer or float, it raises an exception.
        :param power: The exponent to which the Value instance is raised.
        :raises Exception: If power is not an integer or float.
        :return: A new Value instance that is the result of raising the original Value instance to the given power.
        '''
        if not isinstance(power, (int, float)):
            raise Exception(f"Can't raise an instance of type Value to an instance of type {type(power)}")
        
        out = Value(self.data ** power, _prev=(self,), _op=f'**{power}')

        def backward():
            '''
            Backward pass for the exponentiation operation.
            This method computes the gradient of the base with respect to the output.
            It multiplies the gradient of the output by the power and the base raised to (power - 1).
            '''
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = backward

        return out
    
    def relu(self):
        '''
        Implements the ReLU activation function for Value instances.
        :return: A new Value instance that is the result of applying the ReLU function to the original Value instance.
        '''
        out = Value(max(0, self.data), _prev=(self,), _op='ReLU')
        
        def backward():
            '''
            Backward pass for the ReLU activation function.
            This method computes the gradient of the input with respect to the output.
            It sets the gradient to 0 if the input is less than or equal to 0, otherwise it passes the gradient through.
            '''
            self.grad += (1 if self.data > 0 else 0) * out.grad

        out._backward = backward

        return out
    
    def backward(self):
        '''
        Backward pass for the Value instance.
        This method computes the gradients of all Value instances in the autograd graph.
        It starts from the current Value instance and traverses the graph in reverse order.
        '''

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
        
        return topo
            
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        # a/b = a * (1/b) = a * (b ** -1)
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        # Ex: 3 / self
        return other * (self ** -1)
    
    def __rmul__(self, other):
        return self*other
    
    def draw_dot(self):
        '''
        Builds the autograd graph tree for the Value instance.
        This method is used to construct the graph of operations that led to this Value instance.
        :return: A list of Value instances that form the autograd graph tree.
        '''
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(self)

        format = 'svg'
        rankdir = 'BT'

        dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
        for n in nodes:
            dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n._label, n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
        return dot
    