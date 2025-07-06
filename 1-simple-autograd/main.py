# Simple autograd engine for defining a NN with separate, declarative layers 
# Forgetting about optimization here to programmatically notate the underlying math
# Later, instead of this procedural approach, we will use linear algebra tricks 
# to parallelize computations and cut training times by utilizing GPU for those parallel 
# computations. We will achieve that by using higher degree tensors
from typing import Sequence
from lib.activations import Activation
from lib.errors import Error

class Layer:
    pre_activation = []
    output = []
    gradient = []
    def __init__(self, v_weight, v_bias, activation: Activation) -> None:
        self.v_weight = v_weight
        self.v_bias = v_bias
        self.activation = activation
        pass

    def calculate(self, inp): 
        self.pre_activation = self.v_weight * inp + self.v_bias

    def activate(self):
        self.output = self.activation.f(self.pre_activation)

    # Given the derivative of loss w.r.t. layer output dL/dO, 
    # Since output of the layer O depends on weights and the input of the layer:
    # O = activation(W * I + B) 
    # 1.add the chain rule members of this layer aka backpropagate for further layers
    #   dL / dW = dL/dO * dO/dP * dP / dW 
    #   dO / dP = d (sigmoid(W * I + B)) / d(W * I + B) = sigmoid'(W * I + B)
    #   dP / dW = d(W * I + B) / dW = I
    #   where L = loss, W = weights of layer, O = output with activation, P = pre activation
    #   and I = Input of the layer - aka output of the previous layer
    # This process gives us an opportunity to "disect" the effects of layer output on loss 
    # into parts and figure out how much weights contribute to it

    # NOTE: notice the iterative step when calculating dL/dW for 2 neighboring layers:
    # If we feed O1 as input to O2 then 
    # O2 = sigmoid(W2 * O1 + B2), P2 = W2 * O1 + B2
    # dL / dW2 = dL/dO2 * dO2/dP2 * dP2/dW2
    #            ----------------   .......
    # dL / dW1 = dL/dO2 * dO2/dP2 * dP2/dO1 * dO1/dP1 * dP1/dW1
    #            ----------------   ~~~~~~~~~~~~~~~~~   .......
    # Hence: For each layer, update the chain: how much is the change of loss from layer's activation and neuron 
    # functions:
    # delta - how much the loss changes from the change in the subsequent layer's input
    # dP~N/dO~(N-1) - i.e. how much the change in current layer's output will affect the subsequent layer's input
    # dO~N/dP~N - i.e. how much changing the activation's input in current layer will change the current layer's output
    # To calculate how the change in weights affects loss by multiplying it with, multiply delta with:
    # dP1/dW1 - How much the activation input changes from the current layer's weights 
    # These three together combine to show how much this layer's weights affect the next layer's input
    # Multiplying delta with it shows us how layer's weights affect the loss

    # NOTE: in more general case, if we go non-linear for layers 
    # dO / dI = d (sigmoid(W * f(I) + B)) / dI = sigmoid'(W * f(I) + B) * W * f'(I)
    # Because f(I) = I in linear case, f'(I) = 1
    # However, since the FFNN can approximate any polynomial (in some cases also any turing machine)
    # We don't need to do this
    def back(self, dL_dI):
        return dL_dI * self.v_weight * self.activation.df(self.pre_activation)

# TODO: Maybe define Input Layer 
# TODO: Maybe define Output Layer 

class NeuralNetwork:
    def __init__(self, lr, layers: Sequence[Layer], error: Error, epochs, tolerance) -> None:
        self.learning_rate = lr
        self.layers = layers
        self.error = error
        self.epochs = epochs
        self.tolerance = tolerance
        pass

    def forward(self, x):
        self.layers[0].output = x
        for i, layer in enumerate(self.layers, 1):
            layer.calculate(self.layers[i - 1].output)
            layer.activate()

        return self.layers[-1].output

    def backward(self, y_true, output):
        delta = self.error.df(y_true, output)
        for i in range(len(self.layers) - 1, 0, -1):
            delta = self.layers[i].back(delta)
            self.layers[i].gradient = delta * self.layers[i - 1].output

    def update(self):
        for layer in self.layers:
            layer.v_weight -= self.learning_rate * layer.gradient

    def train(self, x, y_true):
        for i in range(self.epochs):
            y = self.forward(x)
            err = self.error.f(y_true, y)
            self.backward(y_true, y)
            self.update()
            print(f"Epoch #{i}: {err}")
            if err < self.tolerance:
                return 


    
