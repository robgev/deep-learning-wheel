import numpy as np
from activations import Activation, sigmoid
from errors import Error, mse
from typing import Sequence

class Layer:
    pre_activation = []
    output = []
    gradient = []
    def __init__(self, v_weight, v_bias, activation: Activation) -> None:
        self.v_weight = np.array(v_weight)
        self.v_bias = np.array(v_bias)
        self.activation = activation
        pass

    def calculate(self, inp): 
        self.pre_activation = self.v_weight * inp + self.v_bias

    def activate(self):
        self.output = self.activation.f(self.pre_activation)
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
            self.layers[i].output = layer.output

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


if __name__ == "__main__":
    hidden = Layer(v_weight=[0.5, 0.5], v_bias=[0, 0], activation=sigmoid)
    output = Layer(v_weight=[0.5], v_bias=[0], activation=sigmoid)

    net = NeuralNetwork(lr=0.5, layers=[hidden, output], error=mse, epochs=100, tolerance=0.0001)

    x = np.array([[0, 0, 1, 1],
                   [0, 1, 0, 1]])
    y_true = np.array([[0, 1, 1, 0]])

    net.train(x, y_true)
    print(net.forward(x))

