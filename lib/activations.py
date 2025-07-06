# Declarative wrappers around common activation functions
import numpy as np

class Activation:
    def __init__(self, f, df) -> None:
        self.f = f
        self.df = df
        pass

def sigmoid_f(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_df(x):
    return sigmoid_f(x) * (1 - sigmoid_f(x))


sigmoid = Activation(sigmoid_f, sigmoid_df)
