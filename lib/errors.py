import numpy as np

class Error:
    def __init__(self, f, df) -> None:
        self.f = f
        self.df = df
        pass

def MSE(yBar, y):
    return 0.5 * np.sum((yBar - y)**2)
def MSE_derivative(yBar, y):
    return y - yBar

mse = Error(MSE, MSE_derivative)
