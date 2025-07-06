Class Layer
- weights: array
- bias: array
- f(x): function
- df(x): function 
- activation(input): Activation extends Layer
- pre_activation: array 
- output: array 
- gradient: array
- calculate(input): self.pre_activation = self.f(input) * self.weights + self.bias
- activate(): - self.output = self.activation(self.pre_activation)

Output Layer 
    - weights: 1s
    - bias: 0s 
    - f(x) = x 
    - df(x) = 1
    - activation: Activation
