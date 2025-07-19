# Simple autograd

Simple autograd engine for defining a NN with separate, declarative layers 
Forgetting about optimization here to programmatically notate the underlying math
Later, instead of this procedural approach, we will use linear algebra tricks 
to parallelize computations and cut training times by utilizing GPU for those parallel 
computations. We will achieve that by using higher degree tensors


Given the derivative of the loss function w.r.t. layer output - dL/dO, 
we need to compute the delta in loss w.r.t. layer weights for each layer.
Since output of the layer O depends on weights and the input of the layer 
then we can say O is a function of W and I:
O = activation(W * I + B) 
Where W is the weight tensor, and I is the input tensor

Hence for every layer we need to:
1. Use the chain rule to add the "components" of this layer - aka backpropagate for further layers
In words - delta of loss from weights is split into delta of loss from output, delta of output from result before 
activation function is applied and delta of pre-activation input w.r.t weights
```
    dL / dW = dL/dO * dO/dP * dP / dW 
```
 Assume we use sigmoid as an activation function:
```
    O = sigmoid(P), P = W * I + B
    dO / dP = d (sigmoid(W * I + B)) / d(W * I + B) = sigmoid'(W * I + B)
```
Then:
```
  dP / dW = d(W * I + B) / dW = I
```
  where L = loss function, W = weights of the layer, O = output with activation, P = output pre activation
  and I = Input of the layer - aka input of neural network or the output of the previous layer

This process gives us an opportunity to "disect" the effects of the layer on loss 
into parts and figure out how much weights contribute to that effect
Then, adjusting the weights towards minimum loss gives as a fitting model

## How can we make this iterative?
Let's notice the iterative step when calculating dL/dW for 2 neighboring layers:
If we feed O1 as input to O2 then 
O2 = sigmoid(W2 * O1 + B2), P2 = W2 * O1 + B2
```
dL / dW2 = dL/dO2 * dO2/dP2 * dP2/dW2
           ----------------   .......
```
```
dL / dW1 = dL/dO2 * dO2/dP2 * dP2/dO1 * dO1/dP1 * dP1/dW1
           ----------------   ~~~~~~~~~~~~~~~~~   .......
```

As you can see first 2 members are the same, each layer's last member is the same.
For each layer, we will have 2 more components to add:
```
dP.(N + 1)/dO.N - i.e. how the change in this layer's output affects the subsequent layer's input
dO.N/dP.N - i.e. how much delta of pre-activation in current layer will change the current layer's output
```

The final component to find each layer's "impact" on loss is to calculate dP.() how the change in weights 
affects layer's pre-activation output:
```
dP.N/dW.N - How the small change in weights changes the layer's pre-activation output 
```
We can say that these three latter components together combine to show how much this layer's weights affect the next layer's input
"Tying" those three together at the end of the "chain" inherited from the outter layer, we get to see how the layer's weights affect the loss

## Why do we take linear functions for neurons?
We don't need to. In more general case, when we have primitive non-linear functions for neurons:
```
dO / dI = d (sigmoid(W * f(I) + B)) / dI = sigmoid'(W * f(I) + B) * W * f'(I)
```
In case of a simple linear function the neuron function is f(I) = I, so f'(I) = 1. 

This is not necessary and neurons can use any functions with known derivatives. 
However, since the FFNN can approximate any polynomial, (in some cases also any turing machine, we will discuss this in next lessons)
we don't need to complicate our lives.

## Making neural networks declarative
The project has 2 classes - Layer and NeuralNetwork, and two utility classes - Activation and Error

To define a NeuralNetwork, we construct it with hyper-parameters, a sequence of Layers and an Error
To define a Layer, we construct it with weights, biases and an Activation
To define an Activation or an Error, we define the activation/error function and its derivative. 

This way you can "declare" a neural network without actually concerning yourself 
with what layers actually do underneath.
