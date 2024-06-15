import numpy as np

weights = np.array([-3.0, -1.0, 2.0])
bias = 1.0
inputs = np.array([1.0, -2.0, 3.0])
target = 0
learning_rate = 0.001

output = None

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

for iteration in range(360):
    linear_output = np.dot(weights, inputs) + bias
    output = relu(linear_output)
    loss = (output - target) ** 2

    dloss_doutput = 2 * (output - target)
    doutput_dlinear = relu_derivative(linear_output)
    dlinear_dweights = inputs
    dlinear_dbias = 1.0

    dloss_dlinear = dloss_doutput * doutput_dlinear
    dloss_dweights = dloss_dlinear * dlinear_dweights
    dloss_dbias = dloss_dlinear * dlinear_dbias

    weights -= learning_rate * dloss_dweights
    bias -= learning_rate * dloss_dbias

    print(f"Iteration {iteration + 1}, Loss: {loss}, Output: {output}")

print(f"Final weights: {weights}")
print(f"Final bias: {bias}")
print(f"Final output: {output}")