import numpy as np

inputs = np.array([1, 2, 3, 4])

weights = np.array(
    [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ]
)

biases = np.array([0.1, 0.2, 0.3])
learning_rate = 0.001

output = None


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)


for i in range(200):

    # Forward pass
    z = np.dot(weights, inputs) + biases
    a = relu(z)
    y = np.sum(a)

    # Calculate loss
    loss = y**2

    # Backward pass
    # Gradient of loss with respect to output y
    derivative_loss_y = 2 * y

    # Gradient of y with respect to a
    derivative_y_a = np.ones_like(a)

    # Gradient of loss with respect to a
    derivative_loss_a = derivative_loss_y * derivative_y_a

    # Gradient of a with respect to z (ReLu derivative)
    derivative_a_z = relu_derivative(z)

    # Gradient of loss with respect to z
    derivative_loss_z = derivative_loss_a * derivative_a_z

    # Gradient of z with respect to weights and biases
    derivative_loss_w = np.outer(derivative_loss_z, inputs)
    derivative_loss_b = derivative_loss_z

    # Update weights and biases
    weights -= learning_rate * derivative_loss_w
    biases -= learning_rate * derivative_loss_b

    print(f"Iteration no.{i + 1}: Loss = {loss}")

print(f"Final weights:\n", weights)
print(f"Final biases:\n", biases)
