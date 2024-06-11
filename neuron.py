class Neuron:
    def __init__(self, weights: list[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def output(self, inputs: list[float]) -> float:
        # Summation of weighted inputs.
        return (
            sum(input_val * weight for input_val, weight in zip(inputs, self.weights))
            + self.bias
        )
