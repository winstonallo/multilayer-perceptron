import numpy as np
from numpy import ndarray
from multiprocessing import Pool, cpu_count
from neural_network import NeuralNetwork
from utils import evaluate_model
from tqdm import tqdm


def train_and_evaluate(params):
    n_layers, n_neurons, learning_rate, n_epochs, x_train, y_train, x_test, y_test, seed = params
    np.random.seed(seed)
    model = NeuralNetwork(
        n_layers=n_layers,
        n_inputs=len(x_train[0]),
        n_outputs=1,
        n_neurons=n_neurons,
        hidden_act="ReLU",
        output_act="Sigmoid",
        loss_func="BCE",
        learning_rate=learning_rate,
        n_epochs=n_epochs,
    )
    model.fit(x_train, y_train)
    test_loss = evaluate_model(model, x_test, y_test)
    return test_loss, params


def tune(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    seed,
    n_layers_opts: list[int] = [2, 3, 4],
    n_neurons_opts: list[int] = [4, 8, 16],
    learning_rate_opts: ndarray = np.arange(0.005, 0.105, 0.005),
    n_epochs_opts: list[int] = range(150, 250, 10),
) -> dict:
    best_loss = float("inf")
    best_params = {}
    results = []

    param_combinations = [
        (n_layers, n_neurons, learning_rate, n_epochs, x_train, y_train, x_test, y_test, seed)
        for n_layers in n_layers_opts
        for n_neurons in n_neurons_opts
        for learning_rate in learning_rate_opts
        for n_epochs in n_epochs_opts
    ]

    with Pool(processes=cpu_count()) as pool:
        with tqdm(
            total=len(param_combinations),
            ascii=" =",
            desc="Tuning Params",
            bar_format="{desc}: {percentage:3.0f}%|{bar}|[{remaining}, {rate_fmt}{postfix}]",
        ) as t:
            for test_loss, params in pool.imap(train_and_evaluate, param_combinations):
                results.append(
                    {
                        "n_layers": params[0],
                        "n_neurons": params[1],
                        "learning_rate": params[2],
                        "n_epochs": params[3],
                        "test_loss": test_loss,
                    }
                )

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = {
                        "n_layers": params[0],
                        "n_neurons": params[1],
                        "learning_rate": round(params[2], 3),
                        "n_epochs": params[3],
                    }
                    model = NeuralNetwork(
                        n_layers=best_params["n_layers"],
                        n_inputs=len(x_train[0]),
                        n_outputs=1,
                        n_neurons=best_params["n_neurons"],
                        hidden_act="ReLU",
                        output_act="Sigmoid",
                        loss_func="BCE",
                        learning_rate=best_params["learning_rate"],
                        n_epochs=best_params["n_epochs"],
                    )
                    model.fit(x_train, y_train)
                    model.save("best")

                t.update()

    return best_params, results
