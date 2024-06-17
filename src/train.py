import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, clear_output
from data import Data, train_test_split
from neural_network import NeuralNetwork
from numpy import ndarray

# Assuming Data and train_test_split are already imported from your project.

data = Data("./training_data/data.csv", drop_columns=["id"])
x_train, y_train, x_test, y_test = train_test_split(data.x, data.y, split=0.3)


def evaluate_model(trained_model: NeuralNetwork, x: ndarray, y_true: ndarray):
    """
    Evaluate the model on the test data.
    """
    y_pred = trained_model.forward(x)
    loss = trained_model.loss_func.forward(y_pred, y_true)
    return loss


def tune_hyperparams(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    n_layers_opts: list[int] = [2, 3, 4],
    n_neurons_opts: list[int] = [4, 8, 16],
    learning_rate_opts: ndarray = np.arange(0.005, 0.105, 0.005),
    n_epochs_opts: list[int] = range(100, 200, 10),
) -> dict:
    best_loss = float("inf")
    best_params = {}
    results = []

    plt.ion()  # Turn on interactive plotting

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))  # Single plot for best loss
    best_losses = []

    for n_layers in n_layers_opts:
        for n_neurons in n_neurons_opts:
            for learning_rate in learning_rate_opts:
                for n_epochs in n_epochs_opts:
                    print(
                        f"Training with: layers={n_layers}, neurons={n_neurons}, lr={learning_rate}, epochs={n_epochs}"
                    )
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
                    results.append((n_layers, n_neurons, learning_rate, n_epochs, test_loss))

                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_params = {
                            "n_layers": n_layers,
                            "n_neurons": n_neurons,
                            "learning_rate": learning_rate,
                            "n_epochs": n_epochs,
                        }
                        print("New best params:", best_params)
                        print("New best test loss:", test_loss)

                    best_losses.append(best_loss)

                    # Update the plot
                    ax.clear()
                    ax.plot(best_losses, label="Best Loss")
                    ax.set_title("Best Loss Over Time")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Loss")
                    ax.legend()

                    # Annotate the best parameters
                    textstr = "\n".join(
                        (
                            f"Best:n_neurons: {best_params['n_neurons']}",
                            f"        n_layers: {best_params['n_layers']}",
                            f"        learning_rate: {best_params['learning_rate']:.3f}",
                            f"        n_epochs: {best_params['n_epochs']}",
                        )
                    )
                    ax.text(-0.15, 1.15, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top")
                    textstr = "\n".join(
                        (
                            f"Current:n_neurons: {n_neurons}",
                            f"             n_layers: {n_layers}",
                            f"             learning_rate: {learning_rate}",
                            f"             n_epochs: {n_epochs}",
                        )
                    )
                    ax.text(0.8, 1.15, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top")
                    plt.draw()
                    plt.pause(0.001)  # Pause to update the plot

    plt.ioff()  # Turn off interactive plotting
    plt.show()

    return best_params, results


# Run the hyperparameter tuning
best_params, results = tune_hyperparams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

# Train the best model
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
test_loss = evaluate_model(model, x_test, y_test)
print("Test Loss:", test_loss)

# Plotting the results
results_df = pd.DataFrame(results, columns=["n_layers", "n_neurons", "learning_rate", "n_epochs", "test_loss"])


# 1. Hyperparameter Optimization Surface Plots
def plot_3d_surface(df, x_param, y_param, z_param):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(df[x_param], df[y_param], df[z_param], cmap="RdYlGn", edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(z_param)
    plt.title(f"Surface plot of {z_param} vs {x_param} and {y_param}")
    plt.show()


plot_3d_surface(results_df, "learning_rate", "n_neurons", "test_loss")

# 2. Parallel Coordinates Plot
from pandas.plotting import parallel_coordinates

results_df_normalized = (results_df - results_df.min()) / (results_df.max() - results_df.min())
parallel_coordinates(results_df_normalized, class_column="test_loss", colormap="RdYlGn")
plt.title("Parallel Coordinates Plot for Hyperparameter Tuning")
plt.show()

# 3. Scatter Plots with Color Encoding
import seaborn as sns

sns.pairplot(results_df, vars=["n_layers", "n_neurons", "learning_rate", "n_epochs"], hue="test_loss", palette="RdYlGn")
plt.suptitle("Pair Plot with Performance Encoding", y=1.02)
plt.show()


# 4. Heatmaps
def plot_heatmap(df, x_param, y_param, z_param):
    pivot_table = df.pivot_table(index=y_param, columns=x_param, values=z_param)
    sns.heatmap(pivot_table, annot=False, fmt=".3f", cmap="RdYlGn", cbar_kws={"label": z_param})
    plt.title(f"Heatmap of {z_param} for {x_param} vs {y_param}")
    plt.show()


plot_heatmap(results_df, "n_layers", "n_neurons", "test_loss")
plot_heatmap(results_df, "learning_rate", "n_epochs", "test_loss")
