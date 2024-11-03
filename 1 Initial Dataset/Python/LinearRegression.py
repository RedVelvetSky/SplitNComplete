#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
from sklearn.metrics import mean_squared_error
import sklearn.model_selection
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=92, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    data = np.hstack([data, np.ones(shape=(data.shape[0], 1))])

    # TODO: Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        for i in range(0, train_data.shape[0], args.batch_size):
            batch = train_data[permutation[i:i + args.batch_size]]
            batch_target = train_target[permutation[i:i + args.batch_size]]

            predictions = batch.dot(weights)
            error = predictions - batch_target
            gradient = batch.T.dot(error) / batch.shape[0]

            # L2 regularization
            weights_no_bias = np.copy(weights)
            weights_no_bias[-1] = 0  # here we exclude bias from regularization
            gradient = gradient + args.l2 * weights_no_bias

            weights = weights - args.learning_rate * gradient

        # TODO: Append current RMSE on train/test to `train_rmses`/`test_rmses`.
        train_predictions = train_data.dot(weights)
        train_rmse = np.sqrt(mean_squared_error(train_predictions, train_target))
        train_rmses.append(train_rmse)

        # RMSE on the test set
        test_predictions = test_data.dot(weights)
        test_rmse = np.sqrt(mean_squared_error(test_predictions, test_target))
        test_rmses.append(test_rmse)

    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on `train_data` (ignoring `args.l2`).
    model = LinearRegression()
    model.fit(train_data, train_target)
    explicit_predictions = model.predict(test_data)
    explicit_rmse = np.sqrt(mean_squared_error(test_target, explicit_predictions))

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(main_args)
    print("Test RMSE: SGD {:.3f}, explicit {:.1f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.3f}".format(weight) for weight in weights[:12]), "...")