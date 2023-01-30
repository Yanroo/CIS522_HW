import numpy as np


class LinearRegression:
    """
    A linear regression model that uses close function to solve the model.
    """
    w: np.ndarray
    b: float

    def __init__(self):
        """
        Initialize the weight and bias.
        """
        self.w = np.array([])
        self.b = np.random.randn(1,)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the training data and compute weight.

        Arguments:
            X (np.ndarray): The input features.
            y (np.ndarray): The input label.

        Returns:
            None

        """
        self.w = np.linalg.inv(X.T@X)@X.T@y
        # self.w = params[0:7]
        # self.b = params[7]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        
        y_hat = X @ self.w
        return y_hat


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        
        self, X: np.ndarray, y: np.ndarray, lr: float = 1e-8, epochs: int = 6000
    ) -> None:
        """
        Fit the training data and compute weight.

        Arguments:
            X (np.ndarray): The input features.
            y (np.ndarray): The input label.
            lr (float):     The learning rate.
            epochs (int):   The total training epochs.

        Returns:
            None

        """

        self.w = np.random.randn(X.shape[1],)
        self.b = np.random.randn(1,)
        
        for epoch in range(epochs):
            y_hat = X @ self.w + self.b
            # print("y_hat: ", y_hat)
            diff = y-y_hat
            N = X.shape[0]
            dw = -2 * (X.T @ diff)/N
            # print("dw: ", dw)
            self.w = self.w - lr * dw
            self.b = self.b - 2 * lr * np.sum(diff)/N
        # print("y_hat: ", y_hat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b


