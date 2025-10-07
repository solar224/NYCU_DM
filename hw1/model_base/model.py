import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0, optimizer='gd'):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.lambda_ = regularization
        self.optimizer = optimizer
        self.weights = None
        self.bias = None
        self.training_loss = []
        self.validation_loss = []
        self.adagrad_sum_w = None
        self.adagrad_sum_b = None

    def _calculate_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        if self.optimizer == 'adagrad':
            self.adagrad_sum_w = np.zeros((n_features, 1))
            self.adagrad_sum_b = 0

        for i in range(self.n_iters):
            y_train_pred = np.dot(X_train, self.weights) + self.bias
            
            dw = (1 / n_samples) * (np.dot(X_train.T, (y_train_pred - y_train)) + 2 * self.lambda_ * self.weights)
            db = (1 / n_samples) * np.sum(y_train_pred - y_train)
            
            if self.optimizer == 'adagrad':
                self.adagrad_sum_w += dw**2
                self.adagrad_sum_b += db**2
                ada_lr_w = self.lr / np.sqrt(self.adagrad_sum_w + 1e-8)
                ada_lr_b = self.lr / np.sqrt(self.adagrad_sum_b + 1e-8)
                self.weights -= ada_lr_w * dw
                self.bias -= ada_lr_b * db
            else: 
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            
            train_rmse = self._calculate_rmse(y_train, y_train_pred)
            self.training_loss.append(train_rmse)
            
            if X_val is not None and y_val is not None:
                y_val_pred = np.dot(X_val, self.weights) + self.bias
                val_rmse = self._calculate_rmse(y_val, y_val_pred)
                self.validation_loss.append(val_rmse)

            if (i+1) % 100 == 0:
                log_message = f"Iteration {i+1}/{self.n_iters}, Training RMSE: {train_rmse:.4f}"
                if X_val is not None:
                    log_message += f", Validation RMSE: {val_rmse:.4f}"
                print(log_message)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias