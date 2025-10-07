
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

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

def feature_engineering(x_data, item_names):
    item_indices = {name: i for i, name in enumerate(item_names)}
    x_data_enhanced = x_data.copy()
    wind_direc_idx = item_indices.get('WIND_DIREC')
    if wind_direc_idx is not None:
        wind_direc_features = x_data[:, wind_direc_idx::18]
        wind_direc_rad = wind_direc_features * np.pi / 180.0
        wind_sin = np.sin(wind_direc_rad)
        wind_cos = np.cos(wind_direc_rad)
        x_data_enhanced = np.concatenate([x_data_enhanced, wind_sin, wind_cos], axis=1)
    pm25_idx = item_indices.get('PM2.5')
    pm10_idx = item_indices.get('PM10')
    new_poly_features = []
    if pm25_idx is not None:
        pm25_features = x_data[:, pm25_idx::18]
        new_poly_features.append(pm25_features ** 2)
    if pm10_idx is not None:
        pm10_features = x_data[:, pm10_idx::18]
        new_poly_features.append(pm10_features ** 2)
    if new_poly_features:
        x_data_enhanced = np.concatenate([x_data_enhanced] + new_poly_features, axis=1)
    return x_data_enhanced

def plot_learning_curve(model, title='Final Model Learning Curve', path='learning_curve_final.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(model.training_loss)), model.training_loss, label='Training RMSE')
    if len(model.validation_loss) > 0:
        plt.plot(range(len(model.validation_loss)), model.validation_loss, label='Validation RMSE')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def preprocess_train_data(raw_df):
    df = raw_df.drop(columns=['Location', 'Date'])
    df['ItemName'] = df['ItemName'].str.strip()
    df = df.replace(['#', '*', 'x', 'A'], np.nan)
    item_name_col = df['ItemName']
    data_cols = df.columns[1:]
    df[data_cols] = df[data_cols].apply(pd.to_numeric, errors='coerce')
    df['ItemName'] = item_name_col
    df.loc[df['ItemName'] == 'RAINFALL', data_cols] = df.loc[df['ItemName'] == 'RAINFALL', data_cols].fillna(0)
    df[data_cols] = df[data_cols].ffill(axis=1).bfill(axis=1)
    if df[data_cols].isnull().values.any():
        df[data_cols] = df[data_cols].fillna(0)
    x_data, y_data = [], []
    n_days = df.shape[0] // 18
    reshaped_data = df[data_cols].values.reshape(n_days, 18, 24)
    item_names = df['ItemName'].unique()
    pm25_index_arr = np.where(item_names == 'PM2.5')[0]
    if len(pm25_index_arr) == 0:
        raise ValueError('PM2.5 not found in ItemName list.')
    pm25_index = pm25_index_arr[0]
    for day in range(n_days):
        for start_hour in range(24 - 9):
            x_sample = reshaped_data[day, :, start_hour:start_hour+9].flatten()
            y_sample = reshaped_data[day, pm25_index, start_hour+9]
            x_data.append(x_sample)
            y_data.append(y_sample)
    x_data_np = np.array(x_data)
    y_data_np = np.array(y_data).reshape(-1, 1)
    valid_indices = ~np.isnan(y_data_np).any(axis=1)
    x_data_clean = x_data_np[valid_indices]
    y_data_clean = y_data_np[valid_indices]
    return x_data_clean, y_data_clean, item_names

def preprocess_test_data(raw_df):
    raw_df.columns = ['index', 'ItemName'] + [f'h{i}' for i in range(9)]
    df = raw_df.copy()
    df['ItemName'] = df['ItemName'].str.strip()
    data_cols = df.columns[2:]
    df[data_cols] = df[data_cols].apply(pd.to_numeric, errors='coerce')
    df.loc[df['ItemName'] == 'RAINFALL', data_cols] = df.loc[df['ItemName'] == 'RAINFALL', data_cols].fillna(0)
    df[data_cols] = df[data_cols].ffill(axis=1).bfill(axis=1)
    if df[data_cols].isnull().values.any():
        df[data_cols] = df[data_cols].fillna(0)
    x_test = []
    for i in range(0, df.shape[0], 18):
        sample_data = df.iloc[i:i+18]
        features = sample_data.iloc[:, 2:].values.flatten()
        x_test.append(features)
    return np.array(x_test)

def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def normalize(train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)
    std[std == 0] = 1
    train_n = (train_X - mean) / std
    val_n = (val_X - mean) / std
    test_n = None if test_X is None else (test_X - mean) / std
    return train_n, val_n, test_n, mean, std

def visualize_feature_groups(weights: np.ndarray, item_names: np.ndarray, save_path: str = 'feature_groups_importance.png'):
    groups = []
    for name in item_names:
        groups.append(name)
    groups += ['WIND_SIN', 'WIND_COS', 'PM2.5^2', 'PM10^2']
    n_core = len(item_names)
    n_per_item = 9
    core_len = n_core * n_per_item
    w = weights.flatten()
    agg = []
    for i in range(n_core):
        idxs = np.arange(i * n_per_item, (i + 1) * n_per_item)
        agg.append(np.sum(np.abs(w[idxs])))
    wind_sin_cos_len = 0
    if len(w) > core_len:
        rem = len(w) - core_len
        chunk = rem // 4
        if chunk > 0:
            sin_idxs = np.arange(core_len, core_len + chunk)
            cos_idxs = np.arange(core_len + chunk, core_len + 2 * chunk)
            pm25_2_idxs = np.arange(core_len + 2 * chunk, core_len + 3 * chunk)
            pm10_2_idxs = np.arange(core_len + 3 * chunk, core_len + 4 * chunk)
            agg += [np.sum(np.abs(w[sin_idxs])), np.sum(np.abs(w[cos_idxs])), np.sum(np.abs(w[pm25_2_idxs])), np.sum(np.abs(w[pm10_2_idxs]))]
            wind_sin_cos_len = 4 * chunk
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(agg)), agg)
    plt.xticks(range(len(agg)), groups, rotation=45, ha='right')
    plt.ylabel('Sum of |weights|')
    plt.title('Feature Group Contribution')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_data_fraction_effect(X: np.ndarray, y: np.ndarray, item_names: np.ndarray, lr: float, iters: int, reg: float, optimizer: str, save_path: str = 'data_fraction_effect.png'):
    X_tr, X_val, y_tr, y_val = train_val_split(X, y, val_ratio=0.2, seed=42)
    X_tr_n, X_val_n, _, _, _ = normalize(X_tr, X_val)
    fracs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rmses = []
    for f in fracs:
        n_use = max(10, int(len(X_tr_n) * f))
        X_use = X_tr_n[:n_use]
        y_use = y_tr[:n_use]
        m = LinearRegression(learning_rate=lr, n_iterations=iters, regularization=reg, optimizer=optimizer)
        m.fit(X_use, y_use, X_val_n, y_val)
        y_pred = m.predict(X_val_n)
        rmses.append(np.sqrt(np.mean((y_pred - y_val) ** 2)))
    plt.figure(figsize=(8, 5))
    plt.plot([int(x*100) for x in fracs], [float(r) for r in rmses], marker='o')
    plt.xlabel('Training data used (%)')
    plt.ylabel('Validation RMSE')
    plt.title('Impact of Training Data Amount on PM2.5 Prediction')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_regularization_effect(X: np.ndarray, y: np.ndarray, lr: float, iters: int, optimizer: str, lambdas: List[float], save_path: str = 'regularization_effect.png'):
    X_tr, X_val, y_tr, y_val = train_val_split(X, y, val_ratio=0.2, seed=42)
    X_tr_n, X_val_n, _, _, _ = normalize(X_tr, X_val)
    rmses = []
    for lam in lambdas:
        m = LinearRegression(learning_rate=lr, n_iterations=iters, regularization=lam, optimizer=optimizer)
        m.fit(X_tr_n, y_tr, X_val_n, y_val)
        y_pred = m.predict(X_val_n)
        rmses.append(np.sqrt(np.mean((y_pred - y_val) ** 2)))
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, [float(r) for r in rmses], marker='o')
    plt.xscale('log')
    plt.xlabel('Regularization λ')
    plt.ylabel('Validation RMSE')
    plt.title('Impact of Regularization on PM2.5 Prediction')
    plt.grid(True, which='both')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    train_path = 'train.csv'
    test_path = 'test.csv'
    train_df = pd.read_csv(train_path, encoding='big5')
    test_df = pd.read_csv(test_path, header=None, encoding='big5')
    x_data, y_data, item_names = preprocess_train_data(train_df)
    x_test_raw = preprocess_test_data(test_df)
    x_data_featured = feature_engineering(x_data, item_names)
    x_test_featured = feature_engineering(x_test_raw, item_names)
    mean_full = np.mean(x_data_featured, axis=0)
    std_full = np.std(x_data_featured, axis=0)
    std_full[std_full == 0] = 1
    x_train_full_norm = (x_data_featured - mean_full) / std_full
    x_test_full_norm = (x_test_featured - mean_full) / std_full
    final_model = LinearRegression(learning_rate=0.5, n_iterations=500000, regularization=0.005, optimizer='adagrad')
    final_model.fit(x_train_full_norm, y_data)
    predictions = final_model.predict(x_test_full_norm)
    submission = pd.DataFrame({'index': ['index_' + str(i) for i in range(len(predictions))], 'answer': predictions.flatten()})
    submission.to_csv('submission_final.csv', index=False)
    plot_learning_curve(final_model, title='Final Model Learning Curve', path='learning_curve_final.png')
    visualize_feature_groups(final_model.weights, item_names, save_path='feature_groups_importance.png')
    plot_data_fraction_effect(x_data_featured, y_data, item_names, lr=0.5, iters=2000, reg=0.005, optimizer='adagrad', save_path='data_fraction_effect.png')
    lambdas = [0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    plot_regularization_effect(x_data_featured, y_data, lr=0.5, iters=100000, optimizer='adagrad', lambdas=lambdas, save_path='regularization_effect.png')

if __name__ == '__main__':
    main()
