import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import LinearRegression

def feature_engineering(x_data, item_names):
    item_indices = {name: i for i, name in enumerate(item_names)}
    x_data_enhanced = x_data.copy()
    wind_direc_idx = item_indices.get('WIND_DIREC')
    if wind_direc_idx is not None:
        wind_direc_features = x_data[:, wind_direc_idx::18]
        wind_direc_rad = wind_direc_features * np.pi / 180.
        wind_sin = np.sin(wind_direc_rad)
        wind_cos = np.cos(wind_direc_rad)
        x_data_enhanced = np.concatenate([x_data_enhanced, wind_sin, wind_cos], axis=1)
    pm25_idx = item_indices.get('PM2.5')
    pm10_idx = item_indices.get('PM10')
    new_poly_features = []
    if pm25_idx is not None:
        pm25_features = x_data[:, pm25_idx::18]
        new_poly_features.append(pm25_features**2)
    if pm10_idx is not None:
        pm10_features = x_data[:, pm10_idx::18]
        new_poly_features.append(pm10_features**2)
    
    if new_poly_features:
        x_data_enhanced = np.concatenate([x_data_enhanced] + new_poly_features, axis=1)
        
    return x_data_enhanced

def plot_learning_curve(model):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(model.training_loss)), model.training_loss, label='Training Loss (RMSE)')
    plt.title('Final Model Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve_final.png')
    plt.show()

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
    pm25_index = np.where(item_names == 'PM2.5')[0][0]
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

def main():
    train_df = pd.read_csv('.././dataset/train.csv', encoding='big5')
    test_df = pd.read_csv('.././dataset/test.csv', header=None, encoding='big5')
    
    print("Preprocessing all data...")
    x_data, y_data, item_names = preprocess_train_data(train_df)
    x_test_raw = preprocess_test_data(test_df)

    print("Performing feature engineering...")
    x_data_featured = feature_engineering(x_data, item_names)
    x_test_featured = feature_engineering(x_test_raw, item_names)
    
    # --- Final Training on ALL Data ---
    print("\n--- Training Final Model on 100% of the training data ---")
    mean_full = np.mean(x_data_featured, axis=0)
    std_full = np.std(x_data_featured, axis=0)
    std_full[std_full == 0] = 1

    x_train_full_norm = (x_data_featured - mean_full) / std_full
    x_test_full_norm = (x_test_featured - mean_full) / std_full
    
    final_model = LinearRegression(learning_rate=0.5, n_iterations=500000, 
                                   regularization=0.005, optimizer='adagrad')
    final_model.fit(x_train_full_norm, y_data) 

    # --- Prediction ---
    print("\nMaking predictions with the final model...")
    predictions = final_model.predict(x_test_full_norm)
    
    submission = pd.DataFrame({
        'index': ['index_' + str(i) for i in range(len(predictions))],
        'answer': predictions.flatten()
    })
    submission.to_csv('submission_final.csv', index=False)
    print("Submission file 'submission_final.csv' created successfully!")
    plot_learning_curve(final_model)

if __name__ == '__main__':
    main()