import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from keras.models import Sequential  
from keras import Input
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from typing import List
import os
import shap
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, -1])  # Last column is "Demand Forecast" (target variable)
    return np.array(X), np.array(y)

def compute_global_shap_importance(model, X_train, X_test, feature_names: List[str], store_id: str, product_id: str) -> pd.DataFrame:
    
    # Extract original shape details
    n_samples, timesteps, n_features = X_train.shape

    # Flatten inputs for SHAP
    X_train_flat = X_train.reshape((n_samples, timesteps * n_features))
    X_test_flat = X_test.reshape((X_test.shape[0], timesteps * n_features))

    # Define wrapper that reshapes input back to 3D for LSTM model
    def model_predict(X_flat):
        X_reshaped = X_flat.reshape((X_flat.shape[0], timesteps, n_features))
        return model.predict(X_reshaped)

    # Reduce background size for performance
    background = shap.sample(X_train_flat, 50)

    # Initialize SHAP KernelExplainer
    explainer = shap.KernelExplainer(model_predict, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_flat[:100])

    # Generate flattened feature names like feature_0_t0, ..., feature_25_t9
    flat_feature_names = [f"{f}_t{t}" for t in range(timesteps) for f in feature_names]

    # Aggregate SHAP values
    mean_shap = np.mean(np.abs(shap_values[0]), axis=0)

    # Build dataframe
    actual_n_flat_features = X_test_flat.shape[1]
    flat_feature_names = [f"feature_{i}" for i in range(actual_n_flat_features)]
    n_flat_features = len(flat_feature_names) 

    feature_importance = pd.DataFrame({
        "Feature": flat_feature_names,
        "Importance": mean_shap,
        "Store ID": [store_id] * n_flat_features,
        "Product ID": [product_id] * n_flat_features
    })

    return feature_importance


def forecast_demand (df_path: str, seq_length: int) -> pd.DataFrame:
 
    # Convert Date to datetime - format is yyyy/mm/dd
    df = pd.read_csv(df_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    # Extract useful time-based features
    df["DayOfWeek"] = df["Date"].dt.weekday # Monday = 0, Tuesday = 1, ..., 6 = Sunday
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Get unique (store, product) pairs
    unique_pairs = df[["Store ID", "Product ID"]].drop_duplicates()
    
    os.makedirs("models_lstm", exist_ok=True)
    overall_metrics = []
    all_shap_importances = []

    for _, row in unique_pairs.iterrows():
        store_id = row["Store ID"]
        product_id = row["Product ID"]
        model_path = f"models_lstm/model_{store_id}_{product_id}.keras"
        
        # Check if model already exists
        if os.path.exists(model_path):
            print(f"Model already exists for Store: {store_id}, Product: {product_id}. Skipping training.")
            continue

        # Filter data for this product-store pair
        df_pair = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)].copy() # Create a subset of data
        
        # Order the features for time series forecasting
        df_pair = df_pair.sort_values("Date")

        # Drop unused columns
        df_pair = df_pair.drop(columns=["Date"])
        features = df_pair.columns.drop("Demand Forecast")
        
        # Normalize features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_pair[features.tolist() + ["Demand Forecast"]])

        # Create sequences
        X, y = create_sequences(scaled_data, seq_length)
        
        # Split data into train and test
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Define LSTM model
        model = Sequential()
        model.add(Input(shape=(seq_length, X.shape[2])))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=16,
            verbose=0,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
        )

        # Evaluation
        y_pred = model.predict(X_test).flatten()
        rmse = root_mean_squared_error(y_test, y_pred)
        overall_metrics.append((store_id, product_id, rmse))
        
        # Save the model
        model.save(f"models_lstm/model_{store_id}_{product_id}.keras")

        print(f"Trained model for Store: {store_id}, Product: {product_id}, RMSE: {rmse:.2f}")
        
        # Computing feature importance using SHAP
        shap_df = compute_global_shap_importance(model, X_train, X_test, features.tolist(), store_id, product_id)
        all_shap_importances.append(shap_df)

    print(f"\nTotal models trained: {len(overall_metrics)}")
    
    return pd.DataFrame(overall_metrics), all_shap_importances
    
def summarize_and_plot_feature_importance(all_importances: list[pd.DataFrame], output_plot_path="shap_summary.png"):
    
    # Concatenate all SHAP importance DataFrames
    combined_df = pd.concat(all_importances, ignore_index=True)

    # Group and average by feature
    global_importance = (combined_df.groupby("Feature")["Importance"].mean().reset_index())
    global_importance = global_importance.sort_values(by="Importance", ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(global_importance["Feature"], global_importance["Importance"], color='skyblue')
    plt.xlabel("Average SHAP Value (Importance)")
    plt.title("Global Feature Importance Across All Store-Product Pairs")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()

    return global_importance