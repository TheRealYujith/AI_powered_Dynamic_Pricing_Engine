import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import joblib
import os

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, -1])  # Last column is "Demand Forecast" (target variable)
    return np.array(X), np.array(y)

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

    for _, row in unique_pairs.iterrows():
        store_id = row["Store ID"]
        product_id = row["Product ID"]

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
        model.add(LSTM(50, activation='relu', input_shape=(seq_length, X.shape[2])))
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
        model.save(f"models_lstm/model_{store_id}_{product_id}.h5")
        joblib.dump(scaler, f"models_lstm/scaler_{store_id}_{product_id}.pkl")

        print(f"Trained model for Store: {store_id}, Product: {product_id}, RMSE: {rmse:.2f}")

    print(f"\nTotal models trained: {len(overall_metrics)}")
    
    return pd.DataFrame(overall_metrics)