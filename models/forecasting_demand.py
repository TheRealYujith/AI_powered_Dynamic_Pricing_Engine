import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def forecast_demand (df: pd.DataFrame):

    # Get unique (store, product) pairs
    unique_pairs = df[["Store ID", "Product ID"]].drop_duplicates()

    overall_metrics = []

    for _, row in unique_pairs.iterrows():
        store_id = row["Store ID"]
        product_id = row["Product ID"]

        # Filter data for this product-store pair
        df_pair = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)]

        features = df_pair.columns.drop(["Date", "Demand Forecast"])
        X = df_pair[features]
        y = df_pair["Demand Forecast"]

        # Split and train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate and save
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        overall_metrics.append((store_id, product_id, rmse))

        model_filename = f"models/model_{store_id}_{product_id}.pkl"
        joblib.dump(model, model_filename)

        print(f"Trained model for Store: {store_id}, Product: {product_id}, RMSE: {rmse:.2f}")

    print(f"\nTotal models trained: {len(overall_metrics)}")
    
    return overall_metrics