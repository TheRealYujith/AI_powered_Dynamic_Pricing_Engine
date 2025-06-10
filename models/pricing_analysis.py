import pandas as pd
import numpy as np
from typing import List
import joblib

def pricing_analysis(df:pd.DataFrame) -> List[dict]:
    unique_pairs = df[["Store ID", "Product ID"]].drop_duplicates()
    price_range = df['Price'].values
    price_points = np.linspace(price_range.min(), price_range.max(), 50)

    results = []

    for _, row in unique_pairs.iterrows():
        store_id = row["Store ID"]
        product_id = row["Product ID"]
        
        df_pair = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)]

        # Load model
        model_filename = f"models/model_{store_id}_{product_id}.pkl"
        try:
            model = joblib.load(model_filename)
        except FileNotFoundError:
            print(f"Model not found for Store: {store_id}, Product: {product_id}")
            continue

        # Use average of non-price features
        avg_features = df_pair.drop(columns=["Date", "Demand Forecast", "Price"]).mean()

        revenues = []
        for price in price_points:
            feature_vector = avg_features.copy()
            feature_vector["Price"] = price
            input_vector = feature_vector.values.reshape(1, -1)

            forecasted_demand = model.predict(input_vector)[0]
            revenue = forecasted_demand * price
            revenues.append((price, revenue))

        # Find price with max revenue
        optimal_price, max_revenue = max(revenues, key=lambda x: x[1])

        results.append({
            "Store ID": store_id,
            "Product ID": product_id,
            "Optimal Price": round(optimal_price, 2),
            "Expected Revenue": round(max_revenue, 2)
        })

        print(f"Store: {store_id}, Product: {product_id} -> Optimal Price: {optimal_price:.2f}, Revenue: {max_revenue:.2f}")

    return results