import pandas as pd
import numpy as np
import re
from keras.models import load_model

def pricing_analysis(df_path: str):
    
    df = pd.read_csv(df_path)
    unique_pairs = df[["Store ID", "Product ID"]].drop_duplicates()
    price_range = df['Price'].values
    price_points = np.linspace(price_range.min(), price_range.max(), 50)

    results = []

    for _, row in unique_pairs.iterrows():
        store_id = row["Store ID"]
        product_id = row["Product ID"]
        
        df_pair = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)]

        # Load model
        model_filename = f"models_lstm/model_{store_id}_{product_id}.keras"
        try:
            model = load_model(model_filename)
        except OSError:
            print(f"Model not found for Store: {store_id}, Product: {product_id}")
            continue

        # Use average of non-price features
        avg_features = df_pair.drop(columns=["Date", "Units Sold", "Demand Forecast", "Price", "Revenue"]).mean()

        revenues = []
        for price in price_points:
            feature_vector = avg_features.copy()
            feature_vector["Price"] = price
            input_vector = feature_vector.values.reshape(1, -1)

            forecasted_demand = model.predict(input_vector)[0][0]
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

    return pd.DataFrame(results)

def search_optimal_results(df: pd.DataFrame, store_id: str, product_id: int):
    
    # Extract the numerical part of the dataset
    store_id = int(re.sub(r"\D", "", store_id))
    product_id = int(re.sub(r"\D", "", product_id))
    
    id_combination = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)]
    
    if id_combination.empty:
        print(f"No results found for Store ID {store_id}, Product ID {product_id}.")
        return None
    
    row = id_combination.iloc[0]
    
    return {
        "Store ID": row["Store ID"],
        "Product ID": row["Product ID"],
        "Optimal Price": row["Optimal Price"],
        "Expected Revenue": row["Expected Revenue"]
    }

    