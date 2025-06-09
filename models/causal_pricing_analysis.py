import pandas as pd
import numpy as np
from typing import List
import joblib

def causal_pricing_analysis(df:pd.DataFrame) -> List[dict]:
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

    return results