import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def causal_pricing_analysis(df:pd.DataFrame) -> dict:
    
    # Prepare data for causal model
    features = [col for col in df.columns if col not in ["Units Sold", "Price", "Revenue", "Date"]]
    X = df[features]
    T = df['Price'].values
    Y = df['Units Sold'].values

    # Splitting data into train-test split (test split isnt needed as the goal isnt predictive accuracy but to estimate the casual effect)
    X_train, _, T_train, _, Y_train, _ = train_test_split(X, T, Y, test_size=0.2, random_state=42) 

    # Scale covariates
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit Causal Forest
    causal_forest = CausalForestDML(
        model_t=RandomForestRegressor(n_estimators=100, max_depth=5),
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
        discrete_treatment=False,
        random_state=42
    )
    causal_forest.fit(Y_train, T_train, X=X_train_scaled)
    
    # Simulate expected revenue across price range
    price_range = np.linspace(T.min(), T.max(), 50) # Creates 50 equally spaced points - controls the granularity of the pricing simulation (balances tradeoff between resolution and computation time)
    avg_X = X.mean().values.reshape(1, -1)
    avg_X_scaled = scaler.transform(avg_X)

    predicted_demand = [causal_forest.model_cate(T0=np.mean(T), T1=p, X=avg_X_scaled)[0] + np.mean(Y)
                        for p in price_range]
    expected_revenue = price_range * predicted_demand

    # Identify optimal price
    optimal_index = np.argmax(expected_revenue)
    optimal_price = price_range[optimal_index]
    optimal_revenue = expected_revenue[optimal_index]
    
    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(price_range, expected_revenue, label='Expected Revenue')
    # plt.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price:.2f}')
    # plt.title('Expected Revenue vs Price')
    # plt.xlabel('Price')
    # plt.ylabel('Expected Revenue')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # print(f"Optimal Price: ${optimal_price:.2f}")
    # print(f"Expected Revenue at Optimal Price: ${optimal_revenue:.2f}")
    
    return {
        "Optimal Price": round(optimal_price, 2),
        "Expected Revenue": round(optimal_revenue, 2)
    }
    
def analyze_all_products(data: pd.DataFrame) -> list:
    results = []

    grouped = data.groupby(["Store ID", "Product ID"])

    for (store_id, product_id), group_df in grouped:
        print(f"Processing Store {store_id} - Product {product_id}")
        result = causal_pricing_analysis(group_df)

        if result:
            result.update({
                "Store ID": store_id,
                "Product ID": product_id
            })
            results.append(result)

    return results

