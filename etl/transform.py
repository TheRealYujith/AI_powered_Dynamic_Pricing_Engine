import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates/NaN
    df = df.dropna()
    
    # Create revenue column
    df['Revenue'] = df['Price'] * df['Units Sold']
    
    # Convert categorical features
    df = pd.get_dummies(df, columns=['Weather Condition', 'Category', 'Region', 'Seasonality'], drop_first=True, dtype=int)

    # Add lag features
    df = df.sort_values(by=["Product ID", "Date"])
    df['Prev_Day_Sales'] = df.groupby('Product ID')['Units Sold'].shift(1)
    df['Prev_Day_Reorders'] = df.groupby('Product ID')['Units Ordered'].shift(1)
    df['Prev_Day_Inventory'] = df.groupby('Product ID')['Inventory Level'].shift(1)
    df['Prev_Day_Demand_Forecast'] = df.groupby('Product ID')['Demand Forecast'].shift(1)
    df['Prev_Day_Price'] = df.groupby('Product ID')['Price'].shift(1)
    df['Prev_Day_Discount'] = df.groupby('Product ID')['Discount'].shift(1)
    
    return df.dropna()