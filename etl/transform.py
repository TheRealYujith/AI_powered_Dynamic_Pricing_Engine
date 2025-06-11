import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates/NaN
    df = df.dropna()
    
    # Create revenue column
    df['Revenue'] = df['Price'] * df['Units Sold']
    
    # Convert categorical features into integers
    df["Product ID"] = df["Product ID"].str.extract(r'(\d+)$').astype(int) # str.extract(r'(\d+)$') - extracts the numeric part at the end of the string
    df["Store ID"] = df["Store ID"].str.extract(r'(\d+)$').astype(int)
    
    # Each unique value is replaced with an integer
    for col in ['Weather Condition', 'Category', 'Region', 'Seasonality']:
        df[col], _ = pd.factorize(df[col])

    # Add lag features
    df = df.sort_values(by=["Product ID", "Date"])
    df['Prev_Day_Sales'] = df.groupby('Product ID')['Units Sold'].shift(1)
    df['Prev_Day_Reorders'] = df.groupby('Product ID')['Units Ordered'].shift(1)
    df['Prev_Day_Inventory'] = df.groupby('Product ID')['Inventory Level'].shift(1)
    df['Prev_Day_Demand_Forecast'] = df.groupby('Product ID')['Demand Forecast'].shift(1)
    df['Prev_Day_Price'] = df.groupby('Product ID')['Price'].shift(1)
    df['Prev_Day_Discount'] = df.groupby('Product ID')['Discount'].shift(1)
    
    return df.dropna()