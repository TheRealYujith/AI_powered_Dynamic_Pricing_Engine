import pandas as pd

def load_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)