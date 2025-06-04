import pandas as pd

def extract_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["Date"])
    return df