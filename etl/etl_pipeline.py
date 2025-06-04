from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data

def run_etl(input_path: str, output_path: str):
    df = extract_data(input_path)
    df_transformed = transform_data(df)
    load_data(df_transformed, output_path)