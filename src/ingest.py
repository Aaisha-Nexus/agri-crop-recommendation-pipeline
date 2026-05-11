import pandas as pd
import os

def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "data", "raw", "Crop_recommendation.csv")
    
    df = pd.read_csv(file_path)
    print("Success! Data loaded:")
    print(df.head())
    return df

if __name__ == "__main__":
    load_data()