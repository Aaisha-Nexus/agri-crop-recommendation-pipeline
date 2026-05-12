import pandas as pd

def clean_data():
    df = pd.read_csv("data/raw/Crop_recommendation.csv")

    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    df.to_csv("data/processed/crop_clean.csv", index=False)

    print("Cleaned data saved successfully!")
    print(df.head())

if __name__ == "__main__":
    clean_data()