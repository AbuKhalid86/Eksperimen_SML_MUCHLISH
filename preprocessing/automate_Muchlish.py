import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_pipeline(df: pd.DataFrame):
    df = df.copy()
    df['PM10'] = df['PM10'].clip(lower=0)
    df['SO2'] = df['SO2'].clip(lower=0)
    df['Humidity'] = df['Humidity'].clip(upper=100)

    X = df.drop('Air Quality', axis=1)
    y = df['Air Quality']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    df_cleaned = pd.DataFrame(X_scaled, columns=X.columns)
    df_cleaned['Air Quality'] = y_encoded
    return df_cleaned

if __name__ == "__main__":
    df_raw = pd.read_csv('namadataset_raw/updated_pollution_dataset.csv')
    df_ready = preprocess_pipeline(df_raw)
    df_ready.to_csv('namadataset_preprocessing/updated_pollution_dataset_preprocessing.csv', index=False)

