import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset_raw.csv")

df["PM10"] = df["PM10"].clip(lower=0)
df["SO2"] = df["SO2"].clip(lower=0)
df["Humidity"] = df["Humidity"].clip(upper=100)

X = df.drop("Air Quality", axis=1)
y = df["Air Quality"]

X_scaled = StandardScaler().fit_transform(X)
y_encoded = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

pd.DataFrame(X_train, columns=X.columns).assign(target=y_train).to_csv("train_data_scaled.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).assign(target=y_test).to_csv("test_data_scaled.csv", index=False)

