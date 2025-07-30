import pandas as pd
import joblib

path = 'teen_phone_addiction_dataset.csv'
df = pd.read_csv(path)

df = df.iloc[:, 1:] 
df = df.iloc[:, 1:]  

X, y = df.iloc[:, 0:22], df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

combined = pd.concat([X_train, X_test])
combined_encoded = pd.get_dummies(combined, drop_first=True)

feature_names = list(combined_encoded.columns)
joblib.dump(feature_names, "feature_names.pkl")

print("Feature names saved to 'feature_names.pkl'")
print("Features:", feature_names)