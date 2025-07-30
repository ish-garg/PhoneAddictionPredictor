import pandas as pd

path = 'teen_phone_addiction_dataset.csv'

df = pd.read_csv(path)

import matplotlib.pyplot as plt

#plt.scatter(df['Name'],df['Gender'])

df = df.iloc[ : , 1 : ]

df = df.iloc[ : , 1 : ]

#df.head()

#df.shape()

#df.head()

#df.shape

X, y  = df.iloc[ : , 0 : 22], df.iloc[ : , -1]

#X

#y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size = 0.9)

#y_train

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

#clf.fit(X_train, y_train)

#c = ['Gender', 'Location', 'School_Grade', 'Phone_Usage_Purpose']

#X_train = X_train.drop(columns=c)

#X_Test = X_test.drop(columns = c)

combined = pd.concat([X_train, X_test])
combined = pd.get_dummies(combined, drop_first=True)  

# Split back to train and test
X_train = combined.iloc[:len(X_train), :]
X_test = combined.iloc[len(X_train):, :]


clf.fit(X_train, y_train)

#clf.predict(X_test)

#X_test = X_test.drop(columns = c)

clf.predict(X_test)

#from sklearn.metrics import accuracy_score

#accuracy_score(y_test, clf.predict(X_test))

from sklearn.metrics import mean_absolute_error

predictions = clf.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")

import joblib
joblib.dump(clf, "model.pkl")