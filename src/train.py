import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("data/auto-mpg.csv", sep=';')

# print(data)

data = data.sample(frac=1)

X = data.drop(['mpg'], axis=1)
y = data['mpg']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor = regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

file_to_write = open('data/models/linreg_lr.pickle', 'wb')
pickle.dump(regressor, file_to_write)


# print(y_pred)



