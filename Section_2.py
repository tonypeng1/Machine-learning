import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

p_dir = os.getcwd()
os.chdir('C://Users//tony3//OneDrive//Documents//Machine Learning A-Z//Part 1 - Data Preprocessing')
p_dir_new = os.getcwd()

data = pd.read_csv('Data.csv')
y = data['Purchased']
X = data.iloc[:, :-1]

X['Age'] = X['Age'].fillna(np.mean(X['Age']))
X['Salary'] = X['Salary'].fillna(np.mean(X['Salary']))

X.info()
ohe = OneHotEncoder()
x = ohe.fit_transform(X.iloc[:, [0]]).toarray()
preprocessor = make_column_transformer((OneHotEncoder(), ['Country']), remainder='passthrough')
X_1 = preprocessor.fit_transform(X)
le = LabelEncoder()
y_1 = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.2, random_state=42)
sc_X = StandardScaler()
X_train_1 = sc_X.fit_transform(X_train)
X_test_1 = sc_X.transform(X_test)
print()

