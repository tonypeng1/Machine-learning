import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
print()

